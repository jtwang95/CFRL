import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
from .utils import glogger
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            return False

        if self.mode == "min":
            delta = val_loss - self.best_score
        elif self.mode == "max":
            delta = self.best_score - val_loss

        if delta < -self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


# Define your neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims):
        super(NeuralNet, self).__init__()
        nn_dims = [in_dim] + hidden_dims + [out_dim]
        modules = []
        for i in range(len(nn_dims) - 1):
            if i == len(nn_dims) - 2:
                modules.append(nn.Sequential(nn.Linear(nn_dims[i], nn_dims[i + 1])))
            else:
                modules.append(
                    nn.Sequential(nn.Linear(nn_dims[i], nn_dims[i + 1]), nn.ReLU())
                )
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class NeuralNetRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, is_standarized=False):
        super(NeuralNetRegressor, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.is_standarized = is_standarized
        self.model = NeuralNet(in_dim, out_dim, hidden_dims)

    @staticmethod
    def standardize(x, mean=None, std=None):
        if mean is None and std is None:
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            return (x - mean) / std, mean, std
        else:
            return (x - mean) / std

    @staticmethod
    def destandardize(x, mean, std):
        return x * std + mean

    def fit(
        self,
        X,
        y,
        epochs,
        learning_rate,
        batch_size,
        is_early_stopping=False,
        test_size=0.2,
        early_stopping_patience=10,
        early_stopping_min_delta=0.005,
        log_interval=1,
    ):
        torch.set_num_threads(1)
        if self.is_standarized:
            X, self.x_mean, self.x_std = self.standardize(X)
            y, self.y_mean, self.y_std = self.standardize(y)

        # Train-test split
        if is_early_stopping:
            early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                mode="min",
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size
            )

            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
        else:
            X_train, y_train = X, y

        # Convert data to PyTorch tensors
        train_data = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,  # weight_decay=1e-4
        )

        # Training loop
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            self.model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print(loss.item(), outputs, labels)

            train_losses.append(loss.item())

            # Early stopping
            if is_early_stopping:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_test)
                    val_loss = criterion(val_outputs, y_test)

                val_losses.append(val_loss.item())

                if early_stopping(val_loss.item()):
                    glogger.info(
                        "Early stopping! Converged at epoch {}".format(epoch + 1)
                    )
                    break

            # Logging the training progress
            if (epoch + 1) % log_interval == 0:
                s = "Epoch [{}/{}], Train Loss: {:.4f}".format(
                    epoch + 1, epochs, loss.item()
                )
                s = (
                    s + ", Val Loss: {:.4f}".format(val_loss.item())
                    if is_early_stopping
                    else s
                )
                glogger.debug(s)

        with torch.no_grad():
            self.model.eval()
            y_pred = self.model(torch.tensor(X, dtype=torch.float32)).numpy()
            self.mse = np.mean((y_pred - y) ** 2, axis=0)
            self.var = self.mse

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            if self.is_standarized:
                X = self.standardize(X, self.x_mean, self.x_std)
                y = self.model(torch.tensor(X, dtype=torch.float32)).numpy()
                y = self.destandardize(y, self.y_mean, self.y_std)
            else:
                y = self.model(torch.tensor(X, dtype=torch.float32)).numpy()
        return y

    def sample(self, X):
        self.model.eval()
        epsilons = np.random.multivariate_normal(
            mean=np.zeros(self.out_dim), cov=np.diag(self.mse), size=X.shape[0]
        )
        with torch.no_grad():
            if self.is_standarized:
                X = self.standardize(X, self.x_mean, self.x_std)
                y = self.model(torch.tensor(X, dtype=torch.float32)).numpy() + epsilons
                y = self.destandardize(y, self.y_mean, self.y_std)
            else:
                y = self.model(torch.tensor(X, dtype=torch.float32)).numpy() + epsilons
        return y


class LinearRegressor:
    def __init__(
        self,
        featurize_method=None,
        degree=1,
        interaction_only=False,
        is_standarized=False,
    ):
        super(LinearRegressor, self).__init__()
        self.featurize_method = featurize_method
        self.degree = degree
        self.interaction_only = interaction_only
        self.is_standarized = is_standarized

    @staticmethod
    def standardize(x, mean=None, std=None):
        if mean is None and std is None:
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            return (x - mean) / std, mean, std
        else:
            return (x - mean) / std

    @staticmethod
    def destandardize(x, mean, std):
        return x * std + mean

    def featurize(self, X):
        if self.featurize_method is None:
            return PolynomialFeatures(
                degree=1, include_bias=False, interaction_only=self.interaction_only
            ).fit_transform(X)
        elif self.featurize_method == "polynomial":
            return PolynomialFeatures(
                degree=self.degree,
                include_bias=False,
                interaction_only=self.interaction_only,
            ).fit_transform(X)

    def train(self, X, Y, **kwargs):
        if self.is_standarized:
            X, self.x_mean, self.x_std = self.standardize(X)
            Y, self.y_mean, self.y_std = self.standardize(Y)
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(self.featurize(X), Y)

        Y_pred = self.model.predict(self.featurize(X))
        self.mse = np.mean((Y_pred - Y) ** 2, axis=0)

    def predict(self, X):
        if self.is_standarized:
            X = self.standardize(X, self.x_mean, self.x_std)
            Y = self.model.predict(self.featurize(X))
            Y = self.destandardize(Y, self.y_mean, self.y_std)
        else:
            Y = self.model.predict(self.featurize(X))
        return Y

    def sample(self, X):
        epsilons = np.random.multivariate_normal(
            mean=np.zeros(self.model.coef_.shape[0]),
            cov=np.diag(self.mse),
            size=X.shape[0],
        )
        if self.is_standarized:
            X = self.standardize(X, self.x_mean, self.x_std)
            Y = self.model.predict(self.featurize(X)) + epsilons
            Y = self.destandardize(Y, self.y_mean, self.y_std)
        else:
            Y = self.model.predict(self.featurize(X)) + epsilons
        return Y


class NeuralNetRegressorBootstrapped(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, B, is_standarized=False):
        super(NeuralNetRegressorBootstrapped, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.is_standarized = is_standarized
        self.B = B
        self.models = nn.ModuleList(
            [NeuralNet(in_dim, out_dim, hidden_dims) for _ in range(B)]
        )

    @staticmethod
    def standardize(x, mean=None, std=None):
        if mean is None and std is None:
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            return (x - mean) / std, mean, std
        else:
            return (x - mean) / std

    @staticmethod
    def destandardize(x, mean, std):
        return x * std + mean

    def fit(
        self,
        X,
        y,
        epochs,
        learning_rate,
        batch_size,
        is_early_stopping=False,
        test_size=0.2,
        early_stopping_patience=10,
        early_stopping_min_delta=0.005,
        log_interval=1,
    ):
        torch.set_num_threads(1)
        if self.is_standarized:
            X, self.x_mean, self.x_std = self.standardize(X)
            y, self.y_mean, self.y_std = self.standardize(y)

        self.vars = []
        for b in range(self.B):
            model_b = self.models[b]

            # Bootstrap sampling
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_b = X[indices]
            y_b = y[indices]

            if is_early_stopping:
                early_stopping = EarlyStopping(
                    patience=early_stopping_patience,
                    min_delta=early_stopping_min_delta,
                    mode="min",
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X_b, y_b, test_size=test_size
                )

                X_test = torch.tensor(X_test, dtype=torch.float32)
                y_test = torch.tensor(y_test, dtype=torch.float32)
            else:
                X_train, y_train = X_b, y_b

            train_data = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            )
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(
                model_b.parameters(),
                lr=learning_rate,  # weight_decay=1e-4
            )
            for epoch in range(epochs):
                model_b.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model_b(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # Early stopping
                if is_early_stopping:
                    model_b.eval()
                    with torch.no_grad():
                        val_outputs = model_b(X_test)
                        val_loss = criterion(val_outputs, y_test)

                    if early_stopping(val_loss.item()):
                        glogger.debug(
                            "Early stopping! Converged at epoch {}".format(epoch + 1)
                        )
                        break

                # Logging the training progress
                if (epoch + 1) % log_interval == 0:
                    s = "Bootstrap Model {}/{} Epoch [{}/{}], Train Loss: {:.4f}".format(
                        b + 1, self.B, epoch + 1, epochs, loss.item()
                    )
                    s = (
                        s + ", Val Loss: {:.4f}".format(val_loss.item())
                        if is_early_stopping
                        else s
                    )
                    glogger.debug(s)

            with torch.no_grad():
                model_b.eval()
                y_pred = model_b(torch.tensor(X, dtype=torch.float32)).numpy()
                mse = np.mean((y_pred - y) ** 2, axis=0)
                self.vars.append(mse)
            self.models.append(model_b)

    def predict(self, X):
        y_sum = np.zeros((X.shape[0], self.out_dim))
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if self.is_standarized:
                    X = self.standardize(X, self.x_mean, self.x_std)
                    y = model(torch.tensor(X, dtype=torch.float32)).numpy()
                    y = self.destandardize(y, self.y_mean, self.y_std)
                else:
                    y = model(torch.tensor(X, dtype=torch.float32)).numpy()
            y_sum += y
        y = y_sum / self.B
        return y

    def sample(self, X):
        y_sum = np.zeros((X.shape[0], self.out_dim))
        for i, model in enumerate(self.models):
            model.eval()
            epsilons = np.random.multivariate_normal(
                mean=np.zeros(self.out_dim), cov=np.diag(self.vars[i]), size=X.shape[0]
            )
            with torch.no_grad():
                if self.is_standarized:
                    X = self.standardize(X, self.x_mean, self.x_std)
                    y = model(torch.tensor(X, dtype=torch.float32)).numpy() + epsilons
                    y = self.destandardize(y, self.y_mean, self.y_std)
                else:
                    y = model(torch.tensor(X, dtype=torch.float32)).numpy() + epsilons
            y_sum += y
        y = y_sum / self.B
        return y


class LinearRegressor:
    def __init__(
        self,
        featurize_method=None,
        degree=1,
        interaction_only=False,
        is_standarized=False,
    ):
        super(LinearRegressor, self).__init__()
        self.featurize_method = featurize_method
        self.degree = degree
        self.interaction_only = interaction_only
        self.is_standarized = is_standarized

    @staticmethod
    def standardize(x, mean=None, std=None):
        if mean is None and std is None:
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            return (x - mean) / std, mean, std
        else:
            return (x - mean) / std

    @staticmethod
    def destandardize(x, mean, std):
        return x * std + mean

    def featurize(self, X):
        if self.featurize_method is None:
            return PolynomialFeatures(
                degree=1, include_bias=False, interaction_only=self.interaction_only
            ).fit_transform(X)
        elif self.featurize_method == "polynomial":
            return PolynomialFeatures(
                degree=self.degree,
                include_bias=False,
                interaction_only=self.interaction_only,
            ).fit_transform(X)

    def train(self, X, Y, **kwargs):
        if self.is_standarized:
            X, self.x_mean, self.x_std = self.standardize(X)
            Y, self.y_mean, self.y_std = self.standardize(Y)
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(self.featurize(X), Y)

        Y_pred = self.model.predict(self.featurize(X))
        self.mse = np.mean((Y_pred - Y) ** 2, axis=0)

    def predict(self, X):
        if self.is_standarized:
            X = self.standardize(X, self.x_mean, self.x_std)
            Y = self.model.predict(self.featurize(X))
            Y = self.destandardize(Y, self.y_mean, self.y_std)
        else:
            Y = self.model.predict(self.featurize(X))
        return Y

    def sample(self, X):
        epsilons = np.random.multivariate_normal(
            mean=np.zeros(self.model.coef_.shape[0]),
            cov=np.diag(self.mse),
            size=X.shape[0],
        )
        if self.is_standarized:
            X = self.standardize(X, self.x_mean, self.x_std)
            Y = self.model.predict(self.featurize(X)) + epsilons
            Y = self.destandardize(Y, self.y_mean, self.y_std)
        else:
            Y = self.model.predict(self.featurize(X)) + epsilons
        return Y


class LinearRBFModel:
    def __init__(
        self,
        rbf_n_components,
        ridge_alpha,
        gamma,
        # seed,
    ):
        self.rbf_n_components = rbf_n_components
        self.ridge_alpha = ridge_alpha
        self.gamma = gamma
        self.model = self._build_model()

    def _build_model(self):
        scaler = StandardScaler()
        rbf_feature = RBFSampler(
            gamma="scale",
            n_components=self.rbf_n_components,
            # gamma=self.gamma,
            # random_state=seed,
        )
        return make_pipeline(scaler, rbf_feature, Ridge(alpha=self.ridge_alpha))
        # return make_pipeline(rbf_feature, Ridge(alpha=self.ridge_alpha))

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = root_mean_squared_error(y_test, y_pred)
        return mse
