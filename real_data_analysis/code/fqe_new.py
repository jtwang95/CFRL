import numpy as np
import torch
import copy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


from core.base_models import NeuralNet
from core.utils import glogger


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


class FQE_NEW:
    def __init__(self, action_size, policy, hidden_dims=[32, 32], **kwargs):
        self.action_size = action_size
        self.policy = policy
        self.hidden_dims = hidden_dims

    def get_all_probs(self, xs, zs, actions):
        N, T, _ = xs.shape  # xs + 1
        if type(xs) is torch.Tensor:
            xs = xs.detach().numpy()
            zs = zs.detach().numpy()
            actions = actions.detach().numpy()
        else:
            xs = copy.deepcopy(xs)
            zs = copy.deepcopy(zs)
            actions = copy.deepcopy(actions)

        if type(self.policy).__name__ == "RandomAgent":
            probs = self.policy.act(
                uat=np.random.uniform(size=N * T), is_return_prob=True
            ).reshape(N, T, self.action_size)
        else:
            probs = np.zeros([N, T, self.action_size])
            for t in range(T):
                xt = xs[:, t]
                if t == 0:
                    probs[:, t] = self.policy.act(
                        xt=xt,
                        xtm1=None,
                        z=zs,
                        atm1=None,
                        rtm1=None,
                        t=t,
                        is_return_prob=True,
                    )
                else:
                    xtm1 = xs[:, t - 1]
                    atm1 = actions[:, t - 1]
                    rtm1 = None
                    probs[:, t] = self.policy.act(
                        xt=xt,
                        xtm1=xtm1,
                        z=zs,
                        atm1=atm1,
                        rtm1=rtm1,
                        t=t,
                        is_return_prob=True,
                    )
        return probs

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

    def fit(self, xs, zs, actions, rewards, inner_iter, max_iter, learning_rate):
        discount_factor = 0.9

        torch.set_num_threads(1)
        # convenience variables
        N, T = actions.shape
        sdim = xs.shape[-1] + 1

        # reshape data
        zs_expanded = torch.tensor(
            np.repeat(zs.reshape(-1, 1, 1), axis=1, repeats=T), dtype=torch.float32
        )
        next_states = torch.tensor(
            np.concatenate(
                [xs[:, 1:, :], zs_expanded.detach().numpy()], axis=2
            ).reshape(-1, sdim),
            dtype=torch.float32,
        ).reshape(-1, sdim)
        states = torch.tensor(
            np.concatenate(
                [xs[:, :-1, :], zs_expanded.detach().numpy()],
                axis=2,
            ),
            dtype=torch.float32,
        ).reshape(-1, sdim)
        actions_tensor = torch.tensor(actions.flatten(), dtype=torch.int64).reshape(
            -1, 1
        )
        rewards_tensor = torch.tensor(rewards.flatten(), dtype=torch.float32).reshape(
            -1, 1
        )

        # init model
        Q_k = NeuralNet(
            in_dim=sdim, out_dim=self.action_size, hidden_dims=self.hidden_dims
        )
        old_Q_k = copy.deepcopy(Q_k)

        # training loop
        for i in range(max_iter):
            with torch.no_grad():
                if i == 0:
                    Y = rewards_tensor
                else:
                    Q_val = old_Q_k(next_states)  # (n, A)
                    all_probs = self.get_all_probs(xs, zs, actions)  # shape (N, T, A)
                    selected_probs = torch.tensor(
                        all_probs[:, 1:].reshape(-1, self.action_size),
                        dtype=torch.float32,
                    )
                    Y = rewards_tensor + discount_factor * torch.mul(
                        Q_val, selected_probs
                    ).sum(1, keepdim=True)
            Q_k = NeuralNet(
                in_dim=sdim, out_dim=self.action_size, hidden_dims=self.hidden_dims
            )
            optimizer = torch.optim.Adam(Q_k.parameters(), lr=learning_rate)
            Q_k.train()
            for it in range(inner_iter):
                Y_pred = Q_k(states).gather(1, actions_tensor)
                loss = torch.nn.MSELoss()(Y_pred, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            glogger.debug(
                "{}, fqe_nn mse:{:.3f}, mean_target:{:.3f}".format(
                    i, loss.item(), np.mean(Y.detach().numpy())
                )
            )

            old_Q_k = copy.deepcopy(Q_k)

        self.model = Q_k
        self.final_loss = loss.item()

    def evaluate(self, xs, zs, actions):
        sdim = xs.shape[-1] + 1
        T = xs.shape[1]
        all_probs = self.get_all_probs(xs, zs, actions)
        xs_ = copy.deepcopy(xs)
        zs_ = copy.deepcopy(zs)

        states = np.concatenate(
            [xs_, np.repeat(zs_.reshape(-1, 1, 1), axis=1, repeats=T)], axis=2
        ).reshape(-1, sdim)

        self.model.eval()
        X = torch.tensor(states, dtype=torch.float32)
        Q_val = self.model.forward(X)
        value = torch.mul(
            Q_val,
            torch.tensor(all_probs.reshape(-1, self.action_size), dtype=torch.float32),
        ).sum(1, keepdim=True)
        value = value.detach().numpy().flatten()
        return value, self.final_loss
