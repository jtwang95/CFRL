import numpy as np
import pandas as pd
import os
import torch
import warnings
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

warnings.simplefilter(action="ignore", category=FutureWarning)
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


from core.base_models import NeuralNetRegressorBootstrapped

from core.utils import *
from sklearn.preprocessing import OneHotEncoder

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "../data/")
MODEL_FOLDER = os.path.join(os.path.dirname(__file__), "../models/")
IS_EARLY_STOPPING = True


class PowerEDEnv:
    def __init__(
        self,
        zvars_cat,
        xvars=["weekly_pain_score", "weekly_pain_interference"],
        reward_multiplication_factor=[1.0, 1.0, 1.0],
        state_variance_factor=1.0,
        z_factor=0.0,
        seed=1,
    ) -> None:
        self.reward_multiplication_factor = reward_multiplication_factor
        self.state_variance_factor = state_variance_factor
        self.z_factor = z_factor
        self.zvars_cat = zvars_cat
        self.xvars = xvars
        self.seed = seed

        self.enc = self.make_encoder(self.zvars_cat, drop_first=True)

    @staticmethod
    def make_encoder(zvars_cat, drop_first=False):
        categories = []
        drop = "first" if drop_first else None
        zvar = zvars_cat
        if zvar == "sex":
            categories.append(np.array([1, 2]))
        elif zvar == "sex_b":
            categories.append(np.array([0, 1]))
        elif zvar == "ethnicity":
            categories.append(np.array([1, 2, 4]))
        elif zvar == "employed":
            categories.append(np.array([1, 2, 3, 4, 5, 6, 7]))
        elif zvar == "depression":
            categories.append(np.array(["none", "mild", "mod/severe"]))
        elif zvar == "race":
            categories.append(
                np.array(
                    [
                        "white",
                        "black",
                        "More than 1",
                        "Native American",
                        "other",
                        "asian",
                    ]
                )
            )
        elif zvar == "race_b":
            categories.append(np.array([0, 1]))
        elif zvar == "education":
            categories.append(
                np.array(["less than high school", "high school", "college"])
            )
        elif zvar == "education_b":
            categories.append(np.array([0, 1]))
        elif zvar == "age_b":
            categories.append(np.array([0, 1]))
        elif zvar == "depression_b":
            categories.append(np.array([0, 1]))
        elif zvar == "ethnicity_b":
            categories.append(np.array([0, 1]))
        else:
            raise ValueError("zvar not found, can not make encoder")
        return OneHotEncoder(categories=categories, drop=drop)

    def encode_z(self, z_cat):
        # return self.enc.fit_transform(z_cat).toarray()
        return np.array(z_cat).reshape(-1, 1)

    def encode_a(self, a):
        enc = OneHotEncoder(categories=[np.array([0, 1, 2])], drop=None)
        return enc.fit_transform(a.reshape(-1, 1)).toarray()

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

    @staticmethod
    def load_training_data(xvars, zvars_cat):
        impute_data = pd.read_csv(os.path.join(DATA_FOLDER, "impute_data.csv"))
        impute_data["rldecision"] = impute_data["rldecision"].replace(0, 1)
        impute_data["race_b"] = (impute_data["race"] == "white").astype(
            int
        )  # 0: NON-WHITE, 1: WHITE
        impute_data["education_b"] = (impute_data["education"] == "college").astype(
            int
        )  # 0: LESS THAN COLLEGE, 1: COLLEGE
        impute_data["sex_b"] = (impute_data["sex"] == 2).astype(
            int
        )  # 0: MALE, 1: FEMALE
        impute_data["age_b"] = (impute_data["age"] >= 40).astype(int)
        impute_data["depression_b"] = (
            impute_data["depression"] == "mod/severe"
        ).astype(int)  # 0: MILD, 1: MOD/SEVERE
        impute_data["ethnicity_b"] = (impute_data["ethnicity"] == 2).astype(
            int
        )  # 0: HISPANIC, 1: NOT HISPANIC

        N = len(np.unique(impute_data["pt_id"]))
        T = len(np.unique(impute_data["week"]))
        xt = np.zeros([N, T, len(xvars)])
        at = np.zeros([N, T], dtype=int)
        rt = np.zeros([N, T])
        impute_data = impute_data.sort_values(["pt_id", "week"], ignore_index=True)
        for week in range(1, T + 1):
            tmp_data = impute_data.loc[impute_data["week"] == week, :]
            xt[:, week - 1, :] = tmp_data.loc[:, xvars].to_numpy()
            at[:, week - 1] = tmp_data.loc[:, "rldecision"].to_numpy() - 1
            rt[:, week - 1] = tmp_data.loc[:, "reward"].to_numpy()

        z_cat = impute_data.loc[impute_data["week"] == 1, zvars_cat]
        z = np.array(z_cat).reshape(-1, 1)

        return xt, at, rt, z

    def fit(self, epochs, lr, hidden_dims, folder=None, early_stopping=True):
        self.trans_reward_model = NeuralNetRegressorBootstrapped(
            in_dim=6,
            out_dim=3,
            hidden_dims=hidden_dims,
            is_standarized=False,
            B=5,
        )
        model_name = os.path.join(
            MODEL_FOLDER,
            "trans_reward_model_{}_B{}_seed{}.pth".format(self.zvars_cat, 5, self.seed),
        )
        if os.path.exists(
            os.path.join(DATA_FOLDER, "impute_data.csv")
        ) and not os.path.exists(model_name):
            xt, at, rt, z = self.load_training_data(self.xvars, self.zvars_cat)
            N, T, _ = xt.shape

            # prepare input
            actions = at[:, 1:T].reshape(-1, 1)
            next_states = xt[:, 1:T, :].reshape(N * (T - 1), -1)
            states = xt[:, 0 : (T - 1), :].reshape(N * (T - 1), -1)
            sensitives = np.repeat(z[:, np.newaxis, :], repeats=T - 1, axis=1).reshape(
                N * (T - 1), -1
            )
            rewards = rt[:, 1:T].reshape(-1, 1)

            # encode action one hot
            actions = self.encode_a(actions)

            # fit marginal z model
            self.marginal_z_model = {
                0: np.mean(z.flatten() == 0),
                1: np.mean(z.flatten() == 1),
            }

            # fit initial state model
            self.initial_state_model = {
                0: np.mean(xt[z.flatten() == 0, 0, :], axis=0),
                1: np.mean(xt[z.flatten() == 1, 0, :], axis=0),
            }

            # fit the transition model
            z_encoded = self.encode_z(sensitives.flatten())
            X = np.concatenate([z_encoded, states, actions], axis=1)
            y = np.concatenate([next_states, rewards], axis=1)
            self.trans_reward_model.fit(
                X,
                y,
                epochs=epochs,
                batch_size=128,
                learning_rate=lr,
                is_early_stopping=early_stopping,
                test_size=0.2,
                early_stopping_patience=5,
                early_stopping_min_delta=0.00,
            )
            self.save_model(model_name)
            y_pred = self.trans_reward_model.predict(X)
            mse = np.mean((y - y_pred) ** 2, axis=0)
            self.next_state_var = mse[0 : xt.shape[-1]]
            self.reward_var = mse[xt.shape[-1] :]
        else:
            self.initial_state_model = {
                0: np.array([4.58714689, 2.24180791]),
                1: np.array([4.17546816, 2.02940075]),
            }
            self.next_state_var = np.array([26.17990925, 5.34577445])
            self.reward_var = np.array([32.64444906])
            self.load_model(model_name)

    def next_state_reward(self, xt, z_cat, at):

        next_states_mean, rewards_mean = self.next_state_reward_mean(xt, z_cat, at)

        errors_next_states = np.random.multivariate_normal(
            mean=np.zeros(xt.shape[-1]),
            cov=np.diag(self.next_state_var),
            size=xt.shape[0],
        )
        errors_rewards = np.random.normal(
            loc=0, scale=np.sqrt(self.reward_var), size=xt.shape[0]
        )

        next_states = next_states_mean + errors_next_states
        rewards = rewards_mean + errors_rewards

        return next_states, rewards

    def next_state_reward_mean(self, xt, z_cat, at):
        N = xt.shape[0]

        # input
        sensitives = self.encode_z(z_cat).reshape(N, -1)
        states = xt.reshape(N, -1)
        actions = self.encode_a(at)

        # next state & reward
        z_encoded = self.encode_z(sensitives.flatten())
        X = np.concatenate([z_encoded, states, actions], axis=1)
        out = self.trans_reward_model.predict(X)
        next_states_mean = out[:, 0 : xt.shape[-1]]
        rewards_mean = out[:, xt.shape[-1] :].flatten()
        idx = (sensitives == 1).flatten()
        if np.sum(idx) > 0:
            next_states_mean[idx] = (
                next_states_mean[idx] + self.z_factor * 0.1 * states[idx]
            )
            rewards_mean[idx] = (
                rewards_mean[idx] + self.z_factor * 0.1 * states[idx][:, 0]
            )

        return next_states_mean, rewards_mean

    def save_model(self, path):
        checkpoint = {
            "state_dict": self.trans_reward_model.state_dict(),
            "vars": getattr(self.trans_reward_model, "vars", []),
            "x_mean": getattr(self.trans_reward_model, "x_mean", None),
            "x_std": getattr(self.trans_reward_model, "x_std", None),
            "y_mean": getattr(self.trans_reward_model, "y_mean", None),
            "y_std": getattr(self.trans_reward_model, "y_std", None),
        }
        torch.save(checkpoint, path)

    def load_model(self, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )

        # Infer B from state_dict keys (e.g., models.9... -> B=10)
        model_indices = [
            int(k.split(".")[1])
            for k in state_dict.keys()
            if k.startswith("models.") and k.split(".")[1].isdigit()
        ]
        if model_indices:
            saved_B = max(model_indices) + 1
            if self.trans_reward_model.B != saved_B:
                self.trans_reward_model = NeuralNetRegressorBootstrapped(
                    in_dim=self.trans_reward_model.in_dim,
                    out_dim=self.trans_reward_model.out_dim,
                    hidden_dims=self.trans_reward_model.hidden_dims,
                    B=saved_B,
                    is_standarized=self.trans_reward_model.is_standarized,
                ).to(device)

        # Safely load state dict
        self.trans_reward_model.load_state_dict(state_dict)

        # Restore metadata
        if isinstance(checkpoint, dict):
            self.trans_reward_model.vars = checkpoint.get("vars", [])
            self.trans_reward_model.x_mean = checkpoint.get("x_mean", None)
            self.trans_reward_model.x_std = checkpoint.get("x_std", None)
            self.trans_reward_model.y_mean = checkpoint.get("y_mean", None)
            self.trans_reward_model.y_std = checkpoint.get("y_std", None)


# initial state model
def generate_intial_states(
    baseline_characteristics,
    zvars_cat,
    xvars=["weekly_pain_score", "weekly_pain_interference"],
    seed=1,
    folder=None,
    epochs=100,
    lr=0.01,
    z_factor=0.0,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if os.path.exists(os.path.join(DATA_FOLDER, "impute_data.csv")):
        impute_data = pd.read_csv(os.path.join(DATA_FOLDER, "impute_data.csv"))
        impute_data = impute_data.loc[impute_data["week"] == 1, :]
        impute_data["race_b"] = (impute_data["race"] == "white").astype(int)
        impute_data["sex_b"] = (impute_data["sex"] == 2).astype(int)
        impute_data["education_b"] = (impute_data["education"] == "college").astype(int)
        impute_data["ethnicity_b"] = (impute_data["ethnicity"] == 2).astype(int)
        # prepare data
        training_y = impute_data[xvars].to_numpy()
        training_x_cat = impute_data[zvars_cat]
        training_x = training_x_cat.to_numpy().reshape(-1, 1)
        model_mean = {}
        model_var = {}
        for z in [0, 1]:
            idx = (training_x == z).flatten()
            training_y_z = training_y[idx]
            model_mean[z] = np.mean(training_y_z, axis=0)
            model_var[z] = np.var(training_y_z, axis=0)
    else:
        if zvars_cat == "sex_b":
            model_mean = {
                0: np.array([4.56618357, 2.14613527]),
                1: np.array([4.3321256, 2.152657]),
            }
            model_var = {
                0: np.array([8.62051506, 0.54406719]),
                1: np.array([8.09533268, 0.57061291]),
            }
        elif zvars_cat == "ethnicity_b":
            model_mean = {
                0: np.array([4.73166667, 1.94222222]),
                1: np.array([4.35564972, 2.18578154]),
            }
            model_var = {
                0: np.array([11.8588213, 0.62393951]),
                1: np.array([7.65593167, 0.54263996]),
            }
        elif zvars_cat == "race_b":
            model_mean = {
                0: np.array([4.41239316, 2.25363248]),
                1: np.array([4.40878553, 2.0881137]),
            }
            model_var = {
                0: np.array([9.84780937, 0.52335931]),
                1: np.array([7.33613728, 0.57467784]),
            }
        elif zvars_cat == "education_b":
            model_mean = {
                0: np.array([4.58714689, 2.24180791]),
                1: np.array([4.17546816, 2.02940075]),
            }
            model_var = {
                0: np.array([7.99512905, 0.54957037]),
                1: np.array([8.56705424, 0.55223485]),
            }

    # Add z_factor to the mean of the second variable for z=1 to reflect the effect of z_factor on the outcome
    model_mean[1] = model_mean[1] + z_factor

    testing_x_cat = baseline_characteristics[zvars_cat]
    testing_x = testing_x_cat.to_numpy().reshape(-1, 1)

    testing_y = np.zeros([testing_x.shape[0], len(xvars)])
    for z in [0, 1]:
        idx = (testing_x == z).flatten()
        errors = np.random.multivariate_normal(
            mean=np.zeros(len(xvars)),
            cov=np.diag(model_var[z]),
            size=sum(idx),
        )
        testing_y[idx] = (
            np.repeat(model_mean[z][np.newaxis, :], np.sum(idx), axis=0) + errors
        )

    if len(xvars) >= 1:
        weekly_pain_score = testing_y[:, 0]
    if len(xvars) >= 2:
        weekly_pain_interference = testing_y[:, 1]

    # cliping
    if len(xvars) == 1:
        state0 = pd.DataFrame({"weekly_pain_score": weekly_pain_score})
    elif len(xvars) == 2:
        state0 = pd.DataFrame(
            {
                "weekly_pain_score": weekly_pain_score,
                "weekly_pain_interference": weekly_pain_interference,
            }
        )

    return state0, model_mean, model_var


# baseline characteristic model
def generate_baseline_characteristics(
    n,
    zvars_cat,
    seed=1,
    epochs=100,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_filename = os.path.join(
        MODEL_FOLDER,
        "powered_baseline_characteristic_model_seed{}.pkl".format(seed),
    )

    if os.path.exists(model_filename):
        synthesizer = CTGANSynthesizer.load(model_filename)
    else:
        metadata = SingleTableMetadata()
        impute_data = pd.read_csv(os.path.join(DATA_FOLDER, "impute_data.csv"))
        demo_data = impute_data.copy()
        demo_data = demo_data[
            [
                "num_substances",
                "baseline_pain",
                "age",
                "sex",
                "ethnicity",
                "employed",
                "depression",
                "race",
                "education",
                "comm_score",
            ]
        ]

        metadata.detect_from_dataframe(data=demo_data)
        metadata.update_column(column_name="sex", sdtype="categorical")
        metadata.update_column(column_name="ethnicity", sdtype="categorical")
        metadata.update_column(column_name="employed", sdtype="categorical")
        metadata.update_column(column_name="depression", sdtype="categorical")
        metadata.update_column(column_name="race", sdtype="categorical")
        metadata.update_column(column_name="education", sdtype="categorical")

        synthesizer = CTGANSynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            epochs=epochs,
        )
        synthesizer.fit(demo_data)
        synthesizer.save(model_filename)

    synthetic_data = synthesizer.sample(num_rows=n, output_file_path="disable")
    synthetic_data["id"] = range(n)
    #
    synthetic_data["race_b"] = (synthetic_data["race"] == "white").astype(int)
    synthetic_data["age_b"] = (synthetic_data["age"] >= 40).astype(int)
    synthetic_data["sex_b"] = (synthetic_data["sex"] == 1).astype(int)
    synthetic_data["education_b"] = (synthetic_data["education"] == "college").astype(
        int
    )
    synthetic_data["depression_b"] = (
        synthetic_data["depression"] == "mod/severe"
    ).astype(int)
    synthetic_data["ethnicity_b"] = (synthetic_data["ethnicity"] == 2).astype(int)
    return synthetic_data[["id"] + [zvars_cat]]


def generate_powered_trajectories(
    N,
    T,
    policy,
    zvars_cat,
    xvars=["weekly_pain_score", "weekly_pain_interference"],
    seed=1,
    folder=None,
    epochs=100,
    lr=0.01,
    state_variance_factor=1.0,
    z_factor=0.0,
):
    if os.path.exists(os.path.join(folder, "simulated_baselines_N{}.csv".format(N))):
        baseline_characteristic = pd.read_csv(
            os.path.join(folder, "simulated_baselines_N{}.csv".format(N))
        )
    else:
        baseline_characteristic = generate_baseline_characteristics(
            n=N,
            seed=seed,
            zvars_cat=zvars_cat,
            epochs=100,
        )
        baseline_characteristic.to_csv(
            os.path.join(folder, "simulated_baselines_N{}.csv".format(N))
        )
    if baseline_characteristic.shape[0] == N:
        pass
    else:
        baseline_characteristic = generate_baseline_characteristics(
            n=N,
            seed=seed,
            zvars_cat=zvars_cat,
            epochs=100,
        )
        baseline_characteristic.to_csv(
            os.path.join(folder, "simulated_baselines_N{}.csv".format(N))
        )

    states0, _, _ = generate_intial_states(
        baseline_characteristics=baseline_characteristic,
        seed=seed,
        zvars_cat=zvars_cat,
        xvars=xvars,
        epochs=epochs,
        lr=lr,
        z_factor=z_factor,
    )
    states0 = states0.to_numpy()

    z_cat = baseline_characteristic[zvars_cat]

    # init environment
    powered_env = PowerEDEnv(
        zvars_cat=zvars_cat,
        xvars=xvars,
        state_variance_factor=state_variance_factor,
        z_factor=z_factor,
        seed=seed,
    )

    powered_env.fit(epochs=epochs, lr=lr, folder=folder, hidden_dims=[64, 64])
    sensitives = powered_env.encode_z(z_cat)

    states = np.zeros([N, T + 1, len(powered_env.xvars)])
    rewards = np.zeros([N, T])
    actions = np.zeros([N, T], dtype=int)
    states[:, 0, :] = states0
    for week in range(1, T + 1):
        weeks = np.ones([N]) * week
        state = states[:, week - 1, :]
        action = policy.act(
            state=state,
            sensitive=sensitives,
            uat=np.random.uniform(size=[N]),
        )
        states[:, week, :], rewards[:, week - 1] = powered_env.next_state_reward(
            xt=states[:, week - 1, :],
            z_cat=z_cat,
            at=action,
        )
        actions[:, week - 1] = action

    df = pd.DataFrame()
    for week in range(1, T + 1):
        weeks = np.ones([N, 1]) * week
        tmp = pd.concat(
            [
                baseline_characteristic,
                pd.DataFrame(weeks, columns=["week"]),
                pd.DataFrame(states[:, week - 1, :], columns=powered_env.xvars),
                pd.DataFrame(actions[:, week - 1].reshape(-1, 1), columns=["action"]),
                pd.DataFrame(rewards[:, week - 1].reshape(-1, 1), columns=["reward"]),
            ],
            axis=1,
        )
        df = pd.concat([df, tmp], axis=0)

    return df, powered_env


class RandomAgent:
    def __init__(self) -> None:
        self.name = "random"

    def act(self, uat, is_return_prob=False, **kargs):
        probs = np.ones([uat.shape[0], 3]) * 1 / 3
        cdf = np.cumsum(probs, axis=1)

        actions = np.zeros([uat.shape[0]], dtype=int)
        visits = np.zeros_like(actions)
        if uat is not None:
            for i in range(cdf.shape[1]):
                masks = (uat.flatten() <= cdf[:, i]).astype(int) - visits
                actions += masks * i
                visits += masks
        else:
            uat = np.random.uniform(size=uat.shape[0])
            for i in range(cdf.shape[1]):
                masks = (uat.flatten() <= cdf[:, i]).astype(int) - visits
                actions += masks * i
                visits += masks

        if is_return_prob:
            return actions, probs
        else:
            return actions
