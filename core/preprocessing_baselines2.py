import os

import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from sklearn.linear_model import LinearRegression
from .base_models import NeuralNetRegressor
from .utils import timer_func
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from .preprocessing import Preprocess


class SDP_avgstate(Preprocess):
    def __init__(
        self,
        xs,
        zs,
        actions,
        rewards,
        reg_model="lm",
        is_normalized=False,
        is_early_stopping=False,
        test_size=0.2,
        early_stopping_patience=5,
        early_stopping_min_delta=0.001,
        cross_folds=1,
        hidden_dims=[32, 32],
        mode="single",  # single, action, sensitive
    ) -> None:
        self.reg_model = reg_model
        self.is_normalized = is_normalized
        self.n_actions = len(np.unique(actions.flatten()))
        self.z_space = np.unique(zs, axis=0)
        self.xdim = xs.shape[-1]
        self.zdim = zs.shape[-1]

        # tunnable parameters
        self.is_early_stopping = is_early_stopping
        self.test_size = test_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.cross_folds = cross_folds
        self.mode = mode
        self.hidden_dims = hidden_dims

        # train
        _, _ = self.preprocess_training(xs, zs, actions, rewards)

    def encode_a(self, a):
        enc = OneHotEncoder(categories=[np.arange(self.n_actions)], drop=None)
        return enc.fit_transform(a.reshape(-1, 1)).toarray()

    def _learn_initial_model(self, xs, zs):
        # learn model at time 0
        model0 = {}
        for z in np.unique(zs, axis=0):
            idx_z = np.all(zs == z, axis=1)
            model0[tuple(z)] = np.mean(xs[idx_z, 0, :], axis=0)
        return model0

    def _learn_transition_models(self, xs, zs, actions, rewards):
        T = xs.shape[1]
        if self.is_normalized:
            states, self.states_mean, self.states_std = self.standardize(
                xs[:, : (T - 1), :].reshape(-1, self.xdim)
            )
            next_states, self.next_states_mean, self.next_states_std = self.standardize(
                np.concatenate(
                    [
                        xs[:, 1:T, :].reshape(-1, self.xdim),
                        rewards[:, :].reshape(-1, 1),
                    ],
                    axis=1,
                )
            )
        else:
            states = xs[:, : (T - 1), :].reshape(-1, self.xdim)
            next_states = np.concatenate(
                [xs[:, 1:T, :].reshape(-1, self.xdim), rewards[:, :].reshape(-1, 1)],
                axis=1,
            )

        if self.reg_model == "lm":
            return self._learn_linear_model(states, next_states, zs, actions, T)
        else:
            return self._learn_neural_model(states, next_states, zs, actions, T)

    def _learn_linear_model(self, states, next_states, zs, actions, T):
        X = np.concatenate(
            [
                np.repeat(zs.reshape(-1, 1, self.zdim), repeats=T - 1, axis=1).reshape(
                    -1, self.zdim
                ),
                actions[:, : (T - 1), np.newaxis].reshape(-1, 1),
                states,
            ],
            axis=1,
        )
        Y = next_states
        return LinearRegression().fit(X, Y)

    def _learn_neural_model(self, states, next_states, zs, actions, T):
        if self.mode == "single":
            return self._learn_single_neural_model(states, next_states, zs, actions, T)
        elif self.mode == "sensitive":
            return self._learn_sensitive_neural_model(
                states, next_states, zs, actions, T
            )

    def _learn_single_neural_model(self, states, next_states, zs, actions, T):
        actions_onehot = self.encode_a(actions[:, : (T - 1)].flatten())
        X = np.concatenate(
            [
                np.repeat(zs.reshape(-1, 1, self.zdim), repeats=T - 1, axis=1).reshape(
                    -1, self.zdim
                ),
                actions_onehot,
                states,
            ],
            axis=1,
        )
        Y = next_states
        model = NeuralNetRegressor(
            in_dim=X.shape[1],
            out_dim=Y.shape[1],
            hidden_dims=self.hidden_dims,
            is_standarized=False,
        )
        model.train(
            X,
            Y,
            epochs=1000,
            learning_rate=0.001,
            batch_size=256,
            is_early_stopping=self.is_early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            test_size=self.test_size,
        )
        return model

    def _learn_sensitive_neural_model(self, states, next_states, zs, actions, T):
        model = {}
        zs = np.repeat(zs[:, np.newaxis, :], axis=1, repeats=T - 1).reshape(
            -1, self.zdim
        )
        for z in np.unique(zs, axis=0):
            idx_z = np.all(zs == z, axis=1)
            states_z = states[idx_z].reshape(-1, self.xdim)
            next_states_z = next_states[idx_z].reshape(-1, next_states.shape[-1])
            actions_z = actions[:, : (T - 1)].reshape(-1, 1)[idx_z]
            actions_z_encoded = self.encode_a(actions_z.flatten())
            X = np.concatenate([actions_z_encoded, states_z], axis=1)
            Y = next_states_z

            model[tuple(z)] = NeuralNetRegressor(
                in_dim=X.shape[1],
                out_dim=Y.shape[1],
                hidden_dims=self.hidden_dims,
                is_standarized=False,
            )
            model[tuple(z)].train(
                X,
                Y,
                epochs=1000,
                learning_rate=0.001,
                batch_size=256,
                is_early_stopping=self.is_early_stopping,
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_min_delta=self.early_stopping_min_delta,
                test_size=self.test_size,
            )
        return model

    def learn_transition_models(self, xs, zs, actions, rewards):
        model0 = self._learn_initial_model(xs, zs)
        model = self._learn_transition_models(xs, zs, actions, rewards)
        return model0, model

    def learn_marginal_dist_z(self, zs):
        marginal_dist_z = {}
        self.z_space = np.unique(zs, axis=0)
        for z in np.unique(zs, axis=0):
            z_idx = np.all(zs == z, axis=1)
            marginal_dist_z[tuple(z)] = sum(z_idx) / len(z_idx)
        return marginal_dist_z

    def _estimate_cf_next_state_reward_mean_tg1(self, model, z, at, xt):
        if self.is_normalized:
            state = self.standardize(xt, self.states_mean, self.states_std)
        else:
            state = xt

        if self.mode == "single":
            at_encoded = self.encode_a(at.flatten())
            m = model.predict(
                np.concatenate(
                    [
                        z,
                        at_encoded,
                        state,
                    ],
                    axis=1,
                )
            )
        elif self.mode == "sensitive":
            m = np.zeros((xt.shape[0], self.xdim + 1))
            for z_ in self.z_space:
                idx_z = np.all(z == z_, axis=1)
                if sum(idx_z) == 0:
                    continue
                states_z = state[idx_z]
                actions_z = at[idx_z].reshape(-1, 1)
                actions_z_encoded = self.encode_a(actions_z.flatten())
                X = np.concatenate([actions_z_encoded, states_z], axis=1)
                m[idx_z] = model[tuple(z_)].predict(X)
        if self.is_normalized:
            m = self.destandardize(m, self.next_states_mean, self.next_states_std)
        return m

    def _process_initial_state(self, initial_model, xt, zs):
        N = xt.shape[0]
        epsilon_hat = xt - np.array([initial_model[tuple(z)] for z in zs])
        xt_c = {
            key: np.broadcast_to(initial_model[key], (N, self.xdim)) + epsilon_hat
            for key in self.marginal_dist_z
        }
        xt_tilde = np.mean(
            np.array([xt_c[key] for key in self.marginal_dist_z]), axis=0
        )
        return xt_tilde, xt_c

    def _process_subsequent_states(
        self, transition_model, xt, xtm1, zs, atm1, rtm1=None
    ):
        N = xt.shape[0]

        m = self._estimate_cf_next_state_reward_mean_tg1(
            transition_model, zs, atm1, xtm1
        )
        xt_mean = m[:, :-1]
        epsilon_hat = xt - xt_mean

        if rtm1 is not None:
            rtm1_mean = m[:, -1]
            epsilon_hat_reward = rtm1 - rtm1_mean

        xt_c = {}
        rtm1_c = {}
        for key, prob in self.marginal_dist_z.items():
            key_vec = np.repeat(np.array(key).reshape(1, -1), repeats=N, axis=0)
            m = self._estimate_cf_next_state_reward_mean_tg1(
                transition_model, key_vec, atm1, self.buffer[key]
            )
            xt_mean = m[:, :-1]
            xt_c[key] = xt_mean + epsilon_hat
            if rtm1 is not None:
                rtm1_mean = m[:, -1]
                rtm1_c[key] = rtm1_mean + epsilon_hat_reward

        xt_tilde = np.mean(
            np.array([xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]),
            axis=0,
        )
        if rtm1 is not None:
            rtm1_tilde = np.sum(
                [rtm1_c[key] * prob for key, prob in self.marginal_dist_z.items()],
                axis=0,
            )
        else:
            rtm1_tilde = None

        return xt_tilde, rtm1_tilde, xt_c

    def preprocess_training(self, xs, zs, actions, rewards):
        # some convenience variables
        N, T, xdim = xs.shape

        # learn marginal distribution of z
        self.marginal_dist_z = self.learn_marginal_dist_z(zs)

        # learn transition models, estimate residuals and estimate counterfactual outcome
        self.model0 = [None for _ in range(self.cross_folds)]
        self.model = [None for _ in range(self.cross_folds)]

        if self.cross_folds == 1:
            self.model0[0], self.model[0] = self.learn_transition_models(
                xs=xs, zs=zs, actions=actions, rewards=rewards
            )
            self.reset_buffer(n=N)
            xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
            rs_tilde = np.zeros([N, T - 1])
            for t in range(T):
                if t == 0:
                    xt_tilde, xt_c = self._process_initial_state(
                        self.model0[0], xs[:, 0, :], zs
                    )
                else:
                    xt_tilde, rtm1_tilde, xt_c = self._process_subsequent_states(
                        self.model[0],
                        xs[:, t, :],
                        xs[:, t - 1, :],
                        zs,
                        actions[:, t - 1],
                        rewards[:, t - 1],
                    )
                    rs_tilde[:, t - 1] = rtm1_tilde
                self.buffer = xt_c.copy()
                xs_tilde[:, t, :] = xt_tilde
        else:
            kf = KFold(n_splits=self.cross_folds)
            xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
            rs_tilde = np.zeros([N, T - 1])
            for i, (train_index, test_index) in enumerate(kf.split(xs)):
                xs_train, xs_test = xs[train_index], xs[test_index]
                zs_train, zs_test = zs[train_index], zs[test_index]
                actions_train, actions_test = actions[train_index], actions[test_index]
                rewards_train, rewards_test = rewards[train_index], rewards[test_index]
                self.model0[i], self.model[i] = self.learn_transition_models(
                    xs=xs_train,
                    zs=zs_train,
                    actions=actions_train,
                    rewards=rewards_train,
                )
                n = xs_test.shape[0]
                self.reset_buffer(n=n)
                for t in range(T):
                    if t == 0:
                        xt_tilde_test, xt_c_test = self._process_initial_state(
                            self.model0[i], xs_test[:, 0, :], zs_test
                        )
                    else:
                        xt_tilde_test, rtm1_tilde_test, xt_c_test = (
                            self._process_subsequent_states(
                                self.model[i],
                                xs_test[:, t, :],
                                xs_test[:, t - 1, :],
                                zs_test,
                                actions_test[:, t - 1],
                                rewards_test[:, t - 1],
                            )
                        )
                        rs_tilde[test_index, t - 1] = rtm1_tilde_test
                    xs_tilde[test_index, t, :] = xt_tilde_test
                    self.buffer = xt_c_test.copy()
        return xs_tilde, rs_tilde

    def preprocess(self, xt, xtm1, atm1, z, rtm1=None, **kwargs):
        # some convenience variables
        N, xdim = xt.shape
        cross_folds = len(self.model)

        # t = 0
        if xtm1 is None and atm1 is None:
            self.reset_buffer(n=N)

            if self.cross_folds == 1:
                xt_tilde, xt_c = self._process_initial_state(self.model0[0], xt, z)
                self.buffer = xt_c.copy()
            else:
                buffer_tmp = {key: np.zeros_like(xt) for key in self.marginal_dist_z}
                xt_tilde = np.zeros((N, self.xdim * len(self.marginal_dist_z.keys())))
                for k in range(cross_folds):
                    xt_tilde_k, xt_c_k = self._process_initial_state(
                        self.model0[k], xt, z
                    )
                    buffer_tmp = {
                        key: buffer_tmp[key] + xt_c_k[key] / cross_folds
                        for key in self.marginal_dist_z
                    }
                    xt_tilde += xt_tilde_k / cross_folds
                self.buffer = buffer_tmp.copy()
        else:
            if cross_folds == 1:
                xt_tilde, rtm1_tilde, xt_c = self._process_subsequent_states(
                    self.model[0], xt, xtm1, z, atm1, rtm1
                )
                self.buffer = xt_c.copy()
            else:
                buffer_tmp = {key: np.zeros_like(xt) for key in self.marginal_dist_z}
                xt_tilde = np.zeros((N, self.xdim * len(self.marginal_dist_z.keys())))
                if rtm1 is not None:
                    rtm1_tilde = np.zeros_like(rtm1)
                for k in range(cross_folds):
                    xt_tilde_k, rtm1_tilde_k, xt_c_k = self._process_subsequent_states(
                        self.model[k], xt, xtm1, z, atm1, rtm1
                    )
                    buffer_tmp = {
                        key: buffer_tmp[key] + xt_c_k[key] / cross_folds
                        for key in self.marginal_dist_z
                    }
                    xt_tilde += xt_tilde_k / cross_folds
                    if rtm1 is not None:
                        rtm1_tilde += rtm1_tilde_k / cross_folds
                self.buffer = buffer_tmp.copy()
        return (xt_tilde, rtm1_tilde) if rtm1 is not None else xt_tilde


class SDP_avgstate_Oracle(Preprocess):
    def __init__(self, env, xs, zs, actions, rewards) -> None:
        self.env = env
        self.xdim = 1
        self.zdim = 1
        self.preprocess_training(xs, zs, actions, rewards)

    @timer_func
    def learn_transition_models(self, **kwargs):
        # learn model at time 0
        model0 = {}
        zs = np.array([[0], [1]])
        for z in np.unique(zs, axis=0):
            model0[tuple(z)] = self.env.f_x0(z=z.reshape(1, 1), ux0=np.zeros((1, 1)))

        # learn model for time > 0
        class Model:
            def __init__(self, env) -> None:
                self.env = env

            def predict(self, xtm1, z, atm1):
                xtm1 = xtm1.reshape(-1, 1)
                z = z.reshape(-1, 1)
                atm1 = atm1.reshape(-1, 1)
                uxt = np.zeros_like(xtm1)
                urt = np.zeros_like(uxt)
                xt = self.env.f_xt(xtm1, z, atm1, uxt)
                rtm1 = self.env.f_rt(xtm1, z, atm1, urt)
                return np.concatenate([xt[:, np.newaxis], rtm1[:, np.newaxis]], axis=1)

        model = Model(self.env)

        return model0, model

    def learn_marginal_dist_z(self, zs):
        marginal_dist_z = {}
        self.z_space = np.unique(zs, axis=0)
        for z in np.unique(zs, axis=0):
            z_idx = np.all(zs == z, axis=1)
            marginal_dist_z[tuple(z)] = sum(z_idx) / len(z_idx)
        return marginal_dist_z

    def _estimate_cf_next_state_reward_mean_tg1(self, model, z, actions, states):
        m = model.predict(states, z, actions)
        return m

    def preprocess_training(self, xs, zs, actions, rewards):
        # some convenience variables
        N, T, xdim = xs.shape

        # learn marginal distribution of z
        self.marginal_dist_z = self.learn_marginal_dist_z(zs)

        # learn transition models, estimate residuals and estimate counterfactual outcome
        self.model0 = [None]
        self.model = [None]

        self.model0[0], self.model[0] = self.learn_transition_models()
        self.reset_buffer(n=N)
        xs_tilde = np.zeros([N, T, xdim])
        rs_tilde = np.zeros([N, T - 1])
        for t in range(T):
            if t == 0:
                epsilon_hat = xs[:, 0, :] - np.array(
                    [self.model0[0][tuple(z)] for z in zs]
                )
                xt_c = {}
                for key, prob in self.marginal_dist_z.items():
                    xt_c_mean = np.broadcast_to(self.model0[0][key], (N, xdim))
                    xt_c[key] = xt_c_mean + epsilon_hat
                # xt_tilde = np.hstack(
                #     [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
                # )
                xt_tilde = np.mean(
                    np.array(
                        [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
                    ),
                    axis=0,
                )
                self.buffer = xt_c.copy()
            else:
                m = self._estimate_cf_next_state_reward_mean_tg1(
                    self.model[0], zs, actions[:, t - 1], xs[:, t - 1, :]
                )
                xs_mean = m[:, :-1]
                r_mean = m[:, -1]
                epsilon_hat = xs[:, t, :] - xs_mean
                epsilon_hat_reward = rewards[:, t - 1] - r_mean
                xt_c, rt_c = {}, {}
                for key, prob in self.marginal_dist_z.items():
                    key_vec = np.repeat(np.array(key).reshape(1, -1), repeats=N, axis=0)
                    m = self._estimate_cf_next_state_reward_mean_tg1(
                        self.model[0], key_vec, actions[:, t - 1], self.buffer[key]
                    )
                    xs_mean = m[:, :-1]
                    r_mean = m[:, -1]
                    xt_c[key] = xs_mean + epsilon_hat
                    rt_c[key] = r_mean + epsilon_hat_reward
                # xt_tilde = np.hstack(
                #     [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
                # )
                xt_tilde = np.mean(
                    np.array(
                        [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
                    ),
                    axis=0,
                )
                rt_tilde = np.sum(
                    [rt_c[key] * prob for key, prob in self.marginal_dist_z.items()],
                    axis=0,
                )
                self.buffer = xt_c.copy()
                rs_tilde[:, t - 1] = rt_tilde
            xs_tilde[:, t, :] = xt_tilde
        return xs_tilde, rs_tilde

    def preprocess(self, xt, xtm1, atm1, z, rtm1=None, **kwargs):
        # some convenience variables
        N, xdim = xt.shape
        cross_folds = len(self.model)
        xt_tilde = None
        rtm1_tilde = None

        # t = 0
        if xtm1 is None and atm1 is None:
            self.reset_buffer(n=N)
            xt_mean = np.zeros_like(xt)

            # estimate epsilon_hat
            for k in range(cross_folds):
                model0 = self.model0[k]

                for z_ in np.unique(z, axis=0):
                    idx_z = np.all(z == z_, axis=1)
                    xt_mean[idx_z, :] += model0[tuple(z_)] / cross_folds
            epsilon_hat = xt - xt_mean

            # estimate counterfactual outcome
            xt_c = {}

            for key, prob in self.marginal_dist_z.items():
                xt_c_mean = np.zeros_like(xt)
                for k in range(cross_folds):
                    model0 = self.model0[k]
                    xt_c_mean += np.broadcast_to(model0[key], (N, xdim)) / cross_folds
                xt_c[key] = xt_c_mean + epsilon_hat
            # xt_tilde = np.hstack(
            #     [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
            # )
            xt_tilde = np.mean(
                np.array([xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]),
                axis=0,
            )
            self.buffer = xt_c.copy()

        else:
            atm1_new = atm1.reshape(-1, 1)
            xt_mean = np.zeros_like(xt)
            rtm1_mean = np.zeros((xt.shape[0]))

            # estimate epsilon_hat
            for k in range(cross_folds):
                model = self.model[k]
                for z_ in np.unique(z, axis=0):
                    idx_z = np.all(z == z_, axis=1)
                    z_ = np.repeat(z_.reshape(1, -1), repeats=sum(idx_z), axis=0)
                    m = self._estimate_cf_next_state_reward_mean_tg1(
                        model, z_, atm1_new[idx_z], xtm1[idx_z]
                    )
                    xt_mean[idx_z, :] += m[:, :-1] / cross_folds
                    rtm1_mean[idx_z] += m[:, -1] / cross_folds

            epsilon_hat = xt - xt_mean
            if rtm1 is not None:
                epsilon_hat_reward = rtm1 - rtm1_mean
            # predict counterfactual outcome
            xt_c = {}
            rtm1_c = {}

            for key, prob in self.marginal_dist_z.items():
                key_vec = np.repeat(np.array(key).reshape(1, -1), repeats=N, axis=0)
                xt_c_mean = np.zeros_like(xt)
                rtm1_c_mean = np.zeros((xt.shape[0]))
                for k in range(cross_folds):
                    model = self.model[k]
                    m = self._estimate_cf_next_state_reward_mean_tg1(
                        model, key_vec, atm1_new, self.buffer[key]
                    )
                    xt_c_mean += m[:, :-1] / cross_folds
                    rtm1_c_mean += m[:, -1] / cross_folds
                xt_c[key] = xt_c_mean + epsilon_hat
                if rtm1 is not None:
                    rtm1_c[key] = rtm1_c_mean + epsilon_hat_reward
            # xt_tilde = np.hstack(
            #     [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
            # )
            xt_tilde = np.mean(
                np.array([xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]),
                axis=0,
            )
            if rtm1 is not None:
                # print(self.marginal_dist_z.items())
                rtm1_tilde = np.sum(
                    [rtm1_c[key] * prob for key, prob in self.marginal_dist_z.items()],
                    axis=0,
                )
            self.buffer = xt_c.copy()
        if rtm1 is None:
            return xt_tilde
        else:
            return xt_tilde, rtm1_tilde


class FLAP(Preprocess):
    def __init__(
        self,
        xs,
        zs,
        actions,
        rewards,
    ) -> None:
        self.n_actions = len(np.unique(actions.flatten()))
        self.z_space = np.unique(zs, axis=0)
        self.xdim = xs.shape[-1]
        self.zdim = zs.shape[-1]

        # train
        _, _ = self.preprocess_training(xs, zs, actions, rewards)

    def learn_model(self, xs, zs):
        model = {}  # {time: {z: mean_state}}
        N, T, xdim = xs.shape
        for t in range(T):
            model[t] = {}
            xs_t = xs[:, t, :]
            for z in np.unique(zs, axis=0):
                idx_z = np.all(zs == z, axis=1)
                model[t][tuple(z)] = np.mean(xs_t[idx_z, :], axis=0)
        return model

    def learn_marginal_dist_z(self, zs):
        marginal_dist_z = {}
        self.z_space = np.unique(zs, axis=0)
        for z in np.unique(zs, axis=0):
            z_idx = np.all(zs == z, axis=1)
            marginal_dist_z[tuple(z)] = sum(z_idx) / len(z_idx)
        return marginal_dist_z

    def _process_state(self, xt, zs, t):
        N, xdim = xt.shape
        epsilon_hat = xt - np.array([self.model[t][tuple(z)] for z in zs])
        xt_c = {
            key: np.broadcast_to(self.model[t][key], (N, xdim)) + epsilon_hat
            for key in self.marginal_dist_z
        }

        xt_tilde = np.zeros((N, xdim))
        for key in self.marginal_dist_z:
            xt_tilde += xt_c[key] * self.marginal_dist_z[key]
        return xt_tilde

    def preprocess_training(self, xs, zs, actions, rewards):
        # some convenience variables
        N, T, xdim = xs.shape

        # learn marginal distribution of z
        self.marginal_dist_z = self.learn_marginal_dist_z(zs)

        # learn transition models, estimate residuals and estimate counterfactual outcome
        self.model = self.learn_model(xs, zs)  # {time: {z: mean_state}}

        xs_tilde = np.zeros([N, T, xdim])
        for t in range(T):
            xs_tilde[:, t, :] = self._process_state(xs[:, t, :], zs, t)

        rs_tilde = rewards
        return xs_tilde, rs_tilde

    def preprocess(self, z, xt, rtm1, t, **kwargs):
        N, xdim = xt.shape
        if t not in self.model:
            raise ValueError(f"Time step {t} not found in the model.")
        xt_tilde = self._process_state(xt, z, t)
        rtm1_tilde = rtm1

        return (xt_tilde, rtm1_tilde) if rtm1 is not None else xt_tilde


class SDP_nostate(Preprocess):
    def __init__(
        self,
        xs,
        zs,
        actions,
        rewards,
        reg_model="lm",
        is_action_onehot=False,
        is_normalized=False,
        is_early_stopping=False,
        test_size=0.2,
        early_stopping_patience=5,
        early_stopping_min_delta=0.001,
        cross_folds=1,
        hidden_dims=[32, 32],
        mode="single",  # single, action, sensitive
    ) -> None:
        self.reg_model = reg_model
        self.is_action_onehot = is_action_onehot
        self.is_normalized = is_normalized
        self.n_actions = len(np.unique(actions.flatten()))
        self.z_space = np.unique(zs, axis=0)
        self.xdim = xs.shape[-1]
        self.zdim = zs.shape[-1]

        # tunnable parameters
        self.is_early_stopping = is_early_stopping
        self.test_size = test_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.cross_folds = cross_folds
        self.hidden_dims = hidden_dims
        self.mode = mode

        # train
        _, _ = self.preprocess_training(xs, zs, actions, rewards)

    def _learn_initial_model(self, xs, zs):
        # learn model at time 0
        model0 = {}
        for z in np.unique(zs, axis=0):
            idx_z = np.all(zs == z, axis=1)
            model0[tuple(z)] = np.mean(xs[idx_z, 0, :], axis=0)
        return model0

    def _learn_transition_models(self, xs, zs, actions, rewards):
        T = xs.shape[1]
        if self.is_normalized:
            states, self.states_mean, self.states_std = self.standardize(
                xs[:, : (T - 1), :].reshape(-1, self.xdim)
            )
            next_states, self.next_states_mean, self.next_states_std = self.standardize(
                np.concatenate(
                    [
                        xs[:, 1:T, :].reshape(-1, self.xdim),
                        rewards[:, :].reshape(-1, 1),
                    ],
                    axis=1,
                )
            )
        else:
            states = xs[:, : (T - 1), :].reshape(-1, self.xdim)
            next_states = np.concatenate(
                [xs[:, 1:T, :].reshape(-1, self.xdim), rewards[:, :].reshape(-1, 1)],
                axis=1,
            )

        if self.reg_model == "lm":
            return self._learn_linear_model(states, next_states, zs, actions, T)
        else:
            return self._learn_neural_model(states, next_states, zs, actions, T)

    def _learn_linear_model(self, states, next_states, zs, actions, T):
        X = np.concatenate(
            [
                np.repeat(zs.reshape(-1, 1, self.zdim), repeats=T - 1, axis=1).reshape(
                    -1, self.zdim
                ),
                actions[:, : (T - 1), np.newaxis].reshape(-1, 1),
                states,
            ],
            axis=1,
        )
        Y = next_states
        return LinearRegression().fit(X, Y)

    def _learn_neural_model(self, states, next_states, zs, actions, T):
        if self.mode == "single":
            return self._learn_single_neural_model(states, next_states, zs, actions, T)
        elif self.mode == "sensitive":
            return self._learn_sensitive_neural_model(
                states, next_states, zs, actions, T
            )

    def _learn_single_neural_model(self, states, next_states, zs, actions, T):
        X = np.concatenate(
            [
                np.repeat(zs.reshape(-1, 1, self.zdim), repeats=T - 1, axis=1).reshape(
                    -1, self.zdim
                ),
                actions[:, : (T - 1), np.newaxis].reshape(-1, 1),
                states,
            ],
            axis=1,
        )
        Y = next_states
        model = NeuralNetRegressor(
            in_dim=X.shape[1], out_dim=Y.shape[1], hidden_dims=self.hidden_dims
        )
        model.train(
            X,
            Y,
            epochs=1000,
            learning_rate=0.001,
            batch_size=128,
            is_early_stopping=self.is_early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            test_size=self.test_size,
        )
        return model

    def _learn_sensitive_neural_model(self, states, next_states, zs, actions, T):
        model = {}
        zs = np.repeat(zs[:, np.newaxis, :], axis=1, repeats=T - 1).reshape(
            -1, self.zdim
        )
        for z in np.unique(zs, axis=0):
            idx_z = np.all(zs == z, axis=1)
            states_z = states[idx_z].reshape(-1, self.xdim)
            next_states_z = next_states[idx_z].reshape(-1, next_states.shape[-1])
            actions_z = actions[:, : (T - 1)].reshape(-1, 1)[idx_z]
            X = np.concatenate([actions_z, states_z], axis=1)
            Y = next_states_z

            model[tuple(z)] = NeuralNetRegressor(
                in_dim=X.shape[1], out_dim=Y.shape[1], hidden_dims=self.hidden_dims
            )
            model[tuple(z)].train(
                X,
                Y,
                epochs=1000,
                learning_rate=0.001,
                batch_size=128,
                is_early_stopping=self.is_early_stopping,
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_min_delta=self.early_stopping_min_delta,
                test_size=self.test_size,
            )
        return model

    def learn_transition_models(self, xs, zs, actions, rewards):
        model0 = self._learn_initial_model(xs, zs)
        model = self._learn_transition_models(xs, zs, actions, rewards)
        return model0, model

    def learn_marginal_dist_z(self, zs):
        marginal_dist_z = {}
        self.z_space = np.unique(zs, axis=0)
        for z in np.unique(zs, axis=0):
            z_idx = np.all(zs == z, axis=1)
            marginal_dist_z[tuple(z)] = sum(z_idx) / len(z_idx)
        return marginal_dist_z

    def _estimate_cf_next_state_reward_mean_tg1(self, model, z, at, xt):
        if self.is_normalized:
            state = self.standardize(xt, self.states_mean, self.states_std)
        else:
            state = xt
        if self.mode == "single":
            m = model.predict(
                np.concatenate(
                    [
                        z,
                        at.reshape(-1, 1),
                        state,
                    ],
                    axis=1,
                )
            )
        elif self.mode == "sensitive":
            m = np.zeros((xt.shape[0], self.xdim + 1))
            for z_ in self.z_space:
                idx_z = np.all(z == z_, axis=1)
                states_z = state[idx_z]
                actions_z = at[idx_z].reshape(-1, 1)
                X = np.concatenate([actions_z, states_z], axis=1)
                m[idx_z] = model[tuple(z_)].predict(X)
        if self.is_normalized:
            m = self.destandardize(m, self.next_states_mean, self.next_states_std)
        return m

    def _process_initial_state(self, initial_model, xt, zs):
        N = xt.shape[0]
        epsilon_hat = xt - np.array([initial_model[tuple(z)] for z in zs])
        xt_c = {
            key: np.broadcast_to(initial_model[key], (N, self.xdim)) + epsilon_hat
            for key in self.marginal_dist_z
        }
        xt_tilde = np.hstack([xt_c[key] for key in self.marginal_dist_z])
        return xt_tilde, xt_c

    def _process_subsequent_states(
        self, transition_model, xt, xtm1, zs, atm1, rtm1=None
    ):
        N = xt.shape[0]
        m = self._estimate_cf_next_state_reward_mean_tg1(
            transition_model, zs, atm1, xtm1
        )
        xt_mean = m[:, :-1]
        epsilon_hat = xt - xt_mean

        if rtm1 is not None:
            rtm1_mean = m[:, -1]
            epsilon_hat_reward = rtm1 - rtm1_mean

        xt_c = {}
        rtm1_c = {}
        for key, prob in self.marginal_dist_z.items():
            key_vec = np.repeat(np.array(key).reshape(1, -1), repeats=N, axis=0)
            m = self._estimate_cf_next_state_reward_mean_tg1(
                transition_model, key_vec, atm1, self.buffer[key]
            )
            xt_mean = m[:, :-1]
            xt_c[key] = xt_mean + epsilon_hat
            if rtm1 is not None:
                rtm1_mean = m[:, -1]
                rtm1_c[key] = rtm1_mean + epsilon_hat_reward

        xt_tilde = np.hstack(
            [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
        )
        if rtm1 is not None:
            rtm1_tilde = np.sum(
                [rtm1_c[key] * prob for key, prob in self.marginal_dist_z.items()],
                axis=0,
            )
        else:
            rtm1_tilde = None

        return xt_tilde, rtm1_tilde, xt_c

    def preprocess_training(self, xs, zs, actions, rewards):
        # some convenience variables
        N, T, xdim = xs.shape

        # learn marginal distribution of z
        self.marginal_dist_z = self.learn_marginal_dist_z(zs)

        # learn transition models, estimate residuals and estimate counterfactual outcome
        self.model0 = [None for _ in range(self.cross_folds)]
        self.model = [None for _ in range(self.cross_folds)]

        if self.cross_folds == 1:
            self.model0[0], self.model[0] = self.learn_transition_models(
                xs=xs, zs=zs, actions=actions, rewards=rewards
            )
            self.reset_buffer(n=N)
            xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
            rs_tilde = np.zeros([N, T - 1])
            for t in range(T):
                if t == 0:
                    xt_tilde, xt_c = self._process_initial_state(
                        self.model0[0], xs[:, 0, :], zs
                    )
                else:
                    xt_tilde, rtm1_tilde, xt_c = self._process_subsequent_states(
                        self.model[0],
                        xs[:, t, :],
                        xs[:, t - 1, :],
                        zs,
                        actions[:, t - 1],
                        rewards[:, t - 1],
                    )
                    rs_tilde[:, t - 1] = rtm1_tilde
                self.buffer = xt_c.copy()
                xs_tilde[:, t, :] = xt_tilde
        else:
            kf = KFold(n_splits=self.cross_folds)
            xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
            rs_tilde = np.zeros([N, T - 1])
            for i, (train_index, test_index) in enumerate(kf.split(xs)):
                xs_train, xs_test = xs[train_index], xs[test_index]
                zs_train, zs_test = zs[train_index], zs[test_index]
                actions_train, actions_test = actions[train_index], actions[test_index]
                rewards_train, rewards_test = rewards[train_index], rewards[test_index]
                self.model0[i], self.model[i] = self.learn_transition_models(
                    xs=xs_train,
                    zs=zs_train,
                    actions=actions_train,
                    rewards=rewards_train,
                )
                n = xs_test.shape[0]
                self.reset_buffer(n=n)
                for t in range(T):
                    if t == 0:
                        xt_tilde_test, xt_c_test = self._process_initial_state(
                            self.model0[i], xs_test[:, 0, :], zs_test
                        )
                    else:
                        xt_tilde_test, rtm1_tilde_test, xt_c_test = (
                            self._process_subsequent_states(
                                self.model[i],
                                xs_test[:, t, :],
                                xs_test[:, t - 1, :],
                                zs_test,
                                actions_test[:, t - 1],
                                rewards_test[:, t - 1],
                            )
                        )
                        rs_tilde[test_index, t - 1] = rtm1_tilde_test
                    # xs_tilde[test_index, t, :] = xt_tilde_test
                    self.buffer = xt_c_test.copy()
        xs_tilde = xs
        return xs_tilde, rs_tilde

    def preprocess(self, xt, xtm1, atm1, z, rtm1=None, **kwargs):
        # some convenience variables
        N, xdim = xt.shape
        cross_folds = len(self.model)

        # t = 0
        if xtm1 is None and atm1 is None:
            self.reset_buffer(n=N)

            if self.cross_folds == 1:
                xt_tilde, xt_c = self._process_initial_state(self.model0[0], xt, z)
                self.buffer = xt_c.copy()
            else:
                buffer_tmp = {key: np.zeros_like(xt) for key in self.marginal_dist_z}
                xt_tilde = np.zeros((N, self.xdim * len(self.marginal_dist_z.keys())))
                for k in range(cross_folds):
                    xt_tilde_k, xt_c_k = self._process_initial_state(
                        self.model0[k], xt, z
                    )
                    buffer_tmp = {
                        key: buffer_tmp[key] + xt_c_k[key] / cross_folds
                        for key in self.marginal_dist_z
                    }
                    xt_tilde += xt_tilde_k / cross_folds
                self.buffer = buffer_tmp.copy()
        else:
            if cross_folds == 1:
                xt_tilde, rtm1_tilde, xt_c = self._process_subsequent_states(
                    self.model[0], xt, xtm1, z, atm1, rtm1
                )
                self.buffer = xt_c.copy()
            else:
                buffer_tmp = {key: np.zeros_like(xt) for key in self.marginal_dist_z}
                xt_tilde = np.zeros((N, self.xdim * len(self.marginal_dist_z.keys())))
                if rtm1 is not None:
                    rtm1_tilde = np.zeros_like(rtm1)
                for k in range(cross_folds):
                    xt_tilde_k, rtm1_tilde_k, xt_c_k = self._process_subsequent_states(
                        self.model[k], xt, xtm1, z, atm1, rtm1
                    )
                    buffer_tmp = {
                        key: buffer_tmp[key] + xt_c_k[key] / cross_folds
                        for key in self.marginal_dist_z
                    }
                    xt_tilde += xt_tilde_k / cross_folds
                    if rtm1 is not None:
                        rtm1_tilde += rtm1_tilde_k / cross_folds
                self.buffer = buffer_tmp.copy()
        # xt_tilde = np.concatenate([xt, z.reshape(-1, 1)], axis=1)
        xt_tilde = xt
        return (xt_tilde, rtm1_tilde) if rtm1 is not None else xt_tilde


class SDP_noreward(Preprocess):
    def __init__(
        self,
        xs,
        zs,
        actions,
        rewards,
        reg_model="lm",
        is_action_onehot=False,
        is_normalized=False,
        is_early_stopping=False,
        test_size=0.2,
        early_stopping_patience=5,
        early_stopping_min_delta=0.001,
        cross_folds=1,
        hidden_dims=[32, 32],
        mode="single",  # single, action, sensitive
    ) -> None:
        self.reg_model = reg_model
        self.is_action_onehot = is_action_onehot
        self.is_normalized = is_normalized
        self.n_actions = len(np.unique(actions.flatten()))
        self.z_space = np.unique(zs, axis=0)
        self.xdim = xs.shape[-1]
        self.zdim = zs.shape[-1]

        # tunnable parameters
        self.is_early_stopping = is_early_stopping
        self.test_size = test_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.cross_folds = cross_folds
        self.hidden_dims = hidden_dims
        self.mode = mode

        # train
        _, _ = self.preprocess_training(xs, zs, actions, rewards)

    def _learn_initial_model(self, xs, zs):
        # learn model at time 0
        model0 = {}
        for z in np.unique(zs, axis=0):
            idx_z = np.all(zs == z, axis=1)
            model0[tuple(z)] = np.mean(xs[idx_z, 0, :], axis=0)
        return model0

    def _learn_transition_models(self, xs, zs, actions, rewards):
        T = xs.shape[1]
        if self.is_normalized:
            states, self.states_mean, self.states_std = self.standardize(
                xs[:, : (T - 1), :].reshape(-1, self.xdim)
            )
            next_states, self.next_states_mean, self.next_states_std = self.standardize(
                np.concatenate(
                    [
                        xs[:, 1:T, :].reshape(-1, self.xdim),
                        rewards[:, :].reshape(-1, 1),
                    ],
                    axis=1,
                )
            )
        else:
            states = xs[:, : (T - 1), :].reshape(-1, self.xdim)
            next_states = np.concatenate(
                [xs[:, 1:T, :].reshape(-1, self.xdim), rewards[:, :].reshape(-1, 1)],
                axis=1,
            )

        if self.reg_model == "lm":
            return self._learn_linear_model(states, next_states, zs, actions, T)
        else:
            return self._learn_neural_model(states, next_states, zs, actions, T)

    def _learn_linear_model(self, states, next_states, zs, actions, T):
        X = np.concatenate(
            [
                np.repeat(zs.reshape(-1, 1, self.zdim), repeats=T - 1, axis=1).reshape(
                    -1, self.zdim
                ),
                actions[:, : (T - 1), np.newaxis].reshape(-1, 1),
                states,
            ],
            axis=1,
        )
        Y = next_states
        return LinearRegression().fit(X, Y)

    def _learn_neural_model(self, states, next_states, zs, actions, T):
        if self.mode == "single":
            return self._learn_single_neural_model(states, next_states, zs, actions, T)
        elif self.mode == "sensitive":
            return self._learn_sensitive_neural_model(
                states, next_states, zs, actions, T
            )

    def _learn_single_neural_model(self, states, next_states, zs, actions, T):
        X = np.concatenate(
            [
                np.repeat(zs.reshape(-1, 1, self.zdim), repeats=T - 1, axis=1).reshape(
                    -1, self.zdim
                ),
                actions[:, : (T - 1), np.newaxis].reshape(-1, 1),
                states,
            ],
            axis=1,
        )
        Y = next_states
        model = NeuralNetRegressor(
            in_dim=X.shape[1], out_dim=Y.shape[1], hidden_dims=self.hidden_dims
        )
        model.train(
            X,
            Y,
            epochs=1000,
            learning_rate=0.001,
            batch_size=128,
            is_early_stopping=self.is_early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            test_size=self.test_size,
        )
        return model

    def _learn_sensitive_neural_model(self, states, next_states, zs, actions, T):
        model = {}
        zs = np.repeat(zs[:, np.newaxis, :], axis=1, repeats=T - 1).reshape(
            -1, self.zdim
        )
        for z in np.unique(zs, axis=0):
            idx_z = np.all(zs == z, axis=1)
            states_z = states[idx_z].reshape(-1, self.xdim)
            next_states_z = next_states[idx_z].reshape(-1, next_states.shape[-1])
            actions_z = actions[:, : (T - 1)].reshape(-1, 1)[idx_z]
            X = np.concatenate([actions_z, states_z], axis=1)
            Y = next_states_z

            model[tuple(z)] = NeuralNetRegressor(
                in_dim=X.shape[1], out_dim=Y.shape[1], hidden_dims=self.hidden_dims
            )
            model[tuple(z)].train(
                X,
                Y,
                epochs=1000,
                learning_rate=0.001,
                batch_size=128,
                is_early_stopping=self.is_early_stopping,
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_min_delta=self.early_stopping_min_delta,
                test_size=self.test_size,
            )
        return model

    def learn_transition_models(self, xs, zs, actions, rewards):
        model0 = self._learn_initial_model(xs, zs)
        model = self._learn_transition_models(xs, zs, actions, rewards)
        return model0, model

    def learn_marginal_dist_z(self, zs):
        marginal_dist_z = {}
        self.z_space = np.unique(zs, axis=0)
        for z in np.unique(zs, axis=0):
            z_idx = np.all(zs == z, axis=1)
            marginal_dist_z[tuple(z)] = sum(z_idx) / len(z_idx)
        return marginal_dist_z

    def _estimate_cf_next_state_reward_mean_tg1(self, model, z, at, xt):
        if self.is_normalized:
            state = self.standardize(xt, self.states_mean, self.states_std)
        else:
            state = xt
        if self.mode == "single":
            m = model.predict(
                np.concatenate(
                    [
                        z,
                        at.reshape(-1, 1),
                        state,
                    ],
                    axis=1,
                )
            )
        elif self.mode == "sensitive":
            m = np.zeros((xt.shape[0], self.xdim + 1))
            for z_ in self.z_space:
                idx_z = np.all(z == z_, axis=1)
                states_z = state[idx_z]
                actions_z = at[idx_z].reshape(-1, 1)
                X = np.concatenate([actions_z, states_z], axis=1)
                m[idx_z] = model[tuple(z_)].predict(X)
        if self.is_normalized:
            m = self.destandardize(m, self.next_states_mean, self.next_states_std)
        return m

    def _process_initial_state(self, initial_model, xt, zs):
        N = xt.shape[0]
        epsilon_hat = xt - np.array([initial_model[tuple(z)] for z in zs])
        xt_c = {
            key: np.broadcast_to(initial_model[key], (N, self.xdim)) + epsilon_hat
            for key in self.marginal_dist_z
        }
        xt_tilde = np.hstack([xt_c[key] for key in self.marginal_dist_z])
        return xt_tilde, xt_c

    def _process_subsequent_states(
        self, transition_model, xt, xtm1, zs, atm1, rtm1=None
    ):
        N = xt.shape[0]
        m = self._estimate_cf_next_state_reward_mean_tg1(
            transition_model, zs, atm1, xtm1
        )
        xt_mean = m[:, :-1]
        epsilon_hat = xt - xt_mean

        if rtm1 is not None:
            rtm1_mean = m[:, -1]
            epsilon_hat_reward = rtm1 - rtm1_mean

        xt_c = {}
        rtm1_c = {}
        for key, prob in self.marginal_dist_z.items():
            key_vec = np.repeat(np.array(key).reshape(1, -1), repeats=N, axis=0)
            m = self._estimate_cf_next_state_reward_mean_tg1(
                transition_model, key_vec, atm1, self.buffer[key]
            )
            xt_mean = m[:, :-1]
            xt_c[key] = xt_mean + epsilon_hat
            if rtm1 is not None:
                rtm1_mean = m[:, -1]
                rtm1_c[key] = rtm1_mean + epsilon_hat_reward

        xt_tilde = np.hstack(
            [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
        )
        if rtm1 is not None:
            rtm1_tilde = np.sum(
                [rtm1_c[key] * prob for key, prob in self.marginal_dist_z.items()],
                axis=0,
            )
        else:
            rtm1_tilde = None

        return xt_tilde, rtm1_tilde, xt_c

    def preprocess_training(self, xs, zs, actions, rewards):
        # some convenience variables
        N, T, xdim = xs.shape

        # learn marginal distribution of z
        self.marginal_dist_z = self.learn_marginal_dist_z(zs)

        # learn transition models, estimate residuals and estimate counterfactual outcome
        self.model0 = [None for _ in range(self.cross_folds)]
        self.model = [None for _ in range(self.cross_folds)]

        if self.cross_folds == 1:
            self.model0[0], self.model[0] = self.learn_transition_models(
                xs=xs, zs=zs, actions=actions, rewards=rewards
            )
            self.reset_buffer(n=N)
            xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
            rs_tilde = np.zeros([N, T - 1])
            for t in range(T):
                if t == 0:
                    xt_tilde, xt_c = self._process_initial_state(
                        self.model0[0], xs[:, 0, :], zs
                    )
                else:
                    xt_tilde, rtm1_tilde, xt_c = self._process_subsequent_states(
                        self.model[0],
                        xs[:, t, :],
                        xs[:, t - 1, :],
                        zs,
                        actions[:, t - 1],
                        rewards[:, t - 1],
                    )
                    rs_tilde[:, t - 1] = rtm1_tilde
                self.buffer = xt_c.copy()
                xs_tilde[:, t, :] = xt_tilde
        else:
            kf = KFold(n_splits=self.cross_folds)
            xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
            rs_tilde = np.zeros([N, T - 1])
            for i, (train_index, test_index) in enumerate(kf.split(xs)):
                xs_train, xs_test = xs[train_index], xs[test_index]
                zs_train, zs_test = zs[train_index], zs[test_index]
                actions_train, actions_test = actions[train_index], actions[test_index]
                rewards_train, rewards_test = rewards[train_index], rewards[test_index]
                self.model0[i], self.model[i] = self.learn_transition_models(
                    xs=xs_train,
                    zs=zs_train,
                    actions=actions_train,
                    rewards=rewards_train,
                )
                n = xs_test.shape[0]
                self.reset_buffer(n=n)
                for t in range(T):
                    if t == 0:
                        xt_tilde_test, xt_c_test = self._process_initial_state(
                            self.model0[i], xs_test[:, 0, :], zs_test
                        )
                    else:
                        xt_tilde_test, rtm1_tilde_test, xt_c_test = (
                            self._process_subsequent_states(
                                self.model[i],
                                xs_test[:, t, :],
                                xs_test[:, t - 1, :],
                                zs_test,
                                actions_test[:, t - 1],
                                rewards_test[:, t - 1],
                            )
                        )
                        rs_tilde[test_index, t - 1] = rtm1_tilde_test
                    xs_tilde[test_index, t, :] = xt_tilde_test
                    self.buffer = xt_c_test.copy()
        return xs_tilde, rs_tilde

    def preprocess(self, xt, xtm1, atm1, z, rtm1=None, **kwargs):
        # some convenience variables
        N, xdim = xt.shape
        cross_folds = len(self.model)

        # t = 0
        if xtm1 is None and atm1 is None:
            self.reset_buffer(n=N)

            if self.cross_folds == 1:
                xt_tilde, xt_c = self._process_initial_state(self.model0[0], xt, z)
                self.buffer = xt_c.copy()
            else:
                buffer_tmp = {key: np.zeros_like(xt) for key in self.marginal_dist_z}
                xt_tilde = np.zeros((N, self.xdim * len(self.marginal_dist_z.keys())))
                for k in range(cross_folds):
                    xt_tilde_k, xt_c_k = self._process_initial_state(
                        self.model0[k], xt, z
                    )
                    buffer_tmp = {
                        key: buffer_tmp[key] + xt_c_k[key] / cross_folds
                        for key in self.marginal_dist_z
                    }
                    xt_tilde += xt_tilde_k / cross_folds
                self.buffer = buffer_tmp.copy()
        else:
            if cross_folds == 1:
                xt_tilde, rtm1_tilde, xt_c = self._process_subsequent_states(
                    self.model[0], xt, xtm1, z, atm1, rtm1
                )
                self.buffer = xt_c.copy()
            else:
                buffer_tmp = {key: np.zeros_like(xt) for key in self.marginal_dist_z}
                xt_tilde = np.zeros((N, self.xdim * len(self.marginal_dist_z.keys())))
                if rtm1 is not None:
                    rtm1_tilde = np.zeros_like(rtm1)
                for k in range(cross_folds):
                    xt_tilde_k, rtm1_tilde_k, xt_c_k = self._process_subsequent_states(
                        self.model[k], xt, xtm1, z, atm1, rtm1
                    )
                    buffer_tmp = {
                        key: buffer_tmp[key] + xt_c_k[key] / cross_folds
                        for key in self.marginal_dist_z
                    }
                    xt_tilde += xt_tilde_k / cross_folds
                    if rtm1 is not None:
                        rtm1_tilde += rtm1_tilde_k / cross_folds
                self.buffer = buffer_tmp.copy()
            rtm1_tilde = rtm1
        return (xt_tilde, rtm1_tilde) if rtm1 is not None else xt_tilde


class SDP_noreward_Oracle(Preprocess):
    def __init__(self, env, xs, zs, actions, rewards) -> None:
        self.env = env
        self.xdim = 1
        self.zdim = 1
        self.preprocess_training(xs, zs, actions, rewards)

    def learn_transition_models(self, **kwargs):
        # learn model at time 0
        model0 = {}
        zs = np.array([[0], [1]])
        for z in np.unique(zs, axis=0):
            model0[tuple(z)] = self.env.f_x0(z=z.reshape(1, 1), ux0=np.zeros((1, 1)))

        # learn model for time > 0
        class Model:
            def __init__(self, env) -> None:
                self.env = env

            def predict(self, xtm1, z, atm1):
                xtm1 = xtm1.reshape(-1, 1)
                z = z.reshape(-1, 1)
                atm1 = atm1.reshape(-1, 1)
                uxt = np.zeros_like(xtm1)
                urt = np.zeros_like(uxt)
                xt = self.env.f_xt(xtm1, z, atm1, uxt)
                rtm1 = self.env.f_rt(xtm1, z, atm1, urt)
                return np.concatenate([xt[:, np.newaxis], rtm1[:, np.newaxis]], axis=1)

        model = Model(self.env)

        return model0, model

    def learn_marginal_dist_z(self, zs):
        marginal_dist_z = {}
        self.z_space = np.unique(zs, axis=0)
        for z in np.unique(zs, axis=0):
            z_idx = np.all(zs == z, axis=1)
            marginal_dist_z[tuple(z)] = sum(z_idx) / len(z_idx)
        return marginal_dist_z

    def _estimate_cf_next_state_reward_mean_tg1(self, model, z, actions, states):
        m = model.predict(states, z, actions)
        return m

    def preprocess_training(self, xs, zs, actions, rewards):
        # some convenience variables
        N, T, xdim = xs.shape

        # learn marginal distribution of z
        self.marginal_dist_z = self.learn_marginal_dist_z(zs)

        # learn transition models, estimate residuals and estimate counterfactual outcome
        self.model0 = [None]
        self.model = [None]

        self.model0[0], self.model[0] = self.learn_transition_models()
        self.reset_buffer(n=N)
        xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
        rs_tilde = np.zeros([N, T - 1])
        for t in range(T):
            if t == 0:
                epsilon_hat = xs[:, 0, :] - np.array(
                    [self.model0[0][tuple(z)] for z in zs]
                )
                xt_c = {}
                for key, prob in self.marginal_dist_z.items():
                    xt_c_mean = np.broadcast_to(self.model0[0][key], (N, xdim))
                    xt_c[key] = xt_c_mean + epsilon_hat
                xt_tilde = np.hstack(
                    [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
                )
                self.buffer = xt_c.copy()
            else:
                m = self._estimate_cf_next_state_reward_mean_tg1(
                    self.model[0], zs, actions[:, t - 1], xs[:, t - 1, :]
                )
                xs_mean = m[:, :-1]
                r_mean = m[:, -1]
                epsilon_hat = xs[:, t, :] - xs_mean
                epsilon_hat_reward = rewards[:, t - 1] - r_mean
                xt_c, rt_c = {}, {}
                for key, prob in self.marginal_dist_z.items():
                    key_vec = np.repeat(np.array(key).reshape(1, -1), repeats=N, axis=0)
                    m = self._estimate_cf_next_state_reward_mean_tg1(
                        self.model[0], key_vec, actions[:, t - 1], self.buffer[key]
                    )
                    xs_mean = m[:, :-1]
                    r_mean = m[:, -1]
                    xt_c[key] = xs_mean + epsilon_hat
                    rt_c[key] = r_mean + epsilon_hat_reward
                xt_tilde = np.hstack(
                    [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
                )
                rt_tilde = np.sum(
                    [rt_c[key] * prob for key, prob in self.marginal_dist_z.items()],
                    axis=0,
                )
                self.buffer = xt_c.copy()
                rs_tilde[:, t - 1] = rt_tilde
            xs_tilde[:, t, :] = xt_tilde
        return xs_tilde, rs_tilde

    def preprocess(self, xt, xtm1, atm1, z, rtm1=None, **kwargs):
        # some convenience variables
        N, xdim = xt.shape
        cross_folds = len(self.model)
        xt_tilde = None
        rtm1_tilde = None

        # t = 0
        if xtm1 is None and atm1 is None:
            self.reset_buffer(n=N)
            xt_mean = np.zeros_like(xt)

            # estimate epsilon_hat
            for k in range(cross_folds):
                model0 = self.model0[k]

                for z_ in np.unique(z, axis=0):
                    idx_z = np.all(z == z_, axis=1)
                    xt_mean[idx_z, :] += model0[tuple(z_)] / cross_folds
            epsilon_hat = xt - xt_mean

            # estimate counterfactual outcome
            xt_c = {}

            for key, prob in self.marginal_dist_z.items():
                xt_c_mean = np.zeros_like(xt)
                for k in range(cross_folds):
                    model0 = self.model0[k]
                    xt_c_mean += np.broadcast_to(model0[key], (N, xdim)) / cross_folds
                xt_c[key] = xt_c_mean + epsilon_hat
            xt_tilde = np.hstack(
                [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
            )
            self.buffer = xt_c.copy()

        else:
            atm1_new = atm1.reshape(-1, 1)
            xt_mean = np.zeros_like(xt)
            rtm1_mean = np.zeros((xt.shape[0]))

            # estimate epsilon_hat
            for k in range(cross_folds):
                model = self.model[k]
                for z_ in np.unique(z, axis=0):
                    idx_z = np.all(z == z_, axis=1)
                    z_ = np.repeat(z_.reshape(1, -1), repeats=sum(idx_z), axis=0)
                    m = self._estimate_cf_next_state_reward_mean_tg1(
                        model, z_, atm1_new[idx_z], xtm1[idx_z]
                    )
                    xt_mean[idx_z, :] += m[:, :-1] / cross_folds
                    rtm1_mean[idx_z] += m[:, -1] / cross_folds

            epsilon_hat = xt - xt_mean
            if rtm1 is not None:
                epsilon_hat_reward = rtm1 - rtm1_mean
            # predict counterfactual outcome
            xt_c = {}
            rtm1_c = {}

            for key, prob in self.marginal_dist_z.items():
                key_vec = np.repeat(np.array(key).reshape(1, -1), repeats=N, axis=0)
                xt_c_mean = np.zeros_like(xt)
                rtm1_c_mean = np.zeros((xt.shape[0]))
                for k in range(cross_folds):
                    model = self.model[k]
                    m = self._estimate_cf_next_state_reward_mean_tg1(
                        model, key_vec, atm1_new, self.buffer[key]
                    )
                    xt_c_mean += m[:, :-1] / cross_folds
                    rtm1_c_mean += m[:, -1] / cross_folds
                xt_c[key] = xt_c_mean + epsilon_hat
                if rtm1 is not None:
                    rtm1_c[key] = rtm1_c_mean + epsilon_hat_reward
            xt_tilde = np.hstack(
                [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
            )
            if rtm1 is not None:
                # print(self.marginal_dist_z.items())
                rtm1_tilde = np.sum(
                    [rtm1_c[key] * prob for key, prob in self.marginal_dist_z.items()],
                    axis=0,
                )
                rtm1_tilde = rtm1
            self.buffer = xt_c.copy()

        if rtm1 is None:
            return xt_tilde
        else:
            return xt_tilde, rtm1_tilde


class SDP_nostate_Oracle(Preprocess):
    def __init__(self, env, xs, zs, actions, rewards) -> None:
        self.env = env
        self.xdim = 1
        self.zdim = 1
        self.preprocess_training(xs, zs, actions, rewards)

    def learn_transition_models(self, **kwargs):
        # learn model at time 0
        model0 = {}
        zs = np.array([[0], [1]])
        for z in np.unique(zs, axis=0):
            model0[tuple(z)] = self.env.f_x0(z=z.reshape(1, 1), ux0=np.zeros((1, 1)))

        # learn model for time > 0
        class Model:
            def __init__(self, env) -> None:
                self.env = env

            def predict(self, xtm1, z, atm1):
                xtm1 = xtm1.reshape(-1, 1)
                z = z.reshape(-1, 1)
                atm1 = atm1.reshape(-1, 1)
                uxt = np.zeros_like(xtm1)
                urt = np.zeros_like(uxt)
                xt = self.env.f_xt(xtm1, z, atm1, uxt)
                rtm1 = self.env.f_rt(xtm1, z, atm1, urt)
                return np.concatenate([xt[:, np.newaxis], rtm1[:, np.newaxis]], axis=1)

        model = Model(self.env)

        return model0, model

    def learn_marginal_dist_z(self, zs):
        marginal_dist_z = {}
        self.z_space = np.unique(zs, axis=0)
        for z in np.unique(zs, axis=0):
            z_idx = np.all(zs == z, axis=1)
            marginal_dist_z[tuple(z)] = sum(z_idx) / len(z_idx)
        return marginal_dist_z

    def _estimate_cf_next_state_reward_mean_tg1(self, model, z, actions, states):
        m = model.predict(states, z, actions)
        return m

    def preprocess_training(self, xs, zs, actions, rewards):
        # some convenience variables
        N, T, xdim = xs.shape

        # learn marginal distribution of z
        self.marginal_dist_z = self.learn_marginal_dist_z(zs)

        # learn transition models, estimate residuals and estimate counterfactual outcome
        self.model0 = [None]
        self.model = [None]

        self.model0[0], self.model[0] = self.learn_transition_models()
        self.reset_buffer(n=N)
        xs_tilde = np.zeros([N, T, xdim])
        rs_tilde = np.zeros([N, T - 1])
        for t in range(T):
            if t == 0:
                epsilon_hat = xs[:, 0, :] - np.array(
                    [self.model0[0][tuple(z)] for z in zs]
                )
                xt_c = {}
                for key, prob in self.marginal_dist_z.items():
                    xt_c_mean = np.broadcast_to(self.model0[0][key], (N, xdim))
                    xt_c[key] = xt_c_mean + epsilon_hat
                xt_tilde = np.hstack(
                    [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
                )
                self.buffer = xt_c.copy()
            else:
                m = self._estimate_cf_next_state_reward_mean_tg1(
                    self.model[0], zs, actions[:, t - 1], xs[:, t - 1, :]
                )
                xs_mean = m[:, :-1]
                r_mean = m[:, -1]
                epsilon_hat = xs[:, t, :] - xs_mean
                epsilon_hat_reward = rewards[:, t - 1] - r_mean
                xt_c, rt_c = {}, {}
                for key, prob in self.marginal_dist_z.items():
                    key_vec = np.repeat(np.array(key).reshape(1, -1), repeats=N, axis=0)
                    m = self._estimate_cf_next_state_reward_mean_tg1(
                        self.model[0], key_vec, actions[:, t - 1], self.buffer[key]
                    )
                    xs_mean = m[:, :-1]
                    r_mean = m[:, -1]
                    xt_c[key] = xs_mean + epsilon_hat
                    rt_c[key] = r_mean + epsilon_hat_reward
                # xt_tilde = np.hstack(
                #     [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
                # )
                rt_tilde = np.sum(
                    [rt_c[key] * prob for key, prob in self.marginal_dist_z.items()],
                    axis=0,
                )
                self.buffer = xt_c.copy()
                rs_tilde[:, t - 1] = rt_tilde
            xs_tilde[:, t, :] = xs[:, t, :]
        return xs_tilde, rs_tilde

    def preprocess(self, xt, xtm1, atm1, z, rtm1=None, **kwargs):
        # some convenience variables
        N, xdim = xt.shape
        cross_folds = len(self.model)
        xt_tilde = None
        rtm1_tilde = None

        # t = 0
        if xtm1 is None and atm1 is None:
            self.reset_buffer(n=N)
            xt_mean = np.zeros_like(xt)

            # estimate epsilon_hat
            for k in range(cross_folds):
                model0 = self.model0[k]

                for z_ in np.unique(z, axis=0):
                    idx_z = np.all(z == z_, axis=1)
                    xt_mean[idx_z, :] += model0[tuple(z_)] / cross_folds
            epsilon_hat = xt - xt_mean

            # estimate counterfactual outcome
            xt_c = {}

            for key, prob in self.marginal_dist_z.items():
                xt_c_mean = np.zeros_like(xt)
                for k in range(cross_folds):
                    model0 = self.model0[k]
                    xt_c_mean += np.broadcast_to(model0[key], (N, xdim)) / cross_folds
                xt_c[key] = xt_c_mean + epsilon_hat
            xt_tilde = np.hstack(
                [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
            )
            self.buffer = xt_c.copy()

        else:
            atm1_new = atm1.reshape(-1, 1)
            xt_mean = np.zeros_like(xt)
            rtm1_mean = np.zeros((xt.shape[0]))

            # estimate epsilon_hat
            for k in range(cross_folds):
                model = self.model[k]
                for z_ in np.unique(z, axis=0):
                    idx_z = np.all(z == z_, axis=1)
                    z_ = np.repeat(z_.reshape(1, -1), repeats=sum(idx_z), axis=0)
                    m = self._estimate_cf_next_state_reward_mean_tg1(
                        model, z_, atm1_new[idx_z], xtm1[idx_z]
                    )
                    xt_mean[idx_z, :] += m[:, :-1] / cross_folds
                    rtm1_mean[idx_z] += m[:, -1] / cross_folds

            epsilon_hat = xt - xt_mean
            if rtm1 is not None:
                epsilon_hat_reward = rtm1 - rtm1_mean
            # predict counterfactual outcome
            xt_c = {}
            rtm1_c = {}

            for key, prob in self.marginal_dist_z.items():
                key_vec = np.repeat(np.array(key).reshape(1, -1), repeats=N, axis=0)
                xt_c_mean = np.zeros_like(xt)
                rtm1_c_mean = np.zeros((xt.shape[0]))
                for k in range(cross_folds):
                    model = self.model[k]
                    m = self._estimate_cf_next_state_reward_mean_tg1(
                        model, key_vec, atm1_new, self.buffer[key]
                    )
                    xt_c_mean += m[:, :-1] / cross_folds
                    rtm1_c_mean += m[:, -1] / cross_folds
                xt_c[key] = xt_c_mean + epsilon_hat
                if rtm1 is not None:
                    rtm1_c[key] = rtm1_c_mean + epsilon_hat_reward
            # xt_tilde = np.hstack(
            #     [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
            # )
            if rtm1 is not None:
                rtm1_tilde = np.sum(
                    [rtm1_c[key] * prob for key, prob in self.marginal_dist_z.items()],
                    axis=0,
                )
            self.buffer = xt_c.copy()
        # xt_tilde = np.concatenate([xt, z.reshape(-1, 1)], axis=1)
        xt_tilde = xt

        if rtm1 is None:
            return xt_tilde
        else:
            return xt_tilde, rtm1_tilde
