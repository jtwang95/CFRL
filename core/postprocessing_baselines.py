import numpy as np
from .base_models import NeuralNet

import torch
import copy

from .utils import glogger


class OfflineFQI_NN:
    def __init__(self, action_size, hidden_dims, device) -> None:
        self.action_size = action_size
        self.hidden_dims = hidden_dims
        self.device = device

    def fit(
        self,
        states,
        actions,
        rewards,
        max_iter,
        inner_iter=500,
        learning_rate=0.1,
        discount_factor=0.9,
        early_stopping_fqi_loss=1e8,
    ):
        torch.set_num_threads(1)

        # convenience variables
        N, T = actions.shape
        sdim = states.shape[-1]

        #  # standarize states
        self.states_mean = np.mean(states, axis=(0, 1), keepdims=False)
        self.states_std = np.std(states, axis=(0, 1), keepdims=False)
        states = (states - self.states_mean) / (self.states_std + 1e-8)

        # reshape data
        states_tensor = torch.tensor(
            states[:, :-1, :].reshape(-1, states.shape[-1]), dtype=torch.float32
        ).to(self.device)
        next_states_tensor = torch.tensor(
            states[:, 1:, :].reshape(-1, states.shape[-1]), dtype=torch.float32
        ).to(self.device)
        actions_tensor = torch.tensor(
            actions.flatten().reshape(-1, 1), dtype=torch.int64
        ).to(self.device)
        rewards_tensor = torch.tensor(rewards.flatten(), dtype=torch.float32).to(
            self.device
        )
        # init model
        current_model = NeuralNet(
            in_dim=sdim, out_dim=self.action_size, hidden_dims=self.hidden_dims
        ).to(self.device)
        old_model = NeuralNet(
            in_dim=sdim, out_dim=self.action_size, hidden_dims=self.hidden_dims
        ).to(self.device)
        # optimizer = torch.optim.Adam(current_model.parameters(), lr=learning_rate)
        # training loop
        for i in range(max_iter):
            # generate target
            current_model = NeuralNet(
                in_dim=sdim, out_dim=self.action_size, hidden_dims=self.hidden_dims
            ).to(self.device)
            optimizer = torch.optim.Adam(current_model.parameters(), lr=learning_rate)
            with torch.no_grad():
                if i == 0:
                    Y = rewards_tensor.unsqueeze(1)
                else:
                    next_q_values = old_model.forward(next_states_tensor)
                    max_next_q_values, _ = torch.max(next_q_values, dim=1, keepdim=True)
                    Y = (
                        rewards_tensor.unsqueeze(1)
                        + discount_factor * max_next_q_values.detach()
                    )

            for _ in range(inner_iter):
                current_model.train()
                q_values_all_actions = current_model.forward(states_tensor)
                Y_pred = q_values_all_actions.gather(1, actions_tensor)
                optimizer.zero_grad()
                loss = torch.nn.MSELoss()(Y, Y_pred)
                loss.backward()
                optimizer.step()
            glogger.debug(
                "{}, fqi_nn mse:{}, mean_target:{}, lr:{}".format(
                    i, loss.item(), np.mean(Y.detach().cpu().numpy()), learning_rate
                )
            )
            # if loss.item() > 10:
            if loss.item() > early_stopping_fqi_loss:
                glogger.warning(
                    "FQI NN diverged, stop training, loss: {}, iter: {}".format(
                        loss.item(), i
                    )
                )
                break
            old_model = copy.deepcopy(current_model)

        self.model = copy.deepcopy(current_model)
        return loss.item()

    def act(self, states):
        self.model.eval()
        states = states.reshape(states.shape[0], -1)
        states = (states - self.states_mean) / (self.states_std + 1e-8)
        tmp = (
            self.model.forward(
                torch.tensor(states, dtype=torch.float32).to(self.device)
            )
            .detach()
            .cpu()
            .numpy()
        )

        return np.argmax(tmp, axis=1)

    def forward_unnormalized(self, states):
        self.model.eval()
        states = states.reshape(states.shape[0], -1)
        states = (states - self.states_mean) / (self.states_std + 1e-8)
        return (
            self.model.forward(
                torch.tensor(states, dtype=torch.float32).to(self.device)
            )
            .detach()
            .cpu()
            .numpy()
        )


class ECOCF:
    def __init__(self, state_size, action_size, hidden_dims, name, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.name = name

        self.agent = OfflineFQI_NN(
            action_size=action_size, hidden_dims=hidden_dims, device=device
        )

    def _train_Q(
        self,
        xs,
        z,
        actions,
        rewards,
        max_iter,
        inner_iter,
        learning_rate,
        discount_factor,
    ):
        N, T = xs.shape[:2]

        z_expanded = np.repeat(z[:, np.newaxis, :], T, axis=1)
        states_p = np.concatenate((xs, z_expanded), axis=-1)

        num_retries = 0
        while num_retries < 10:
            self.final_fqi_loss = self.agent.fit(
                states_p,
                actions,
                rewards,
                max_iter,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            if self.final_fqi_loss <= 2:
                break
            num_retries += 1

    def _train_postprocess(self, xs, zs):
        # learn marginal distribution of z
        self.marginal_dist_z = {}
        self.z_space = np.unique(zs, axis=0)
        for z in np.unique(zs, axis=0):
            z_idx = np.all(zs == z, axis=1)
            self.marginal_dist_z[tuple(z)] = sum(z_idx) / len(z_idx)

        # learn conditional expectation E(S_t | Z=z)
        self.model = {}  # {time: {z: mean_state}}
        N, T, xdim = xs.shape
        for t in range(T):
            self.model[t] = {}
            xs_t = xs[:, t, :]
            for z in np.unique(zs, axis=0):
                idx_z = np.all(zs == z, axis=1)
                self.model[t][tuple(z)] = np.mean(xs_t[idx_z, :], axis=0)

    def train(
        self,
        xs,
        z,
        actions,
        rewards,
        max_iter,
        inner_iter,
        learning_rate,
        discount_factor,
    ):
        self._train_Q(
            xs,
            z,
            actions,
            rewards,
            max_iter,
            inner_iter,
            learning_rate,
            discount_factor,
        )
        self._train_postprocess(xs, z)
        return self.final_fqi_loss

    def act(self, xt, z, xtm1, atm1, t, uat=None, is_return_prob=False, **kwargs):
        N = xt.shape[0]

        Q_values = np.zeros((N, self.action_size))
        for z1 in self.z_space:
            p_z1 = self.marginal_dist_z[tuple(z1)]
            q_values_inner = np.zeros((N, self.action_size))
            for z2 in self.z_space:
                p_z2 = self.marginal_dist_z[tuple(z2)]
                s_z1 = (
                    xt
                    - np.array([self.model[t][tuple(zi)] for zi in z])
                    + np.repeat(np.array(self.model[t][tuple(z1)]), N, axis=0).reshape(
                        -1, xt.shape[-1]
                    )
                )
                states = np.concatenate(
                    (s_z1, np.repeat(z2.reshape(1, -1), N, axis=0)), axis=-1
                )
                q_values_z2_sz1 = self.agent.forward_unnormalized(states)
                q_values_inner += p_z2 * q_values_z2_sz1
            Q_values += p_z1 * q_values_inner
        probs = np.zeros((N, self.action_size))
        actions = np.argmax(Q_values, axis=1)
        probs[np.arange(len(actions)), actions] = 1.0
        if is_return_prob:
            return probs
        else:
            return actions
