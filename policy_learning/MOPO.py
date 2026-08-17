import os

import numpy as np
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from core.base_models import NeuralNet


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.position = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)

    def push(self, state, action, reward, next_state):
        """Push a single transition."""
        self._insert_batch(
            state[np.newaxis],
            action[np.newaxis],
            reward[np.newaxis],
            next_state[np.newaxis],
        )

    def push_batch(self, states, actions, rewards, next_states):
        """Push a batch of transitions."""
        self._insert_batch(states, actions, rewards, next_states)

    def _insert_batch(self, states, actions, rewards, next_states):
        batch_size = states.shape[0]
        for i in range(batch_size):
            idx = (self.position + i) % self.capacity
            self.states[idx] = states[i]
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.next_states[idx] = next_states[i]
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return dict(
            states=self.states[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_states=self.next_states[idxs],
        )

    def __len__(self):
        return self.size


class QLearning:
    def __init__(
        self, action_size, state_dim, hidden_dims=[32, 32], learning_rate=0.01
    ) -> None:
        self.action_size = action_size
        self.hidden_dims = hidden_dims
        self.policy_net = NeuralNet(
            in_dim=state_dim, out_dim=action_size, hidden_dims=hidden_dims
        )
        self.target_net = NeuralNet(
            in_dim=state_dim, out_dim=action_size, hidden_dims=hidden_dims
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )

    def train_one_step(
        self,
        states,
        actions,
        rewards,
        next_states,
        tau=0.005,  # Soft update parameter
        discount_factor=0.9,
    ):
        torch.set_num_threads(1)

        # convenience variables
        N, sdim = states.shape

        # reshape data
        next_states = next_states.reshape(-1, sdim)
        states = states.reshape(-1, sdim)
        actions = actions[:, :].flatten()
        rewards = rewards[:, :].flatten()

        # tensor conversion
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions.reshape(-1, 1), dtype=torch.int64)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)

        # one step training
        self.target_net.eval()
        tmp = self.target_net.forward(next_states_tensor).detach()
        Q_max = torch.max(tmp, dim=1)[0]
        Y_target = (rewards_tensor + discount_factor * Q_max).reshape(-1, 1)
        self.policy_net.train()
        Y_pred = self.policy_net.forward(states_tensor).gather(1, actions_tensor)
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(Y_pred, Y_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target network
        for target_param, local_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1 - tau) * target_param.data
            )
        return (
            loss.item(),
            Y_pred.detach().numpy().mean(),
        )

    def _predict_action(self, states):
        """Predict the action for given states using the policy network."""
        states = states.reshape(states.shape[0], -1)
        states = torch.tensor(states, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.policy_net.forward(states).detach().numpy()
        return np.argmax(q_values, axis=1)

    def act(self, states, epsilon=0, **kwargs):
        if epsilon > 0:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                return np.random.randint(0, self.action_size, size=(states.shape[0],))
            else:
                return self._predict_action(states)
        else:
            # Greedy action selection
            return self._predict_action(states)


class MOPO_SDP_QLearning:
    def __init__(
        self,
        sdp_preprocessor,
        rollout_length,
        batch_size,
        real_ratio,
        rollout_batch_size,
        num_iterations,
        hidden_dims,
    ):
        self.policy = QLearning(
            action_size=sdp_preprocessor.n_actions,
            state_dim=sdp_preprocessor.xdim * len(sdp_preprocessor.z_space),
            hidden_dims=hidden_dims,
            learning_rate=0.001,
        )
        self.sdp_preprocessor = sdp_preprocessor
        self.offline_buffer = ReplayBuffer(
            capacity=1000000,
            state_dim=sdp_preprocessor.xdim * len(sdp_preprocessor.z_space),
            action_dim=1,
        )
        self.model_buffer = ReplayBuffer(
            capacity=1000000,
            state_dim=sdp_preprocessor.xdim * len(sdp_preprocessor.z_space),
            action_dim=1,
        )
        self.rollout_length = rollout_length
        self.rollout_batch_size = rollout_batch_size
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.real_ratio = real_ratio

        # load dynamics model from SDP preprocessor

    def _sample_initial_states(self):
        states = self.offline_buffer.sample(self.rollout_batch_size)["states"]
        return states

    def _next_state_reward(self, states, actions, state_dim):
        assert self.sdp_preprocessor.is_normalized == False

        num_z = states.shape[-1] // state_dim

        K = len(self.sdp_preprocessor.model)

        # sample noise for each state and reward
        states_epsilons = np.zeros_like(states)
        rewards_epsilons = np.zeros((states.shape[0], 1))
        for k in range(K):
            # mse = self.sdp_preprocessor.model[k].mse
            mse = np.mean(self.sdp_preprocessor.model[k].vars, axis=0)
            epsilons = np.random.multivariate_normal(
                mean=np.zeros(state_dim + 1), cov=np.diag(mse), size=states.shape[0]
            )
            states_epsilons += np.repeat(epsilons[:, :-1], num_z, axis=1)
            rewards_epsilons += epsilons[:, -1:]
        states_epsilons /= K
        rewards_epsilons /= K

        next_states = np.zeros_like(states)
        rewards = np.zeros((states.shape[0], 1))
        for i in range(num_z):
            zi = i
            z = np.ones((states.shape[0], 1)) * zi
            states_z = states[:, i * state_dim : (i + 1) * state_dim]
            for k in range(K):
                if self.sdp_preprocessor.mode == "single":
                    at_encoded = self.sdp_preprocessor.encode_a(actions.flatten())
                    X = np.concatenate(
                        [
                            z,
                            at_encoded,
                            states_z,
                        ],
                        axis=1,
                    )
                    m = self.sdp_preprocessor.model[k].predict(X)
                elif self.sdp_preprocessor.mode == "sensitive":
                    at_encoded = self.sdp_preprocessor.encode_a(actions.flatten())
                    X = np.concatenate([at_encoded, states_z], axis=1)
                    m = self.sdp_preprocessor.model[k][tuple(z)].predict(X)

                next_states[:, i * state_dim : (i + 1) * state_dim] += m[:, :-1]
                rewards[:, 0] += (
                    m[:, -1] * self.sdp_preprocessor.marginal_dist_z[tuple([zi])]
                )

        next_states /= K
        rewards /= K

        next_states += states_epsilons
        rewards += rewards_epsilons

        return next_states, rewards

    def rollout_transitions(self):
        init_states = self._sample_initial_states()
        # roll out the dynamics model
        all_states = np.zeros(
            (init_states.shape[0], self.rollout_length, init_states.shape[1])
        )

        states = init_states.copy()
        for t in range(self.rollout_length):
            uat = np.random.uniform(0, 1, size=(states.shape[0]))
            actions = self.policy.act(
                states, uat=uat, epsilon=0.1
            )  # ! epsilon-greedy here for online learning
            next_states, rewards = self._next_state_reward(
                init_states, actions, self.sdp_preprocessor.xdim
            )
            self.model_buffer.push_batch(states, actions, rewards, next_states)
            states = next_states
            all_states[:, t, :] = states

    def update_offline_buffer(self, xs, zs, actions, rewards):
        # self.tmp_xs = xs
        T = xs.shape[1]
        for t in range(T):
            if t == 0:
                xt = xs[:, t, :]
                xtm1 = None
                atm1 = None
                z = zs
                rtm1 = None
                xt_tilde = self.sdp_preprocessor.preprocess(xt, xtm1, atm1, z, rtm1)
                prev_xt_tilde = xt_tilde
            else:
                xt = xs[:, t, :]
                xtm1 = xs[:, t - 1, :]
                atm1 = actions[:, t - 1]
                z = zs
                rtm1 = rewards[:, t - 1]
                xt_tilde, rtm1_tilde = self.sdp_preprocessor.preprocess(
                    xt, xtm1, atm1, z, rtm1
                )
                self.offline_buffer.push_batch(
                    prev_xt_tilde, atm1, rtm1_tilde, xt_tilde
                )
                prev_xt_tilde = xt_tilde
        return self.offline_buffer.size

    def train(self):
        real_sample_size = int(self.batch_size * self.real_ratio)
        model_sample_size = self.batch_size - real_sample_size
        for iteration in range(self.num_iterations):
            self.rollout_transitions()
            real_batch = self.offline_buffer.sample(real_sample_size)
            model_batch = self.model_buffer.sample(model_sample_size)

            states = np.concatenate(
                [real_batch["states"], model_batch["states"]], axis=0
            )
            actions = np.concatenate(
                [real_batch["actions"], model_batch["actions"]], axis=0
            )
            rewards = np.concatenate(
                [real_batch["rewards"], model_batch["rewards"]], axis=0
            )
            next_states = np.concatenate(
                [real_batch["next_states"], model_batch["next_states"]], axis=0
            )

            loss, predicted_value = self.policy.train_one_step(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                tau=0.005,  # Soft update parameter
                discount_factor=0.9,
            )

    def act(self, xt, z, xtm1, atm1, t, uat=None, is_return_prob=False, **kwargs):
        states = self.sdp_preprocessor.preprocess(
            xt=xt, xtm1=xtm1, z=z, atm1=atm1, rtm1=None, t=t
        )
        actions = self.policy.act(states, epsilon=0.0)
        if is_return_prob:
            return actions, None
        else:
            return actions


if __name__ == "__main__":
    # Example usage
    import sys, copy

    sys.path.append(
        "/home/jitwang/University of Michigan Dropbox/Jitao Wang/Fairness_RL/codes/Fairness_RL"
    )  # Adjust the path as necessary
    from core.preprocessing import SDP
    from simulation.code.envs import SimpleEnv, RandomPolicy, BehaviorPolicy

    N, T = 100, 20
    HIDDEN_DIMS = [32, 32]
    Z = np.random.binomial(1, 0.5, size=[N])
    env = SimpleEnv(env_name="nlm3", z_coef=1, seed=1)
    (
        zs,
        xs,
        actions,
        rewards,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = env.sample_trajectory(
        N, T, Z, seed=1, policy=BehaviorPolicy(10), include_counter=True
    )

    preprocessor = SDP(
        xs=copy.deepcopy(xs),
        zs=copy.deepcopy(zs),
        actions=copy.deepcopy(actions),
        rewards=copy.deepcopy(rewards),
        is_early_stopping=True,
        test_size=0.2,
        early_stopping_min_delta=0.001,
        early_stopping_patience=10,
        reg_model="nn",
        hidden_dims=HIDDEN_DIMS,
    )

    mopo = MOPO_SDP_QLearning(
        preprocessor,
        rollout_length=20,
        batch_size=32,
        real_ratio=0.5,
        rollout_batch_size=10,
        num_iterations=10000,
        hidden_dims=HIDDEN_DIMS,
    )
    mopo.update_offline_buffer(xs, zs, actions, rewards)
    mopo.train()
