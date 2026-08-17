import numpy as np


# reward metric
def evaluate_reward_through_simulation(env, N, T, policy, seed, gamma=0.9):
    np.random.seed(seed)
    Z = np.random.binomial(1, p=0.5, size=[N, 1])
    _, _, actions, rewards = env.sample_trajectory(N, T, Z=Z, seed=seed, policy=policy)

    discounted_factor = np.repeat(
        np.array([[gamma**i for i in range(T)]]), repeats=N, axis=0
    )
    discounted_cumulative_reward = np.mean(discounted_factor * rewards) * T
    return discounted_cumulative_reward, np.mean(actions)


# fairness metric in the paper
def evaluate_fairness_through_simulation_given_observed_actions(
    env, N, T, policy, seed
):
    np.random.seed(seed)
    Z = np.random.binomial(n=1, p=0.5, size=[N])
    (_, _, actions_real, _, _, _, actions_counter, _) = (
        env.sample_trajectory_given_observed_actions(
            n=N,
            T=T,
            seed=seed,
            Z=Z,
            policy=policy,
            include_counter=True,
        )
    )

    cf_metric_t = np.mean(np.abs(actions_real - actions_counter), axis=0)
    cf_metric = np.mean(cf_metric_t)
    return cf_metric, cf_metric_t
