import numpy as np
import pandas as pd


# reward metric
def evaluate_reward_through_simulation(env, N, T, policy, seed, gamma=0.9):
    # torch.random.manual_seed(seed)
    np.random.seed(seed)
    Z = np.random.binomial(1, p=0.5, size=[N, 1])
    _, _, _, rewards, _ = env.sample_trajectory(N, T, Z=Z, seed=seed, policy=policy)

    discounted_factor = np.repeat(
        np.array([[gamma**i for i in range(T)]]), repeats=N, axis=0
    )
    discounted_cumulative_reward = np.mean(discounted_factor * rewards) * T
    return discounted_cumulative_reward


# fairness metric
def evaluate_fairness_through_simulation(env, N, T, policy, seed, is_logit=False):
    np.random.seed(seed)
    # torch.random.manual_seed(seed)
    Z0 = np.random.binomial(n=1, p=0, size=[N])
    (
        _,
        _,
        actions_real,
        _,
        logits_real,
        _,
        _,
        actions_counter,
        _,
        logits_counter,
    ) = env.sample_trajectory(
        n=N,
        T=T,
        seed=seed,
        Z=Z0,
        policy=policy,
        include_counter=True,
    )
    if is_logit:
        cf_metric_t = np.mean(np.abs(logits_real - logits_counter), axis=0)
        cf_metric = np.mean(cf_metric_t)
    else:
        cf_metric_t = np.mean(np.abs(actions_real - actions_counter), axis=0)
        cf_metric = np.mean(cf_metric_t)
    return cf_metric, cf_metric_t
