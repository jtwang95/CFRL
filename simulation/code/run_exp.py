import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from core.preprocessing_baselines import *
from evaluation_metrics import (
    evaluate_reward_through_simulation,
    evaluate_fairness_through_simulation_given_observed_actions,
)
from envs import SimpleEnv, BehaviorPolicy, RandomPolicy
from core.preprocessing import SDP, SDPOracle

from policy_learning.FQI import OfflineFQIWrapped
from policy_learning.MOPO import MOPO_SDP_QLearning
from core.utils import glogger
import numpy as np
import pandas as pd
import copy
import torch
import multiprocessing as mp
from tqdm import tqdm

from core.preprocessing_baselines2 import (
    FLAP,
    SDP_avgstate,
    SDP_avgstate_Oracle,
    SDP_noreward,
    SDP_noreward_Oracle,
    SDP_nostate,
    SDP_nostate_Oracle,
)
from core.postprocessing_baselines import ECOCF


glogger.setLevel("INFO")


def run_exp1_one(env_name, methods, method_policy, N, T, z_coef, seed):
    torch.set_num_threads(1)
    T_eval = 10
    N_eval = 10000
    eval_seed = seed * 10
    HIDDEN_DIMS = [32, 32]

    env = SimpleEnv(env_name=env_name, z_coef=z_coef, seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    Z = np.random.binomial(1, 0.5, size=[N])
    working_policy = BehaviorPolicy(seed=seed)

    if method_policy == "FQI_NN":
        learning_algorithm = OfflineFQIWrapped
        max_iters = 100
        inner_iter = 500
        learning_rate = 0.01
        discount_factor = 0.9
    else:
        raise ValueError("Method policy not found")
    (
        zs,
        xs,
        actions,
        rewards,
        _,
        _,
        _,
        _,
    ) = env.sample_trajectory(
        N, T, Z, seed=seed, policy=working_policy, include_counter=True
    )

    policies = []
    for method in methods:
        if method == "random":
            policies.append(RandomPolicy())
        elif method == "behavior":
            policies.append(BehaviorPolicy(seed=seed))
        elif method == "full":
            preprocessor = ConcatenateProprocess()
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=2,
                action_size=2,
                name="Full",
                hidden_dims=HIDDEN_DIMS,
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            policies.append(agent)
        elif method == "unaware":
            preprocessor = UnawarenessPreprocess()
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=1,
                action_size=2,
                name="Unaware",
                hidden_dims=HIDDEN_DIMS,
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            policies.append(agent)
        elif method == "ours":
            preprocessor = SDP(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                is_early_stopping=False,
                test_size=0.2,
                early_stopping_min_delta=0.001,
                early_stopping_patience=10,
                reg_model="nn",
                is_normalized=False,
                hidden_dims=HIDDEN_DIMS,
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=2,
                action_size=2,
                name="ours",
                hidden_dims=HIDDEN_DIMS,
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            policies.append(agent)
        elif method == "ours_cf10":
            preprocessor = SDP(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                reg_model="nn",
                is_normalized=False,
                cross_folds=10,
                hidden_dims=HIDDEN_DIMS,
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=2,
                action_size=2,
                name="ours_cf10",
                hidden_dims=HIDDEN_DIMS,
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            policies.append(agent)
        elif method == "oracle":
            preprocessor = SDPOracle(
                env=env,
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=2,
                action_size=2,
                name="oracle",
                hidden_dims=HIDDEN_DIMS,
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            policies.append(agent)

        elif method == "flap":
            preprocessor = FLAP(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=1,
                action_size=2,
                name="flap",
                hidden_dims=HIDDEN_DIMS,
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            policies.append(agent)
        elif method == "ecocf":
            agent = ECOCF(
                state_size=2,
                action_size=2,
                name="ecocf",
                hidden_dims=HIDDEN_DIMS,
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            policies.append(agent)
        elif method == "sdp_avg":
            preprocessor = SDP_avgstate(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                reg_model="nn",
                is_normalized=False,
                hidden_dims=HIDDEN_DIMS,
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=1,
                action_size=2,
                name="sdp_avg",
                hidden_dims=HIDDEN_DIMS,
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            policies.append(agent)
        elif method == "sdp_avg_oracle":
            preprocessor = SDP_avgstate_Oracle(
                env=env,
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=1,
                action_size=2,
                name="avg_oracle",
                hidden_dims=HIDDEN_DIMS,
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            policies.append(agent)
        elif method == "sdp_noreward":
            preprocessor = SDP_noreward(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                reg_model="nn",
                is_normalized=False,
                hidden_dims=HIDDEN_DIMS,
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=2,
                action_size=2,
                name="sdp_noreward",
                hidden_dims=HIDDEN_DIMS,
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            policies.append(agent)
        elif method == "sdp_noreward_oracle":
            preprocessor = SDP_noreward_Oracle(
                env=env,
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=2,
                action_size=2,
                name="sdp_noreward_oracle",
                hidden_dims=HIDDEN_DIMS,
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            policies.append(agent)
        elif method == "sdp_nostate":
            preprocessor = SDP_nostate(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                reg_model="nn",
                is_normalized=False,
                hidden_dims=HIDDEN_DIMS,
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=1,
                action_size=2,
                name="sdp_nostate",
                hidden_dims=HIDDEN_DIMS,
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            policies.append(agent)
        elif method == "sdp_nostate_oracle":
            preprocessor = SDP_nostate_Oracle(
                env=env,
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                state_size=1,
                action_size=2,
                name="sdp_nostate_oracle",
                hidden_dims=HIDDEN_DIMS,
            )
            agent.train(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
                max_iter=max_iters,
                inner_iter=inner_iter,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )
            policies.append(agent)
        elif method == "mopo":
            preprocessor = SDP(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                is_early_stopping=False,
                test_size=0.2,
                early_stopping_min_delta=0.001,
                early_stopping_patience=10,
                reg_model="nn",
                is_normalized=False,
                hidden_dims=HIDDEN_DIMS,
                # cross_folds=10,
            )
            agent = MOPO_SDP_QLearning(
                sdp_preprocessor=preprocessor,
                rollout_length=1,
                rollout_batch_size=100,
                num_iterations=5000,
                batch_size=64,
                real_ratio=0.1,
                hidden_dims=HIDDEN_DIMS,
            )
            agent.name = "MOPO_SDP_QLearning"
            agent.update_offline_buffer(
                copy.deepcopy(xs),
                copy.deepcopy(zs),
                copy.deepcopy(actions),
                copy.deepcopy(rewards),
            )
            agent.train()
            policies.append(agent)
        else:
            raise ValueError("Method not found")

    # evaluate reward and fairness
    df_n = pd.DataFrame()
    for policy in policies:
        discounted_cumulative_reward, mean_actions = evaluate_reward_through_simulation(
            env=env, N=N_eval, T=T_eval, policy=policy, seed=eval_seed
        )
        cf_outs_given_observed_actions = (
            evaluate_fairness_through_simulation_given_observed_actions(
                env, N_eval, T_eval, policy, seed=eval_seed
            )
        )

        df_n = pd.concat(
            [
                df_n,
                pd.DataFrame(
                    {
                        "N": [N],
                        "T": [T],
                        "cf_g": [cf_outs_given_observed_actions[0]],
                        "reward": [discounted_cumulative_reward],
                        "mean_action": [mean_actions],
                        "method": [policy.name],
                        "zcoef": [z_coef],
                    }
                ),
            ]
        )
    return df_n


def run_exp1():
    NREP = 50
    CORES = 20

    # N effect
    # Ns = [30, 50, 70, 100, 200, 500, 1000, 2000]
    # Ns = [50]
    # Ts = [10]
    # z_coefs = [1]

    # T effect
    Ns = [1000]
    Ts = [10, 20, 30, 40, 50]
    z_coefs = [1]

    # z_coef effect
    # Ns = [1000]
    # Ts = [10]
    # z_coefs = [0, 0.25, 0.5, 0.75, 1.0]
    # z_coefs = [0, 0.5, 1]

    env_name = "nlm3sr"
    # suffix = datetime_suffix()
    method_policy = "FQI_NN"

    methods = [
        "full",
        "ecocf",
        "random",
        # "behavior",
        # "sdp_nostate",
        # "sdp_noreward",
        # "sdp_noreward_oracle",
        # "sdp_nostate_oracle",
        "unaware",
        "ours",
        # "oracle",
        "flap",
        # "sdp_avg",
        # "sdp_avg_oracle",
        "ours_cf10",
        "mopo",
    ]

    mp.set_start_method("spawn")

    df_n = pd.DataFrame()
    jobs = []
    for N in Ns:
        for T in Ts:
            for z_coef in z_coefs:
                for _ in range(NREP):
                    jobs.append(
                        (
                            env_name,
                            methods,
                            method_policy,
                            N,
                            T,
                            z_coef,
                            _ + 10,
                        )
                    )
    with mp.Pool(CORES) as p:
        out = list(
            tqdm(
                p.imap(run_exp1_one_star, jobs),
                total=len(jobs),
                desc="run exp",
            )
        )

    for d in out:
        df_n = pd.concat([df_n, d])

    # unaggregate outfile
    df_n.round(3).to_csv(
        "../out/exp1_{}_N_{}_T_{}_zoef{}_method{}.csv".format(
            env_name,
            "_".join([str(N) for N in Ns]),
            "_".join([str(T) for T in Ts]),
            "_".join([str(z_coef) for z_coef in z_coefs]),
            method_policy,
        ),
        index=False,
    )


def run_exp1_one_star(args):
    return run_exp1_one(*args)


if __name__ == "__main__":
    run_exp1()
