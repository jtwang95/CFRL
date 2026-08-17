import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import copy
import torch

sys.path.extend(["../../"])

from core.preprocessing_baselines import UnawarenessPreprocess, ConcatenateProprocess

from policy_learning.FQI import OfflineFQIWrapped
from random_agent import RandomAgent
from evaluate_metrics import (
    evaluate_fairness_through_model,
    evaluate_reward_through_fqe,
)
from semi_synthetic.code.powered_env import PowerEDEnv
from core.preprocessing import SDP
from core.preprocessing_baselines2 import FLAP
from core.postprocessing_baselines import ECOCF

# from utils import glogger
from core.utils import datetime_suffix, glogger
import multiprocessing as mp
from tqdm import tqdm

glogger.setLevel("INFO")

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "../data/")
SDP_EPOCHS = 1000
FQI_ITERS = 100
FQE_ITERS = 100
INNER_ITERS = 1000
ENCODE_ACTIONS = True
HIDDEN_DIMS = [128, 128]
ENV_HIDDEN_DIMS = [64, 64]
LR = 0.01
EARLY_STOPPING = True
FQE_LR = 0.01
FQE_INNER_ITERS = 500

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def run_exp(seed, zvars):
    xvars = ["weekly_pain_score", "weekly_pain_interference"]
    xs_, actions_, rewards_, zs_ = PowerEDEnv.load_training_data(
        xvars=xvars, zvars_cat=zvars
    )

    # xs: (N,T,xdim), actions: (N,T-1), rewards: (N,T-1), zs: (N,1)
    actions_ = actions_[:, 1:]
    rewards_ = rewards_[:, 1:]

    (xs, xs_test, actions, actions_test, rewards, rewards_test, zs, zs_test) = (
        train_test_split(xs_, actions_, rewards_, zs_, test_size=0.2, random_state=seed)
    )
    np.random.seed(seed)
    torch.manual_seed(seed)

    # env: use all data to train the environment
    powered_env = PowerEDEnv(xvars=xvars, zvars_cat=zvars)
    powered_env.fit(
        epochs=1000, lr=1e-4, hidden_dims=ENV_HIDDEN_DIMS, early_stopping=EARLY_STOPPING
    )

    # training
    agents = [RandomAgent()]
    # agents = []
    final_fqi_loss = {"random": 0}
    ###########################################################
    preprocessor = SDP(
        xs=copy.deepcopy(xs),
        zs=copy.deepcopy(zs),
        actions=copy.deepcopy(actions),
        rewards=copy.deepcopy(rewards),
        reg_model="nn",
        is_normalized=False,
        is_early_stopping=True,
        test_size=0.2,
        cross_folds=1,
        mode="single",
        hidden_dims=ENV_HIDDEN_DIMS,
        learning_rate=1e-4,
        batch_size=128,
        epochs=SDP_EPOCHS,
        early_stopping_min_delta=0.00,
        early_stopping_patience=5,
    )
    agent = OfflineFQIWrapped(
        preprocessor=preprocessor,
        state_size=preprocessor.xdim * 2,
        action_size=3,
        name="ours_cf1_single",
        hidden_dims=HIDDEN_DIMS,
        device=device,
    )
    final_fqi_loss[agent.name] = agent.train(
        copy.deepcopy(xs),
        copy.deepcopy(zs),
        copy.deepcopy(actions),
        copy.deepcopy(rewards),
        max_iter=FQI_ITERS,
        inner_iter=INNER_ITERS,
        learning_rate=LR,
    )
    agents.append(agent)

    ############################################################
    preprocessor = SDP(
        xs=copy.deepcopy(xs),
        zs=copy.deepcopy(zs),
        actions=copy.deepcopy(actions),
        rewards=copy.deepcopy(rewards),
        reg_model="nn",
        is_normalized=False,
        is_early_stopping=True,
        cross_folds=20,
        mode="single",
        hidden_dims=HIDDEN_DIMS,
        learning_rate=1e-4,
        batch_size=128,
        epochs=SDP_EPOCHS,
        early_stopping_min_delta=0.00,
        early_stopping_patience=5,
    )
    agent = OfflineFQIWrapped(
        preprocessor=preprocessor,
        state_size=preprocessor.xdim * 2,
        action_size=3,
        name="ours_cf20_single",
        hidden_dims=HIDDEN_DIMS,
    )
    final_fqi_loss[agent.name] = agent.train(
        copy.deepcopy(xs),
        copy.deepcopy(zs),
        copy.deepcopy(actions),
        copy.deepcopy(rewards),
        max_iter=FQI_ITERS,
        inner_iter=INNER_ITERS,
        learning_rate=LR,
    )
    agents.append(agent)

    ############################################################
    preprocessor = ConcatenateProprocess()
    agent = OfflineFQIWrapped(
        preprocessor=preprocessor,
        state_size=3,
        action_size=3,
        name="full",
        hidden_dims=HIDDEN_DIMS,
    )
    final_fqi_loss[agent.name] = agent.train(
        copy.deepcopy(xs),
        copy.deepcopy(zs),
        copy.deepcopy(actions),
        copy.deepcopy(rewards),
        max_iter=FQI_ITERS,
        inner_iter=INNER_ITERS,
        learning_rate=LR,
    )
    agents.append(agent)

    ###########################################################
    preprocessor = UnawarenessPreprocess()
    agent = OfflineFQIWrapped(
        preprocessor=preprocessor,
        state_size=2,
        action_size=3,
        name="unaware",
        hidden_dims=HIDDEN_DIMS,
    )
    final_fqi_loss[agent.name] = agent.train(
        copy.deepcopy(xs),
        copy.deepcopy(zs),
        copy.deepcopy(actions),
        copy.deepcopy(rewards),
        max_iter=FQI_ITERS,
        inner_iter=INNER_ITERS,
        learning_rate=LR,
    )
    agents.append(agent)

    # # ############################################################
    preprosessor = FLAP(
        xs=copy.deepcopy(xs),
        zs=copy.deepcopy(zs),
        actions=copy.deepcopy(actions),
        rewards=copy.deepcopy(rewards),
    )
    agent = OfflineFQIWrapped(
        preprocessor=preprosessor,
        state_size=2,
        action_size=3,
        hidden_dims=HIDDEN_DIMS,
        name="FLAP",
    )
    final_fqi_loss[agent.name] = agent.train(
        copy.deepcopy(xs),
        copy.deepcopy(zs),
        copy.deepcopy(actions),
        copy.deepcopy(rewards),
        max_iter=FQI_ITERS,
        inner_iter=INNER_ITERS,
        learning_rate=LR,
        discount_factor=0.9,
    )
    agents.append(agent)

    # # ############################################################
    agent = ECOCF(
        state_size=3,
        action_size=3,
        hidden_dims=HIDDEN_DIMS,
        name="ECOCF",
    )
    final_fqi_loss[agent.name] = agent.train(
        copy.deepcopy(xs),
        copy.deepcopy(zs),
        copy.deepcopy(actions),
        copy.deepcopy(rewards),
        max_iter=FQI_ITERS,
        inner_iter=INNER_ITERS,
        learning_rate=LR,
        discount_factor=0.9,
    )
    agents.append(agent)
    ##########################################################
    # evaluation
    res = pd.DataFrame()

    for agent in agents:
        (
            fair_metric,
            mean_actions_z1gez0,
            mean_actions_z1lez0,
            mean_actions_z1eqz0,
            action_0_prop,
            action_1_prop,
            action_2_prop,
        ) = evaluate_fairness_through_model(
            agent, powered_env, xs_test, zs_test, actions_test, zvars
        )
        value, fqe_final_loss = evaluate_reward_through_fqe(
            agent,
            xs_test,
            zs_test,
            actions_test,
            rewards_test,
            hidden_dims=HIDDEN_DIMS,
            inner_iter=FQE_INNER_ITERS,
            max_iter=FQE_ITERS,
            learning_rate=FQE_LR,
            xs_train=xs,
            zs_train=zs,
            actions_train=actions,
        )

        tmp = pd.DataFrame(
            {
                "method": [agent.name],
                "zvar": [zvars],
                "fair_metric": [fair_metric],
                "value": [value],
                "fqe_final_loss": [fqe_final_loss],
                "final_fqi_loss": [final_fqi_loss[agent.name]],
                "mean_actions_z1gez0": [mean_actions_z1gez0],
                "mean_actions_z1lez0": [mean_actions_z1lez0],
                "mean_actions_z1eqz0": [mean_actions_z1eqz0],
                "action_0_prop": [action_0_prop],
                "action_1_prop": [action_1_prop],
                "action_2_prop": [action_2_prop],
                "seed": [seed],
            }
        )
        res = pd.concat([res, tmp], axis=0, ignore_index=True)
    return res


def run_exp_star(args):
    return run_exp(*args)


if __name__ == "__main__":
    NREP = 20
    N_CORES = 5
    zvars = ["education_b", "race_b", "sex_b", "ethnicity_b"]

    mp.set_start_method("spawn", force=True)
    res = pd.DataFrame()
    jobs = []
    for zvar in zvars:
        for i in range(10, NREP + 10):
            jobs.append((i, zvar))
    with mp.Pool(N_CORES) as p:
        out = list(
            tqdm(
                p.imap(run_exp_star, jobs),
                total=len(jobs),
            )
        )
    res = pd.concat(out, axis=0, ignore_index=True)

    res.to_csv(
        os.path.join(
            os.path.dirname(__file__), "../results/real_data_results_{}.csv"
        ).format(datetime_suffix()),
        index=False,
    )
    print(res)
    agg = (
        res.groupby(["method", "zvar"])
        .agg(
            {
                "fair_metric": ["mean", "std"],
                "value": ["mean", "std"],
            }
        )
        .reset_index()
    )
    print(agg)
