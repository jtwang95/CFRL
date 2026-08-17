import numpy as np
import pandas as pd
import sys
import os
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from powered_env import (
    generate_baseline_characteristics,
    generate_intial_states,
    generate_powered_trajectories,
    RandomAgent,
)


from core.preprocessing import SDP
from core.preprocessing_baselines import UnawarenessPreprocess, ConcatenateProprocess
from core.postprocessing_baselines import ECOCF
from core.preprocessing_baselines2 import FLAP

from core.utils import glogger
from policy_learning.FQI import OfflineFQIWrapped

import multiprocessing as mp
from tqdm import tqdm
import torch

glogger.setLevel("INFO")

GAMMA = 0.9
SDP_EPOCHS = 1000
LR = 0.01
EPOCHS_ENV = 1000
LR_ENV = 0.0002
T_SET = 12
INCLUDE_WEEK = False
FQI_ITERS = 100
HIDDEN_DIMS = [64, 64]
FQI_HIDDEN_DIMS = [64, 64]

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def get_z_xs_actions_rewards_from_data(data, xvars, zvar_cat):
    N = len(np.unique(data["id"]))
    T = T_SET
    xs = np.zeros([N, T, len(xvars)])
    actions = np.zeros([N, T], dtype=int)
    rewards = np.zeros([N, T])
    data = data.sort_values(["id", "week"])
    for week in range(1, T + 1):
        xs[:, week - 1, :] = data.loc[data["week"] == week, xvars].to_numpy()
        actions[:, week - 1] = data.loc[data["week"] == week, "action"].to_numpy()
        rewards[:, week - 1] = data.loc[data["week"] == week, "reward"].to_numpy()

    tmp = data.loc[data["week"] == 1, zvar_cat]
    ref_z = 1
    zs = (tmp == ref_z).to_numpy().astype(int)
    zs = zs.reshape(-1, 1)

    actions = actions[:, :-1]
    rewards = rewards[:, :-1]

    return zs, xs, actions, rewards, ref_z


def eval_cf_metric(
    powered_env,
    baseline_characteristic,
    baseline_characteristic_c,
    zvar_cat,
    xvars,
    z_factor,
    seed,
    policy,
    ref_z,
):
    N_eval = baseline_characteristic.shape[0]
    T = 12
    folder = None

    x0 = generate_intial_states(
        baseline_characteristics=baseline_characteristic,
        seed=seed,
        folder=folder,
        zvars_cat=zvar_cat,
        xvars=xvars,
        z_factor=z_factor,
    )[0].to_numpy()
    x0_c = generate_intial_states(
        baseline_characteristics=baseline_characteristic_c,
        seed=seed,
        folder=folder,
        zvars_cat=zvar_cat,
        xvars=xvars,
        z_factor=z_factor,
    )[0].to_numpy()

    z_cat = baseline_characteristic[[zvar_cat]]
    z_cat_c = baseline_characteristic_c[[zvar_cat]]

    # to isolate buffer, we need to copy the policy
    policy_c = copy.deepcopy(policy)

    xt = np.zeros([N_eval, T + 1, len(xvars)])
    rt = np.zeros([N_eval, T])
    at = np.zeros([N_eval, T])

    xt_c = np.zeros([N_eval, T + 1, len(xvars)])
    rt_c = np.zeros([N_eval, T])
    at_c = np.zeros([N_eval, T])

    tmp_z = (
        (baseline_characteristic[zvar_cat] == ref_z)
        .to_numpy()
        .astype(int)[:, np.newaxis]
    )
    tmp_z_c = (
        (baseline_characteristic_c[zvar_cat] == ref_z)
        .to_numpy()
        .astype(int)[:, np.newaxis]
    )

    xt[:, 0, :] = x0
    xt_c[:, 0, :] = x0_c
    uat = np.random.uniform(size=N_eval)
    at[:, 0] = policy.act(
        z=tmp_z,
        xt=xt[:, 0],
        atm=None,
        xtm1=None,
        atm1=None,
        uat=uat,
        t=0,
        is_return_prob=False,
    )
    at_c[:, 0] = policy_c.act(
        z=tmp_z_c,
        xt=xt_c[:, 0],
        atm=None,
        xtm1=None,
        atm1=None,
        uat=uat,
        t=0,
        is_return_prob=False,
    )

    for week in range(1, T):
        errors_next_states = np.random.multivariate_normal(
            mean=np.zeros(xt.shape[-1]),
            cov=np.diag(powered_env.next_state_var),
            size=xt.shape[0],
        )
        errors_rewards = np.random.normal(
            loc=0, scale=np.sqrt(powered_env.reward_var), size=xt.shape[0]
        )
        xt[:, week, :], rt[:, week - 1] = powered_env.next_state_reward_mean(
            xt=xt[:, week - 1, :],
            z_cat=z_cat,
            at=at[:, week - 1],
        )
        xt[:, week, :] += errors_next_states
        rt[:, week - 1] += errors_rewards

        xt_c[:, week, :], rt_c[:, week - 1] = powered_env.next_state_reward_mean(
            xt=xt_c[:, week - 1, :],
            z_cat=z_cat_c,
            at=at_c[:, week - 1],
        )
        xt_c[:, week, :] += errors_next_states
        rt_c[:, week - 1] += errors_rewards

        uat = np.random.uniform(size=N_eval)

        at[:, week] = policy.act(
            xt=xt[:, week],
            z=tmp_z,
            atm1=at[:, week - 1],
            xtm1=xt[:, week - 1],
            is_return_prob=False,
            uat=uat,
            t=week,
        )
        at_c[:, week] = policy_c.act(
            xt=xt_c[:, week],
            z=tmp_z_c,
            atm1=at[:, week - 1],
            xtm1=xt_c[:, week - 1],
            is_return_prob=False,
            uat=uat,
            t=week,
        )
    m = np.repeat(
        np.array([GAMMA**i for i in range(T)])[np.newaxis, :],
        repeats=N_eval,
        axis=0,
    )
    discounted_reward = np.mean(np.multiply(rt, m)) * T
    fairness_metric = np.mean(at != at_c)
    tmp = pd.DataFrame(
        {
            "method": [policy.name],
            "fair_metric": [fairness_metric],
            "discounted_reward": [discounted_reward],
            "z_factor": [z_factor],
        }
    )
    return tmp


def run_exp_one(N, seed, folder, state_variance_factor, z_factor, zvar_cat, xvars):

    inner_iter = 500

    np.random.seed(seed)
    torch.manual_seed(seed)

    simulated_data, powered_env = generate_powered_trajectories(
        N=N,
        T=T_SET,
        policy=RandomAgent(),
        seed=seed,
        zvars_cat=zvar_cat,
        xvars=xvars,
        epochs=EPOCHS_ENV,
        lr=LR_ENV,
        folder=folder,
        state_variance_factor=state_variance_factor,
        z_factor=z_factor,
    )
    # simulated_data.to_csv(
    #     os.path.join(folder, "data", "simulated_data_seed{}.csv".format(seed)),
    #     index=False,
    # )

    zs, xs, actions, rewards, ref_z = get_z_xs_actions_rewards_from_data(
        data=simulated_data,
        xvars=xvars,
        zvar_cat=zvar_cat,
    )

    for i in range(len(xvars)):
        glogger.debug(
            "{}, min:{}, max:{}".format(
                xvars[i], np.min(xs[:, :, i]), np.max(xs[:, :, i])
            )
        )

    agents = [RandomAgent()]
    d_final_fqi_loss = {}

    #########################################################
    preprosessor = ConcatenateProprocess()
    agent = OfflineFQIWrapped(
        preprocessor=preprosessor,
        state_size=3,
        action_size=3,
        hidden_dims=FQI_HIDDEN_DIMS,
        name="Full",
        device=device,
    )
    agent.train(
        copy.deepcopy(xs),
        copy.deepcopy(zs),
        copy.deepcopy(actions),
        copy.deepcopy(rewards),
        max_iter=FQI_ITERS,
        inner_iter=inner_iter,
        learning_rate=LR,
        discount_factor=0.9,
    )
    agents.append(agent)
    d_final_fqi_loss[agent.name] = agent.final_fqi_loss

    ###########################################################
    preprosessor = UnawarenessPreprocess()
    agent = OfflineFQIWrapped(
        preprocessor=preprosessor,
        state_size=2,
        action_size=3,
        hidden_dims=FQI_HIDDEN_DIMS,
        name="FTU",
        device=device,
    )
    agent.train(
        copy.deepcopy(xs),
        copy.deepcopy(zs),
        copy.deepcopy(actions),
        copy.deepcopy(rewards),
        max_iter=FQI_ITERS,
        inner_iter=inner_iter,
        learning_rate=LR,
        discount_factor=0.9,
    )
    agents.append(agent)
    d_final_fqi_loss[agent.name] = agent.final_fqi_loss

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
        hidden_dims=FQI_HIDDEN_DIMS,
        name="FLAP",
        device=device,
    )
    agent.train(
        copy.deepcopy(xs),
        copy.deepcopy(zs),
        copy.deepcopy(actions),
        copy.deepcopy(rewards),
        max_iter=FQI_ITERS,
        inner_iter=inner_iter,
        learning_rate=LR,
        discount_factor=0.9,
    )
    agents.append(agent)
    d_final_fqi_loss[agent.name] = agent.final_fqi_loss

    # # ############################################################
    agent = ECOCF(
        state_size=3,
        action_size=3,
        hidden_dims=FQI_HIDDEN_DIMS,
        name="ECOCF",
        device=device,
    )
    agent.train(
        copy.deepcopy(xs),
        copy.deepcopy(zs),
        copy.deepcopy(actions),
        copy.deepcopy(rewards),
        max_iter=FQI_ITERS,
        inner_iter=inner_iter,
        learning_rate=LR,
        discount_factor=0.9,
    )
    agents.append(agent)
    d_final_fqi_loss[agent.name] = agent.final_fqi_loss

    ###########################################################
    preprocessor = SDP(
        xs=copy.deepcopy(xs),
        zs=copy.deepcopy(zs),
        actions=copy.deepcopy(actions),
        rewards=copy.deepcopy(rewards),
        reg_model="nn",
        is_normalized=False,
        cross_folds=10,
        hidden_dims=[64, 64],
        is_early_stopping=True,
        test_size=0.2,
        early_stopping_min_delta=0.00,
        early_stopping_patience=5,
        learning_rate=0.0005,
        batch_size=128,
        epochs=SDP_EPOCHS,
    )
    agent = OfflineFQIWrapped(
        preprocessor=preprocessor,
        state_size=4,
        action_size=3,
        hidden_dims=FQI_HIDDEN_DIMS,
        name="sdp_cf10",
        device=device,
    )
    agent.train(
        copy.deepcopy(xs),
        copy.deepcopy(zs),
        copy.deepcopy(actions),
        copy.deepcopy(rewards),
        max_iter=FQI_ITERS,
        inner_iter=inner_iter,
        learning_rate=LR,
        discount_factor=0.9,
    )
    agents.append(agent)
    d_final_fqi_loss[agent.name] = agent.final_fqi_loss

    # #############################################################
    preprocessor = SDP(
        xs=copy.deepcopy(xs),
        zs=copy.deepcopy(zs),
        actions=copy.deepcopy(actions),
        rewards=copy.deepcopy(rewards),
        is_early_stopping=True,
        test_size=0.2,
        early_stopping_min_delta=0.00,
        early_stopping_patience=5,
        reg_model="nn",
        is_normalized=False,
        hidden_dims=[64, 64],
        mode="single",
        learning_rate=0.0005,
        batch_size=128,
        epochs=SDP_EPOCHS,
        cross_folds=1,
    )
    agent = OfflineFQIWrapped(
        preprocessor=preprocessor,
        state_size=4,
        action_size=3,
        hidden_dims=FQI_HIDDEN_DIMS,
        name="sdp",
        device=device,
    )
    agent.train(
        copy.deepcopy(xs),
        copy.deepcopy(zs),
        copy.deepcopy(actions),
        copy.deepcopy(rewards),
        max_iter=FQI_ITERS,
        inner_iter=inner_iter,
        learning_rate=LR,
        discount_factor=0.9,
    )
    agents.append(agent)
    d_final_fqi_loss[agent.name] = agent.final_fqi_loss

    # evaluation
    N_eval = 10000
    baseline_characteristic = generate_baseline_characteristics(
        n=N_eval, seed=seed, zvars_cat=zvar_cat
    )
    baseline_characteristic_c = baseline_characteristic.copy()

    if zvar_cat == "race_b":
        baseline_characteristic_c[zvar_cat].replace({0: 1, 1: 0}, inplace=True)
    elif zvar_cat == "education_b":
        baseline_characteristic_c[zvar_cat].replace({0: 1, 1: 0}, inplace=True)
    elif zvar_cat == "sex_b":
        baseline_characteristic_c[zvar_cat].replace({0: 1, 1: 0}, inplace=True)
    elif zvar_cat == "ethnicity_b":
        baseline_characteristic_c[zvar_cat].replace({0: 1, 1: 0}, inplace=True)

    res = pd.DataFrame()
    for agent in agents:
        tmp = eval_cf_metric(
            powered_env=powered_env,
            baseline_characteristic=baseline_characteristic,
            baseline_characteristic_c=baseline_characteristic_c,
            zvar_cat=zvar_cat,
            xvars=xvars,
            z_factor=z_factor,
            seed=seed,
            policy=agent,
            ref_z=ref_z,
        )
        tmp["final_fqi_loss"] = d_final_fqi_loss.get(agent.name, -1)
        res = pd.concat([res, tmp], axis=0, ignore_index=True)
    return res


def run_exp_one_star(args):
    return run_exp_one(*args)


if __name__ == "__main__":
    N_REPEAT = 50
    N_CORES = 1  # 20
    Ns = [1000]
    state_variance_factors = [1.0]
    z_factors = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # z_factors = [0.0]

    # ZVAR_CAT = "education_b"
    # ZVAR_CAT = "race_b"
    # ZVAR_CAT = "sex_b"
    ZVAR_CAT = "ethnicity_b"
    XVARS = ["weekly_pain_score", "weekly_pain_interference"]
    TRANS_MODEL_TYPE = "nn"
    REWARD_MODEL_TYPE = "nn"

    # create folder
    folder = os.path.join(
        os.path.dirname(__file__),
        "../results/{}_tm{}_rm{}_N{}_{}_T{}".format(
            ZVAR_CAT, TRANS_MODEL_TYPE, REWARD_MODEL_TYPE, Ns[0], Ns[-1], T_SET
        ),
    )
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "models"), exist_ok=True)
    os.makedirs(os.path.join(folder, "data"), exist_ok=True)

    # train transition model
    mp.set_start_method("spawn")
    _ = generate_powered_trajectories(
        N=10,
        T=T_SET,
        policy=RandomAgent(),
        seed=1,
        epochs=1000,
        lr=LR_ENV,
        zvars_cat=ZVAR_CAT,
        xvars=XVARS,
        folder=folder,
    )

    res = pd.DataFrame()
    jobs = []
    for N in Ns:
        for svf in state_variance_factors:
            for zf in z_factors:
                for nrep in range(N_REPEAT):
                    jobs.append((N, nrep + 1, folder, svf, zf, ZVAR_CAT, XVARS))
    with mp.Pool(N_CORES) as p:
        out = list(
            tqdm(
                p.imap(run_exp_one_star, jobs),
                total=len(jobs),
                desc="run exp",
            )
        )
    res = pd.concat(out, axis=0, ignore_index=True)

    res.to_csv(
        os.path.join(folder, "semi_synthetic_results.csv"),
        index=False,
    )
    print(res)
    print(
        res.groupby(["method", "z_factor"])[
            ["fair_metric", "discounted_reward", "final_fqi_loss"]
        ].mean()
    )
