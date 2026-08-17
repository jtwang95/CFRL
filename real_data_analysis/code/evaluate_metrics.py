import numpy as np
from fqe_new import FQE_NEW as FQE


def evaluate_fairness_through_model(
    agent, powered_env, xs_test, zs_test, actions_test, zvars_cat, **kwargs
):

    N, T, xdim = xs_test.shape

    xs_test_c = np.zeros_like(xs_test)
    zs_test_c = 1 - zs_test

    uxt = np.zeros_like(xs_test)

    model0_mean = powered_env.initial_state_model
    for z_ in [0, 1]:
        idx = zs_test.flatten() == z_
        xs_test_c[idx, 0, :] = xs_test[idx, 0] - model0_mean[z_] + model0_mean[1 - z_]
        uxt[idx, 0, :] = xs_test[idx, 0] - model0_mean[z_]

    for t in range(1, T):
        xs_test_c[:, t, :] = (
            xs_test[:, t, :]
            - powered_env.next_state_reward_mean(
                xs_test[:, t - 1, :],
                zs_test,
                actions_test[:, t - 1],
            )[0]
            + powered_env.next_state_reward_mean(
                xs_test_c[:, t - 1, :],
                1 - zs_test,
                actions_test[:, t - 1],
            )[0]
        )

    actions_factual = np.zeros([N, T])

    uat = np.random.uniform(size=[xs_test.shape[0], T])

    for t in range(T):
        if t == 0:
            actions_taken = agent.act(
                xt=xs_test[:, 0],
                z=zs_test.copy(),
                xtm1=None,
                atm1=None,
                uat=uat[:, 0],
                t=t,
            )
        else:
            actions_taken = agent.act(
                xt=xs_test[:, t],
                z=zs_test.copy(),
                xtm1=xs_test[:, t - 1],
                atm1=actions_test[:, t - 1],
                uat=uat[:, t],
                t=t,
            )

        actions_factual[:, t] = actions_taken.flatten()

    actions_counterfactual = np.zeros([N, T])
    for t in range(T):
        if t == 0:
            actions_taken = agent.act(
                xt=xs_test_c[:, 0],
                z=zs_test_c.copy(),
                xtm1=None,
                atm1=None,
                uat=uat[:, 0],
                t=t,
            )
        else:
            actions_taken = agent.act(
                xt=xs_test_c[:, t],
                z=zs_test_c.copy(),
                xtm1=xs_test_c[:, t - 1],
                atm1=actions_test[:, t - 1],
                uat=uat[:, t],
                t=t,
            )

        actions_counterfactual[:, t] = actions_taken.flatten()

    print(np.mean(actions_factual == actions_counterfactual, axis=0))
    cf_metric = np.mean(actions_factual != actions_counterfactual)

    # identify who is disadvantaged
    idx_z0 = zs_test.flatten() == 0
    idx_z1 = zs_test.flatten() == 1
    actions_z0 = np.zeros([N, T])
    actions_z1 = np.zeros([N, T])
    actions_z0[idx_z0] = actions_factual[idx_z0]
    actions_z0[idx_z1] = actions_counterfactual[idx_z1]
    actions_z1[idx_z0] = actions_counterfactual[idx_z0]
    actions_z1[idx_z1] = actions_factual[idx_z1]
    actions_z1gez0 = actions_z1 > actions_z0
    actions_z1lez0 = actions_z1 < actions_z0
    actions_z1eqz0 = actions_z1 == actions_z0

    return (
        cf_metric,
        np.mean(actions_z1gez0),
        np.mean(actions_z1lez0),
        np.mean(actions_z1eqz0),
        np.mean(actions_factual == 0),
        np.mean(actions_factual == 1),
        np.mean(actions_factual == 2),
    )


def evaluate_reward_through_fqe(
    agent,
    xs_test,
    zs_test,
    actions_test,
    rewards_test,
    hidden_dims,
    inner_iter,
    max_iter,
    learning_rate,
    xs_train,
    zs_train,
    actions_train,
    **kwargs,
):

    N, T, xdim = xs_test.shape

    # FQE estimate
    num_tries = 0
    while num_tries < 10:
        fqe = FQE(
            action_size=3,
            policy=agent,
            n_actions=3,
            hidden_dims=hidden_dims,
        )
        fqe.fit(
            xs=xs_test,
            zs=zs_test,
            actions=actions_test,
            rewards=rewards_test,
            inner_iter=inner_iter,
            max_iter=max_iter,
            learning_rate=learning_rate,
        )
        value, fqe_final_loss = fqe.evaluate(
            xs=xs_train, zs=zs_train, actions=actions_train
        )
        if fqe_final_loss < 3:
            break
        else:
            print(
                "FQE final loss too large, try to refit the FQE model. fqe_final_loss: {}, num_tries: {}",
                fqe_final_loss,
                num_tries,
            )
        num_tries += 1

    return np.mean(value), fqe_final_loss
