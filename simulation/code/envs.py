"""
this file contains the environment class for the simulation study
"""

import numpy as np
from scipy.special import expit
import copy


class BehaviorPolicy:
    def __init__(self, seed=1) -> None:
        super().__init__()
        np.random.seed(seed)
        self.eta = np.array([-1.39, 0.0, 2.77, 0.0])
        self.name = "behavior"

    def act(self, xt, z, uat, is_return_prob=False, **kwargs):
        n = xt.shape[0]
        M = np.concatenate(
            [np.ones([n, 1]), xt, z, xt * z],
            axis=1,
        )
        ps = expit(M @ self.eta)
        action_behavior = (uat.flatten() <= ps.flatten()).astype(int)
        action_random = (uat.flatten() <= 0.5).astype(int)

        idx_random = np.random.uniform(0, 1, size=[n]) <= 0.0  # epsilon-greedy
        action = action_behavior * (1 - idx_random) + action_random * idx_random
        if is_return_prob:
            return np.vstack([1 - ps, ps]).T
        else:
            return action


class RandomPolicy:
    def __init__(self, p1=0.5) -> None:
        self.name = "random"
        self.p1 = p1

    def act(self, uat, is_return_prob=False, **kwargs):
        action = (uat.flatten() <= self.p1).astype(int)
        # logits = np.ones_like(uat) * logit(self.p1)
        p1 = np.ones_like(uat) * self.p1
        p0 = 1 - p1
        if is_return_prob:
            return np.vstack([p0, p1]).T
        else:
            return action


def get_env_settings(name, z_coef):
    if name == "lm1":

        def f_x0(z, ux0):
            gamma0 = np.array([-0.7, 0.8 * z_coef, 1])
            n = z.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    z,
                    ux0,
                ],
                axis=1,
            )
            x0 = M @ gamma0
            return x0

        def f_xt(xtm1, z, atm1, uxt):
            gamma = np.array([-0.7, 0.8 * z_coef, 0.5, 0.4, 1])
            n = xtm1.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    z,
                    xtm1,
                    atm1.reshape(-1, 1) - 0.5,
                    uxt,
                ],
                axis=1,
            )
            xt = M @ gamma
            return xt

        def f_rt(xt, z, at, urt):
            lmbda = np.array([-0.2, 0.3, 0.8, 0.8, -0.6, -0.7, -1.6])
            n = xt.shape[0]
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [np.ones([n, 1]), xt, z, at, xt * z, xt * at, z * at], axis=1
            )
            rt = M @ lmbda
            return rt

    elif name == "lm2":

        def f_x0(z, ux0):
            gamma0 = np.array([-0.7, 0.8 * z_coef, 1])
            n = z.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    z,
                    ux0,
                ],
                axis=1,
            )
            x0 = M @ gamma0
            return x0

        def f_xt(xtm1, z, atm1, uxt):
            gamma = np.array([-0.7, 0.8 * z_coef, 0.5, 0.4, 0.3, 0.2, 0.4 * z_coef, 1])
            n = xtm1.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    z,
                    xtm1,
                    atm1.reshape(-1, 1) - 0.5,
                    xtm1 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1 * z,
                    z * (atm1.reshape(-1, 1) - 0.5),
                    uxt,
                ],
                axis=1,
            )
            xt = M @ gamma
            return xt

        def f_rt(xt, z, at, urt):
            lmbda = np.array([-0.2, 0.3, 0.8, 0.8, -0.6, -0.7, -1.6])
            n = xt.shape[0]
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [np.ones([n, 1]), xt, z, at, xt * z, xt * at, z * at], axis=1
            )
            rt = M @ lmbda
            return rt

    elif name == "lm3":

        def f_x0(z, ux0):
            gamma0 = np.array([0.2, 0.8 * z_coef, 1])
            n = z.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    z,
                    ux0,
                ],
                axis=1,
            )
            x0 = M @ gamma0
            return x0

        def f_xt(xtm1, z, atm1, uxt):
            gamma = np.array(
                [0.2, 0.8 * z_coef, 0.5, 0.4, 0.3, 0.3 * z_coef, 0.4 * z_coef, 1]
            )
            n = xtm1.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    (z - 0.5),
                    xtm1,
                    atm1.reshape(-1, 1) - 0.5,
                    xtm1 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1 * (z - 0.5),
                    (z - 0.5) * (atm1.reshape(-1, 1) - 0.5),
                    uxt,
                ],
                axis=1,
            )
            xt = M @ gamma
            return xt

        def f_rt(xt, z, at, urt):
            lmbda = np.array(
                [0.2, 0.3, 0.4 * z_coef, 0.3, 0.2 * z_coef, 0.3, -1.0 * z_coef]
            )
            n = xt.shape[0]
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [np.ones([n, 1]), xt, z, at, xt * z, xt * at, z * at],
                axis=1,
            )
            rt = M @ lmbda
            return rt

    elif name == "lm4":

        def f_x0(z, ux0):
            gamma0 = np.array([-0.3, 1 * z_coef, 1])
            n = z.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    z,
                    ux0,
                ],
                axis=1,
            )
            x0 = M @ gamma0
            return x0

        def f_xt(xtm1, z, atm1, uxt):
            gamma = np.array(
                [-0.3, 1 * z_coef, 0.5, 0.4, 0.3, 0.3 * z_coef, 0.4 * z_coef, 1]
            )
            n = xtm1.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    (z - 0.5),
                    xtm1,
                    atm1.reshape(-1, 1) - 0.5,
                    xtm1 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1 * (z - 0.5),
                    (z - 0.5) * (atm1.reshape(-1, 1) - 0.5),
                    uxt,
                ],
                axis=1,
            )
            xt = M @ gamma
            return xt

        def f_rt(xt, z, at, urt):
            lmbda = np.array(
                [-0.3, 0.3, 0.5 * z_coef, 0.5, 0.2 * z_coef, 0.7, -1.0 * z_coef]
            )
            n = xt.shape[0]
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [np.ones([n, 1]), xt, z, at, xt * z, xt * at, z * at],
                axis=1,
            )
            rt = M @ lmbda
            return rt

    elif name == "nlm1":

        def f_x0(z, ux0):
            gamma0 = np.array([-0.7, 0.8 * z_coef, 1])
            n = z.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    z,
                    ux0,
                ],
                axis=1,
            )
            x0 = M @ gamma0
            return x0

        def f_xt(xtm1, z, atm1, uxt):
            gamma = np.array(
                [-1, 0.8 * z_coef, 0.5, 0.4, 0.3, 0.3 * z_coef, 0.4 * z_coef, 1]
            )
            n = xtm1.shape[0]
            xtm1 = (np.sin(xtm1) + np.cos(xtm1)) / 2
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    z,
                    xtm1,
                    atm1.reshape(-1, 1) - 0.5,
                    xtm1 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1 * z,
                    z * (atm1.reshape(-1, 1) - 0.5),
                    uxt,
                ],
                axis=1,
            )
            xt = M @ gamma
            return xt

        def f_rt(xt, z, at, urt):
            lmbda = np.array(
                [-0.2, 0.3, 0.8 * z_coef, 0.8, -0.6 * z_coef, -0.7, -1.6 * z_coef]
            )
            n = xt.shape[0]
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [np.ones([n, 1]), xt, z, at, xt * z, xt * at, z * at], axis=1
            )
            rt = M @ lmbda
            return rt

    elif name == "nlm2":

        def f_x0(z, ux0):
            gamma0 = np.array([-0.7, 0.8 * z_coef, 1])
            n = z.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    z,
                    ux0,
                ],
                axis=1,
            )
            x0 = M @ gamma0
            return x0

        def f_xt(xtm1, z, atm1, uxt):
            gamma = np.array(
                [-1, 0.8 * z_coef, 0.5, 0.4, 0.3, 0.3 * z_coef, 0.4 * z_coef, 1]
            )
            n = xtm1.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    z,
                    xtm1,
                    atm1.reshape(-1, 1) - 0.5,
                    np.abs(xtm1) * (atm1.reshape(-1, 1) - 0.5),
                    xtm1 * z,
                    (z - 0.5) * (atm1.reshape(-1, 1) - 0.5),
                    uxt,
                ],
                axis=1,
            )
            xt = M @ gamma
            return xt

        def f_rt(xt, z, at, urt):
            lmbda = np.array(
                [-0.2, 0.3, 0.8 * z_coef, 0.8, -0.6 * z_coef, -0.7, -1.6 * z_coef, 0.5]
            )
            n = xt.shape[0]
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    xt,
                    z,
                    at,
                    np.abs(xt) * z,
                    xt * at,
                    z * at,
                    (at - 0.5) * urt,
                ],
                axis=1,
            )
            rt = M @ lmbda
            return rt

    elif name == "nlm3":
        # similar to nlm1, add non-linearity in reward function

        def f_x0(z, ux0):
            gamma0 = np.array([-0.7, 0.8 * z_coef, 1])
            n = z.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    z,
                    ux0,
                ],
                axis=1,
            )
            x0 = M @ gamma0
            return x0

        def f_xt(xtm1, z, atm1, uxt):
            gamma = np.array([-1, 0.8, 0.9, 0.4, 0.7, 0.3, 0.8, 0.8, 1])
            n = xtm1.shape[0]
            # xtm1 = 3 * np.sin(xtm1 * 4) + 2 * np.cos(xtm1)
            xtm1 = np.sign(xtm1) * np.abs(xtm1) ** 0.5
            M = np.concatenate(
                [
                    np.ones([n, 1]) * -1,
                    z * 0.8,
                    xtm1 * 0.5,
                    (atm1.reshape(-1, 1) - 0.5) * -0.4,
                    (xtm1 * (atm1.reshape(-1, 1) - 0.5)) * 0.2,
                    (xtm1 * z) * 0.6,
                    (z * (atm1.reshape(-1, 1) - 0.5)) * -0.8,
                    (z * (atm1.reshape(-1, 1) - 0.5) * xtm1) * 0.8,
                    uxt,
                ],
                axis=1,
            )
            xt = M @ np.ones(
                [
                    9,
                ]
            )
            return xt

        def f_rt(xt, z, at, urt):
            lmbda = np.array([-0.2, 0.3, 0.4, 0.8, -1, 0.6, -1.6, -0.5])
            n = xt.shape[0]
            # xt = np.sin(xt) + np.cos(xt)
            xt = np.sign(xt) * np.abs(xt) ** (1 / 2)
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [
                    (np.ones([n, 1])) * 0.1,
                    xt * 0.3,
                    z * 0.4,
                    (at - 0.5) * 0.3,
                    xt * z * -1,
                    xt * (at - 0.5) * 0.6,
                    z * (at - 0.5) * -1.6,
                    z * (at - 0.5) * xt * 0.5,
                ],
                axis=1,
            )
            rt = M @ np.ones(
                [
                    8,
                ]
            )
            return rt

    elif name == "nlm3s":

        def f_x0(z, ux0):
            gamma0 = np.array([-0.7, 0.8 * z_coef, 1])
            n = z.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    z,
                    ux0,
                ],
                axis=1,
            )
            x0 = M @ gamma0
            return x0

        def f_xt(xtm1, z, atm1, uxt):
            # gamma = np.array([-1, 0.8, 0.9, 0.4, 0.7, 0.3, 0.8, 0.8, 1])
            n = xtm1.shape[0]
            # xtm1 = 3 * np.sin(xtm1 * 4) + 2 * np.cos(xtm1)
            xtm1 = np.sign(xtm1) * np.abs(xtm1) ** 0.5
            ##ecocf, flap: change xtm1*z, xtm1*z*atm1
            ##
            M = np.concatenate(
                [
                    np.ones([n, 1]) * -1,
                    z * 0.8 * z_coef,
                    xtm1 * 0.5,
                    (atm1.reshape(-1, 1) - 0.5) * -0.8,
                    (xtm1 * (atm1.reshape(-1, 1) - 0.5)) * 0.6,
                    (xtm1 * z) * -0.6 * z_coef,
                    (z * (atm1.reshape(-1, 1) - 0.5)) * -0.8 * z_coef,
                    (z * (atm1.reshape(-1, 1) - 0.5) * xtm1) * 0.8 * z_coef,
                    uxt,
                ],
                axis=1,
            )
            # M = np.concatenate(
            #     [
            #         np.ones([n, 1]) * -1,
            #         z * 0.6 * z_coef,
            #         xtm1 * 0.5,
            #         (atm1.reshape(-1, 1) - 0.5) * -0.4,
            #         (xtm1 * (atm1.reshape(-1, 1) - 0.5)) * 0.2,
            #         (xtm1 * z) * -0.6 * z_coef,
            #         (z * (atm1.reshape(-1, 1) - 0.5)) * -0.8 * z_coef,
            #         (z * (atm1.reshape(-1, 1) - 0.5) * xtm1) * 0.8 * z_coef,
            #         uxt,
            #     ],
            #     axis=1,
            # )
            xt = M @ np.ones(
                [
                    9,
                ]
            )
            return xt

        def f_rt(xt, z, at, urt):
            lmbda = np.array([-0.2, 0.3, 0.4, 0.8, -1, 0.6, -1.6, -0.5])
            n = xt.shape[0]
            # xt = np.sin(xt) + np.cos(xt)
            xt = np.sign(xt) * np.abs(xt) ** (1 / 2)
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [
                    (np.ones([n, 1])) * 0.1,
                    xt * 0.3,
                    z * 0.4,
                    (at - 0.5) * 0.3,
                    xt * z * -1,
                    xt * (at - 0.5) * 0.6,
                    z * (at - 0.5) * -1.0,
                    z * (at - 0.5) * xt * 1.0,
                    abs(xt) ** 0.5 * 0.3,
                ],
                axis=1,
            )  # x_t * a_t is important for fair
            # M = np.concatenate(
            #     [
            #         (np.ones([n, 1])) * 0.1,
            #         xt * 0.3,
            #         z * 0.4,
            #         (at - 0.5) * 0.3,
            #         xt * z * -1,
            #         xt * (at - 0.5) * 0.6,
            #         z * (at - 0.5) * -1.0,
            #         z * (at - 0.5) * xt * 0.6,
            #     ],
            #     axis=1,
            # )  # x_t * a_t is important for fair
            rt = M @ np.ones(
                [
                    M.shape[-1],
                ]
            )
            return rt

    elif name == "nlm3sr":

        def f_x0(z, ux0):
            gamma0 = np.array([-0.7, 0.8 * z_coef, 1])
            n = z.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    z,
                    ux0,
                ],
                axis=1,
            )
            x0 = M @ gamma0
            return x0

        def f_xt(xtm1, z, atm1, uxt):
            # gamma = np.array([-1, 0.8, 0.9, 0.4, 0.7, 0.3, 0.8, 0.8, 1])
            n = xtm1.shape[0]
            # xtm1 = 3 * np.sin(xtm1 * 4) + 2 * np.cos(xtm1)
            xtm1 = np.sign(xtm1) * np.abs(xtm1) ** 0.5
            ##ecocf, flap: change xtm1*z, xtm1*z*atm1
            ##
            M = np.concatenate(
                [
                    np.ones([n, 1]) * -1,
                    z * 0.8 * z_coef,
                    xtm1 * 0.5,
                    (atm1.reshape(-1, 1) - 0.5) * -0.8,
                    (xtm1 * (atm1.reshape(-1, 1) - 0.5)) * 0.6,
                    (xtm1 * z) * 0.6 * z_coef,
                    (z * (atm1.reshape(-1, 1) - 0.5)) * -0.8 * z_coef,
                    (z * (atm1.reshape(-1, 1) - 0.5) * xtm1) * 0.8 * z_coef,
                    uxt,
                ],
                axis=1,
            )
            # M = np.concatenate(
            #     [
            #         np.ones([n, 1]) * -1,
            #         z * 0.6 * z_coef,
            #         xtm1 * 0.5,
            #         (atm1.reshape(-1, 1) - 0.5) * -0.4,
            #         (xtm1 * (atm1.reshape(-1, 1) - 0.5)) * 0.2,
            #         (xtm1 * z) * -0.6 * z_coef,
            #         (z * (atm1.reshape(-1, 1) - 0.5)) * -0.8 * z_coef,
            #         (z * (atm1.reshape(-1, 1) - 0.5) * xtm1) * 0.8 * z_coef,
            #         uxt,
            #     ],
            #     axis=1,
            # )
            xt = M @ np.ones(
                [
                    9,
                ]
            )
            return xt

        def f_rt(xt, z, at, urt):
            lmbda = np.array([-0.2, 0.3, 0.4, 0.8, -1, 0.6, -1.6, -0.5])
            n = xt.shape[0]
            # xt = np.sin(xt) + np.cos(xt)
            xt = np.sign(xt) * np.abs(xt) ** (1 / 2)
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [
                    (np.ones([n, 1])) * 0.1,
                    xt * 0.3,
                    z * 0.4 * z_coef,
                    (at - 0.5) * 0.3,
                    xt * z * -1 * z_coef,
                    xt * (at - 0.5) * 0.6,
                    z * (at - 0.5) * -1.6 * z_coef,
                    z * (at - 0.5) * xt * 0.5 * z_coef,
                    # abs(xt) ** 0.5 * 0.3,
                ],
                axis=1,
            )  # x_t * a_t is important for fair
            # M = np.concatenate(
            #     [
            #         (np.ones([n, 1])) * 0.1,
            #         xt * 0.3,
            #         z * 0.4,
            #         (at - 0.5) * 0.3,
            #         xt * z * -1,
            #         xt * (at - 0.5) * 0.6,
            #         z * (at - 0.5) * -1.0,
            #         z * (at - 0.5) * xt * 0.6,
            #     ],
            #     axis=1,
            # )  # x_t * a_t is important for fair
            rt = M @ np.ones(
                [
                    M.shape[-1],
                ]
            )
            return rt

    return f_x0, f_xt, f_rt


class SimpleEnv:
    def __init__(self, seed=1, sigma=1, env_name="lm1", z_coef=1) -> None:
        # self.seed = seed
        self.sigma = sigma
        self.z_coef = z_coef
        np.random.seed(seed)  # generate random parameters
        self.f_x0, self.f_xt, self.f_rt = get_env_settings(env_name, self.z_coef)

    def sample_trajectory(
        self,
        n,
        T,
        Z,
        seed,
        policy=None,
        include_counter=False,
    ):
        """
        include_counter: bool, whether to include the counterfactual trajectory,
            this is different from the definition in the paper
        """
        # isolate buffer
        policy_c = copy.deepcopy(policy)

        np.random.seed(seed)  # random traj
        Z = Z.reshape(n, -1)
        Z_c = 1 - Z

        X = np.zeros([n, T + 1, 1], dtype=float)
        A = np.zeros([n, T], dtype=int)
        R = np.zeros([n, T], dtype=float)

        X_c = np.zeros([n, T + 1, 1], dtype=float)
        A_c = np.zeros([n, T], dtype=int)
        R_c = np.zeros([n, T], dtype=float)

        # t == 0
        ux0 = np.random.normal(0, self.sigma, [n, 1])
        X[:, 0, 0] = self.f_x0(
            z=Z,
            ux0=ux0,
        )
        X_c[:, 0, 0] = self.f_x0(
            z=Z_c,
            ux0=ux0,
        )

        ua0 = np.random.uniform(0, 1, size=[n])

        A[:, 0] = policy.act(
            z=Z,
            xt=X[:, 0],
            atm=A[:, :0],
            t=0,
            xtm1=None,
            atm1=None,
            uat=ua0,
        )
        A_c[:, 0] = policy_c.act(
            z=Z_c, xt=X_c[:, 0], atm=A_c[:, :0], t=0, xtm1=None, atm1=None, uat=ua0
        )

        ur0 = np.random.normal(0, self.sigma, [n, 1])
        R[:, 0] = self.f_rt(xt=X[:, 0], z=Z, at=A[:, 0], urt=ur0)
        R_c[:, 0] = self.f_rt(xt=X_c[:, 0], z=Z_c, at=A_c[:, 0], urt=ur0)

        # t = 1 to T-1
        for t in range(1, T):
            uxt = np.random.normal(0, self.sigma, [n, 1])
            X[:, t, 0] = self.f_xt(xtm1=X[:, t - 1], z=Z, atm1=A[:, t - 1], uxt=uxt)
            X_c[:, t, 0] = self.f_xt(
                xtm1=X_c[:, t - 1], z=Z_c, atm1=A_c[:, t - 1], uxt=uxt
            )
            uat = np.random.uniform(0, 1, size=[n])
            # print("cf: uat", uat[:5])
            A[:, t] = policy.act(
                z=Z,
                xt=X[:, t],
                atm=A[:, :t],
                t=t,
                xtm1=X[:, t - 1],
                atm1=A[:, t - 1],
                uat=uat,
            )
            A_c[:, t] = policy_c.act(
                z=Z_c,
                xt=X_c[:, t],
                atm=A_c[:, :t],
                t=t,
                xtm1=X_c[:, t - 1],
                atm1=A_c[:, t - 1],
                uat=uat,
            )

            urt = np.random.normal(0, self.sigma, [n, 1])
            R[:, t] = self.f_rt(xt=X[:, t], z=Z, at=A[:, t], urt=urt)
            R_c[:, t] = self.f_rt(xt=X_c[:, t], z=Z_c, at=A_c[:, t], urt=urt)

        # t = T
        uxt = np.random.normal(0, self.sigma, [n, 1])
        X[:, T, 0] = self.f_xt(
            xtm1=X[:, T - 1],
            z=Z,
            atm1=A[:, T - 1],
            uxt=uxt,
        )
        X_c[:, T, 0] = self.f_xt(
            xtm1=X_c[:, T - 1],
            z=Z_c,
            atm1=A_c[:, T - 1],
            uxt=uxt,
        )
        out = [Z, X, A, R]
        if include_counter:
            out.extend([Z_c, X_c, A_c, R_c])
        return out

    def sample_trajectory_given_observed_actions(
        self,
        n,
        T,
        Z,
        seed,
        policy=None,
        include_counter=True,
    ):
        """
        this is a different version of sample trajectory, where the historical actions
        are observed actions for both factual trajectory and counterfactual trajectory
        """
        # to isolate buffer, we need to copy the policy
        policy_c = copy.deepcopy(policy)

        np.random.seed(seed)  # random traj
        Z = Z.reshape(n, -1)
        Z_c = 1 - Z

        X = np.zeros([n, T + 1, 1], dtype=float)
        A = np.zeros([n, T], dtype=int)
        R = np.zeros([n, T], dtype=float)

        X_c = np.zeros([n, T + 1, 1], dtype=float)
        A_c = np.zeros([n, T], dtype=int)
        R_c = np.zeros([n, T], dtype=float)

        # t == 0
        ux0 = np.random.normal(0, self.sigma, [n, 1])
        X[:, 0, 0] = self.f_x0(
            z=Z,
            ux0=ux0,
        )
        X_c[:, 0, 0] = self.f_x0(
            z=Z_c,
            ux0=ux0,
        )

        ua0 = np.random.uniform(0, 1, size=[n])

        A[:, 0] = policy.act(
            z=Z,
            xt=X[:, 0],
            atm=A[:, :0],
            t=0,
            xtm1=None,
            atm1=None,
            uat=ua0,
        )
        A_c[:, 0] = policy_c.act(
            z=Z_c,
            xt=X_c[:, 0],
            atm=A[:, :0],
            t=0,
            xtm1=None,
            atm1=None,
            uat=ua0,
        )

        ur0 = np.random.normal(0, self.sigma, [n, 1])
        # R[:, 0] = self.f_rt(xt=X[:, 0], z=Z, at=A[:, 0], urt=ur0)
        # R_c[:, 0] = self.f_rt(xt=X_c[:, 0], z=Z_c, at=A_c[:, 0], urt=ur0)

        # t = 1 to T-1
        for t in range(1, T):
            uxt = np.random.normal(0, self.sigma, [n, 1])
            X[:, t, 0] = self.f_xt(xtm1=X[:, t - 1], z=Z, atm1=A[:, t - 1], uxt=uxt)
            X_c[:, t, 0] = self.f_xt(
                xtm1=X_c[:, t - 1], z=Z_c, atm1=A[:, t - 1], uxt=uxt
            )  # change atm1 to oberseved actions
            # print("sim,", t, X[:5, t, :])
            uat = np.random.uniform(0, 1, size=[n])
            # print("cf_g: uat", uat[:5])
            A[:, t] = policy.act(
                z=Z,
                xt=X[:, t],
                atm=A[:, :t],
                t=t,
                xtm1=X[:, t - 1],
                atm1=A[:, t - 1],
                uat=uat,
            )
            A_c[:, t] = policy_c.act(
                z=Z_c,
                xt=X_c[:, t],
                atm=A[:, :t],
                t=t,
                xtm1=X_c[:, t - 1],
                atm1=A[:, t - 1],
                uat=uat,
            )
            # if np.any(A[:, t] != A_c[:, t]):
            #     print("A[:,t] != A_c[:,t]", t, A[:, t], A_c[:, t])
            #     # print which one is different
            #     idx_diff = np.where(A[:, t] != A_c[:, t])[0]
            #     print("idx_diff", idx_diff)
            # correct the buffer
            # if hasattr(policy_c, "preprocessor"):
            #     if hasattr(policy_c.preprocessor, "buffer"):
            #         X_tilde = {Z}

            urt = np.random.normal(0, self.sigma, [n, 1])
            # R[:, t] = self.f_rt(xt=X[:, t], z=Z, at=A[:, t], urt=urt)
            # R_c[:, t] = self.f_rt(xt=X_c[:, t], z=Z_c, at=A_c[:, t], urt=urt)

        # t = T
        uxt = np.random.normal(0, self.sigma, [n, 1])
        X[:, T, 0] = self.f_xt(
            xtm1=X[:, T - 1],
            z=Z,
            atm1=A[:, T - 1],
            uxt=uxt,
        )
        X_c[:, T, 0] = self.f_xt(
            xtm1=X_c[:, T - 1],
            z=Z_c,
            atm1=A[:, T - 1],
            uxt=uxt,
        )

        # check
        # for i in range(T):
        #     print("sim,", i, X[:5, i, :])
        #     if i == 5:
        #         break

        out = [Z, X, A, None]
        if include_counter:
            out.extend([Z_c, X_c, A_c, None])
        return out


# if __name__ == "__main__":
#     env = SimpleEnv(seed=1)
#     Z = np.random.binomial(1, 0.5, size=[10, 1])
#     np.random.seed(1)
#     Z, X, A, R, L, Z_c, X_c, A_c, R_c, L_c = env.sample_trajectory(
#         n=10, T=2, Z=Z, seed=1, policy=BehaviorPolicy(), include_counter=True
#     )
#     print(L)
