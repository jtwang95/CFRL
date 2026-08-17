import numpy as np


# action model
class RandomAgent:
    def __init__(self) -> None:
        self.name = "random"
        self.state_size = 0  # not used

    def act(self, uat=None, is_return_prob=False, **kargs):
        probs = np.ones([uat.shape[0], 3]) * 1 / 3
        cdf = np.cumsum(probs, axis=1)

        actions = np.zeros([uat.shape[0]], dtype=int)
        visits = np.zeros_like(actions)
        if uat is not None:
            for i in range(cdf.shape[1]):
                masks = (uat.flatten() <= cdf[:, i]).astype(int) - visits
                actions += masks * i
                visits += masks
        else:
            uat = np.random.uniform(size=uat.shape[0])
            for i in range(cdf.shape[1]):
                masks = (uat.flatten() <= cdf[:, i]).astype(int) - visits
                actions += masks * i
                visits += masks

        if is_return_prob:
            return probs
        else:
            return actions
