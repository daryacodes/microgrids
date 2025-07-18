
---

# 🌍 env't: 

import gym
from gym import spaces
import numpy as np

class SmartMicrogridEnv(gym.Env):
    def __init__(self):
        super(SmartMicrogridEnv, self).__init__()

        self.max_storage = 100.0
        self.max_demand = 20.0
        self.max_generation = 25.0

        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)  # charge/discharge rate
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -10.0]),
            high=np.array([self.max_storage, self.max_demand, self.max_generation]),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.storage = 50.0
        self.demand = np.random.uniform(5.0, self.max_demand)
        self.generation = np.random.uniform(10.0, self.max_generation)
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.storage, self.demand, self.generation], dtype=np.float32)

    def step(self, action):
        action = np.clip(action[0], -10.0, 10.0)

        self.storage += action
        self.storage = np.clip(self.storage, 0, self.max_storage)

        net_energy = self.generation + (action if action < 0 else 0)
        unmet_demand = max(0, self.demand - net_energy)

        reward = -unmet_demand - 0.1 * abs(action)

        self.demand = np.random.uniform(5.0, self.max_demand)
        self.generation = np.random.uniform(10.0, self.max_generation)

        return self._get_obs(), reward, False, {}

    def render(self, mode='human'):
        print(f"Storage: {self.storage:.1f}, Demand: {self.demand:.1f}, Generation: {self.generation:.1f}")
