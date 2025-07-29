import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HospitalShiftEnv(gym.Env):  # ✅ This is valid because we imported gymnasium as gym
    def __init__(self, states, actions):
        super(HospitalShiftEnv, self).__init__()
        self.states = states
        self.actions_data = actions
        self.current_step = 0

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.states.shape[1],),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.current_step = 0
        return self.states[self.current_step].astype(np.float32), {}

    def step(self, action):
        doctor_chosen = action + 3
        doctor_actual = self.actions_data[self.current_step]

        if doctor_chosen == doctor_actual:
            reward = 10
        elif abs(doctor_chosen - doctor_actual) == 1:
            reward = 5
        else:
            reward = -5

        self.current_step += 1
        done = self.current_step >= len(self.states)

        if not done:
            next_state = self.states[self.current_step].astype(np.float32)
        else:
            next_state = np.zeros_like(self.states[0], dtype=np.float32)

        return next_state, reward, done, False, {}
