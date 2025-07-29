import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from hospital_env import HospitalShiftEnv
from data_loader import load_data

# Load data
states, actions = load_data("dataset .csv")

# Create environment
env = HospitalShiftEnv(states, actions)

# Optional: Validate environment
check_env(env, warn=True)

# Train model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

# Save model
model.save("doctor_shift_rl_model")
print("✅ Training complete and model saved!")
