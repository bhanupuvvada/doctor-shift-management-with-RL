import numpy as np
from stable_baselines3 import PPO
from hospital_env import HospitalShiftEnv
from data_loader import load_data

# Load trained model
model = PPO.load("doctor_shift_rl_model")

# Load hospital data
states, actions = load_data("dataset.csv")

# Create environment
env = HospitalShiftEnv(states, actions)

# Reset environment
obs, _ = env.reset()

total_reward = 0
step = 0
print("📊 Doctor Shift Prediction Log\n")

# Loop through dataset steps manually
for step in range(len(states)):
    action, _ = model.predict(obs, deterministic=True)
    predicted_doctors = int(action + 3)
    actual_doctors = int(actions[step])

    obs, reward, _, _, _ = env.step(action)
    total_reward += reward

    print(f"Hour {step+1:>3}: Predicted = {predicted_doctors} | Actual = {actual_doctors} | Reward = {reward}")

print(f"\n Finished evaluation — Total Reward: {total_reward}")
