import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from hospital_env import HospitalShiftEnv
from data_loader import load_data

# Load model and data
model = PPO.load("doctor_shift_rl_model")
states, actions = load_data("dataset.csv")
env = HospitalShiftEnv(states, actions)

obs, _ = env.reset()
rewards = []
predicted = []
actual = []

for step in range(len(states)):
    action, _ = model.predict(obs, deterministic=True)
    predicted_doctors = int(action + 3)
    actual_doctors = int(actions[step])
    obs, reward, _, _, _ = env.step(action)

    predicted.append(predicted_doctors)
    actual.append(actual_doctors)
    rewards.append(reward)

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards, label='Reward')
plt.xlabel('Hour')
plt.ylabel('Reward')
plt.title('Reward Over Time')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(predicted, label='Predicted Doctors', color='blue')
plt.plot(actual, label='Actual Doctors', color='green', alpha=0.6)
plt.xlabel('Hour')
plt.ylabel('Doctor Count')
plt.title('Doctor Scheduling: Predicted vs Actual')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("reward_graph.png")   # ✅ Save as image
print("✅ Graph saved as reward_graph.png")

