import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from hospital_env import HospitalShiftEnv
from data_loader import load_data
import matplotlib.pyplot as plt

# Page title
st.title("🧠 Smart Doctor Shift Predictor (RL Model)")

# File upload
uploaded_file = st.file_uploader("Upload your dataset CSV", type=["csv"])

if uploaded_file is not None:
    st.success("✅ Dataset uploaded!")
    
    # Save and load model
    df = pd.read_csv(uploaded_file)
    df.to_csv("uploaded_dataset.csv", index=False)
    
    # Load and clean
    states, actions = load_data("uploaded_dataset.csv")
    env = HospitalShiftEnv(states, actions)
    model = PPO.load("doctor_shift_rl_model")

    # Predict
    obs, _ = env.reset()
    predicted = []
    actual = []
    rewards = []

    for step in range(len(states)):
        action, _ = model.predict(obs, deterministic=True)
        predicted_doctors = int(action + 3)
        actual_doctors = int(actions[step])
        obs, reward, _, _, _ = env.step(action)

        predicted.append(predicted_doctors)
        actual.append(actual_doctors)
        rewards.append(reward)

    # Display predictions
    st.subheader("📋 Predicted vs Actual Doctor Shifts")
    results_df = pd.DataFrame({
        "Hour": list(range(1, len(predicted)+1)),
        "Predicted Doctors": predicted,
        "Actual Doctors": actual,
        "Reward": rewards
    })
    st.dataframe(results_df)

    # Line plot
    st.subheader("📊 Reward & Shift Comparison")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(rewards, label="Reward", color='purple')
    ax[0].set_title("Reward Over Time")
    ax[0].set_xlabel("Hour")
    ax[0].grid(True)

    ax[1].plot(predicted, label="Predicted", color='blue')
    ax[1].plot(actual, label="Actual", color='green', alpha=0.6)
    ax[1].set_title("Doctor Shifts: Predicted vs Actual")
    ax[1].set_xlabel("Hour")
    ax[1].legend()
    ax[1].grid(True)

    st.pyplot(fig)
