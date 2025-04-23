import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from leg_env import LegEnv

# Create logs directory
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Create the environment
env = LegEnv(render_mode="human")
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Create callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="models/",
    name_prefix="leg_robot"
)

eval_callback = EvalCallback(
    env,
    best_model_save_path="models/",
    log_path="logs/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# Create the model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)

# Train the model
print("Starting training...")
start_time = time.time()
model.learn(
    total_timesteps=1000000,
    callback=[checkpoint_callback, eval_callback]
)
print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Save the final model
model.save("models/final_model")
env.save("models/vec_normalize.pkl")

# Close the environment
env.close() 