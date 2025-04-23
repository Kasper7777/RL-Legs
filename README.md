# Leg Robot Stair Climbing Project

This project implements a reinforcement learning environment for training a quadruped robot to climb stairs and reach a target using PyBullet and Stable Baselines3.

## Project Structure

- `leg_env.py`: The Gymnasium environment for the leg robot
- `leg_robot.urdf`: URDF file defining the quadruped robot
- `target.urdf`: URDF file defining the target object
- `train.py`: Training script using PPO algorithm
- `Dockerfile`: Container configuration for running the project
- `requirements.txt`: Python dependencies

## Requirements

- Docker
- NVIDIA GPU (recommended for faster training)

## Setup and Running

1. Build the Docker image:
```bash
docker build -t leg-robot .
```

2. Run the training:
```bash
docker run --gpus all -it leg-robot
```

The training will:
- Create a quadruped robot with 12 joints (3 per leg)
- Set up a staircase environment
- Place a target (yellow sphere) at the top of the stairs
- Train the robot using PPO to climb the stairs and reach the target

## Training Details

- The robot receives observations about its position, orientation, joint angles, and target position
- Actions control the 12 joints of the robot
- Rewards are given for:
  - Getting closer to the target
  - Reaching the target (+100 reward)
  - Penalties for falling (-100 reward)
- Training uses PPO with the following hyperparameters:
  - Learning rate: 3e-4
  - Batch size: 64
  - Number of epochs: 10
  - Gamma: 0.99
  - GAE Lambda: 0.95
  - Clip range: 0.2
  - Entropy coefficient: 0.01

## Monitoring Training

Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir logs
```

## Results

The trained model will be saved in the `models/` directory, including:
- Checkpoints during training
- The best model based on evaluation
- The final model
- Environment normalization parameters 