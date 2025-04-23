import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

class LegEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(LegEnv, self).__init__()
        
        # Initialize PyBullet
        if render_mode == "human":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        
        # Observation space: [robot position, robot orientation, joint angles, target position]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        
        # Load the leg robot URDF
        self.robot = None
        self.target = None
        self.stairs = []
        
        # Simulation parameters
        self.time_step = 1.0/240.0
        self.max_steps = 1000
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation
        p.resetSimulation()
        
        # Load plane
        p.loadURDF("plane.urdf")
        
        # Load stairs
        self._create_stairs()
        
        # Load robot
        self.robot = p.loadURDF("leg_robot.urdf", [0, 0, 0.5])
        
        # Load target (food)
        self.target = p.loadURDF("target.urdf", [5, 0, 0.5])
        
        # Reset simulation parameters
        self.current_step = 0
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action):
        # Apply action to robot joints
        self._apply_action(action)
        
        # Step simulation
        p.stepSimulation()
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._check_done()
        
        self.current_step += 1
        
        return observation, reward, done, False, {}
    
    def _create_stairs(self):
        # Create a set of stairs
        stair_width = 0.5
        stair_height = 0.1
        stair_length = 1.0
        
        for i in range(5):
            stair_pos = [i * stair_length, 0, i * stair_height / 2]
            stair = p.loadURDF("cube.urdf", stair_pos, 
                             p.getQuaternionFromEuler([0, 0, 0]),
                             globalScaling=[stair_length, stair_width, stair_height])
            self.stairs.append(stair)
    
    def _apply_action(self, action):
        # Apply action to robot joints
        for i in range(12):  # Assuming 12 joints
            p.setJointMotorControl2(
                self.robot,
                i,
                p.POSITION_CONTROL,
                targetPosition=action[i],
                force=100
            )
    
    def _get_observation(self):
        # Get robot position and orientation
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot)
        
        # Get joint angles
        joint_states = p.getJointStates(self.robot, range(12))
        joint_angles = [state[0] for state in joint_states]
        
        # Get target position
        target_pos, _ = p.getBasePositionAndOrientation(self.target)
        
        # Combine all observations
        observation = np.concatenate([
            robot_pos,
            robot_orn,
            joint_angles,
            target_pos
        ])
        
        return observation
    
    def _calculate_reward(self):
        # Get positions
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        target_pos, _ = p.getBasePositionAndOrientation(self.target)
        
        # Calculate distance to target
        distance = np.linalg.norm(np.array(robot_pos) - np.array(target_pos))
        
        # Base reward is negative distance (closer is better)
        reward = -distance
        
        # Add bonus for reaching target
        if distance < 0.5:
            reward += 100
            
        # Penalize falling
        if robot_pos[2] < 0.1:
            reward -= 100
            
        return reward
    
    def _check_done(self):
        # Check if episode is done
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        target_pos, _ = p.getBasePositionAndOrientation(self.target)
        
        # Episode ends if:
        # 1. Robot falls
        # 2. Robot reaches target
        # 3. Maximum steps reached
        return (robot_pos[2] < 0.1 or 
                np.linalg.norm(np.array(robot_pos) - np.array(target_pos)) < 0.5 or
                self.current_step >= self.max_steps)
    
    def close(self):
        p.disconnect(self.client) 