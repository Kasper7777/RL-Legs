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
            
        # Check if connection is successful
        if self.client < 0:
            raise RuntimeError("Failed to connect to PyBullet physics server")
            
        # Set physics engine parameters for better stability
        p.setPhysicsEngineParameter(
            numSolverIterations=50,  # Increased from default for better stability
            numSubSteps=4,           # More substeps for smoother simulation
            erp=0.2,                 # Error reduction parameter
            contactERP=0.2,          # Contact error reduction
            frictionERP=0.2          # Friction error reduction
        )
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load robot to get number of joints
        self.robot = p.loadURDF("leg_robot.urdf", [0, 0, 0.5])
        self.num_joints = p.getNumJoints(self.robot)
        p.removeBody(self.robot)  # Remove the temporary robot
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
        
        # Observation space: [robot position(3), robot orientation(4), joint angles(n), joint velocities(n), target position(3)]
        obs_size = 3 + 4 + self.num_joints + self.num_joints + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Reset robot and other objects
        self.robot = None
        self.target = None
        self.stairs = []
        
        # Simulation parameters
        self.time_step = 1.0/240.0
        self.max_steps = 1000
        self.current_step = 0
        
        # Physics parameters
        self.max_force = 15.0  # Increased from 10.0 for better joint control
        self.max_velocity = 0.5  # Increased from 0.3 for faster response
        self.joint_damping = 0.5  # Keep moderate damping
        self.target_height = 0.5  # Target height for the robot
        self.stability_threshold = 0.2  # Maximum allowed tilt in radians
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation
        p.resetSimulation()
        
        # Ensure gravity is properly set
        p.setGravity(0, 0, -9.81)
        
        # Load plane
        p.loadURDF("plane.urdf")
        
        # Load stairs
        self._create_stairs()
        
        # Load robot
        self.robot = p.loadURDF("leg_robot.urdf", [0, 0, self.target_height])
        
        # Disable collisions between connected links
        self._setup_collision_filtering()
        
        # Configure robot joints
        for i in range(self.num_joints):
            # Set joint damping and friction
            p.changeDynamics(
                self.robot, 
                i, 
                linearDamping=0.3,  # Increased from 0.2
                angularDamping=0.3,  # Increased from 0.2
                jointDamping=self.joint_damping,
                lateralFriction=2.0,  # Increased from 1.5
                spinningFriction=1.5,  # Increased from 1.0
                rollingFriction=1.0    # Increased from 0.5
            )
            
            # Configure joint motors - use only position control
            info = p.getJointInfo(self.robot, i)
            if info[3] > -1:  # If joint is not fixed
                p.setJointMotorControl2(
                    self.robot,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=0,
                    force=self.max_force,
                    maxVelocity=self.max_velocity,
                    positionGain=0.2,  # Increased from 0.1 for stiffer control
                    velocityGain=0.1   # Keep moderate velocity damping
                )
        
        # Load target (food)
        self.target = p.loadURDF("target.urdf", [5, 0, self.target_height])
        
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
            stair = p.loadURDF("cube.urdf", 
                             basePosition=stair_pos,
                             baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                             globalScaling=stair_length,
                             useFixedBase=True)  # Make stairs static
            self.stairs.append(stair)
    
    def _apply_action(self, action):
        # Apply action to robot joints with position control
        for i in range(self.num_joints):
            # Scale action to very small joint angles (in radians)
            target_angle = action[i] * 0.1  # Keep moderate scaling
            
            # Get current joint state
            current_state = p.getJointState(self.robot, i)
            current_angle = current_state[0]
            current_velocity = current_state[1]
            
            # Smoother low-pass filter with stronger smoothing
            smoothed_angle = 0.95 * current_angle + 0.05 * target_angle
            
            p.setJointMotorControl2(
                self.robot,
                i,
                p.POSITION_CONTROL,
                targetPosition=smoothed_angle,
                force=self.max_force,
                maxVelocity=self.max_velocity,
                positionGain=0.2,  # Increased from 0.1 for stiffer control
                velocityGain=0.1   # Keep moderate velocity damping
            )
    
    def _get_observation(self):
        # Get robot position and orientation
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot)
        
        # Get joint angles and velocities
        joint_states = p.getJointStates(self.robot, range(self.num_joints))
        joint_angles = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        # Get target position
        target_pos, _ = p.getBasePositionAndOrientation(self.target)
        
        # Combine all observations
        observation = np.concatenate([
            robot_pos,          # 3 values
            robot_orn,          # 4 values
            joint_angles,       # num_joints values
            joint_velocities,   # num_joints values
            target_pos          # 3 values
        ])
        
        return observation
    
    def _calculate_reward(self):
        # Get positions and velocities
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot)
        target_pos, _ = p.getBasePositionAndOrientation(self.target)
        
        # Calculate distance to target
        distance = np.linalg.norm(np.array(robot_pos) - np.array(target_pos))
        
        # Get robot velocity and angular velocity
        robot_vel, robot_ang_vel = p.getBaseVelocity(self.robot)
        forward_velocity = robot_vel[0]  # x-component of velocity
        
        # Calculate stability metrics
        euler_angles = p.getEulerFromQuaternion(robot_orn)
        roll, pitch = euler_angles[0], euler_angles[1]
        stability = 1.0 - (abs(roll) + abs(pitch)) / (2 * self.stability_threshold)
        stability = max(0.0, min(1.0, stability))  # Clamp between 0 and 1
        
        # Calculate smoothness metrics
        joint_states = p.getJointStates(self.robot, range(self.num_joints))
        joint_velocities = np.array([state[1] for state in joint_states])
        joint_accelerations = np.array([state[2] for state in joint_states])
        smoothness = 1.0 - (np.mean(np.abs(joint_velocities)) + np.mean(np.abs(joint_accelerations))) / 2.0
        smoothness = max(0.0, min(1.0, smoothness))
        
        # Calculate reward components
        distance_reward = -distance  # Negative reward for distance
        velocity_reward = forward_velocity * stability  # Velocity reward weighted by stability
        stability_reward = stability * 2.0  # Increased stability reward
        smoothness_reward = smoothness  # Reward for smooth movement
        
        # Height penalty to keep robot close to ground
        height_error = abs(robot_pos[2] - self.target_height)
        height_penalty = -height_error * 3.0  # Increased penalty
        
        # Angular velocity penalty to prevent spinning
        angular_velocity_penalty = -np.linalg.norm(robot_ang_vel) * 1.0
        
        # Combine rewards with weights
        reward = (
            0.1 * distance_reward +
            0.2 * velocity_reward +
            0.3 * stability_reward +
            0.2 * smoothness_reward +
            0.1 * height_penalty +
            0.1 * angular_velocity_penalty
        )
        
        # Add bonus for reaching target
        if distance < 0.5:
            reward += 100
            
        # Penalize falling or flying
        if robot_pos[2] < 0.1 or robot_pos[2] > self.target_height + 0.5:
            reward -= 100
            
        # Penalize excessive tilting
        if abs(roll) > self.stability_threshold or abs(pitch) > self.stability_threshold:
            reward -= 50
            
        return reward
    
    def _check_done(self):
        # Check if episode is done
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot)
        target_pos, _ = p.getBasePositionAndOrientation(self.target)
        euler_angles = p.getEulerFromQuaternion(robot_orn)
        roll, pitch = euler_angles[0], euler_angles[1]
        
        # Episode ends if:
        # 1. Robot falls
        # 2. Robot flies too high
        # 3. Robot reaches target
        # 4. Maximum steps reached
        # 5. Robot tilts too much
        return (robot_pos[2] < 0.1 or 
                robot_pos[2] > self.target_height + 0.5 or
                np.linalg.norm(np.array(robot_pos) - np.array(target_pos)) < 0.5 or
                self.current_step >= self.max_steps or
                abs(roll) > self.stability_threshold or
                abs(pitch) > self.stability_threshold)
    
    def close(self):
        if p.isConnected():
            p.disconnect(self.client)
            self.client = -1

    def _setup_collision_filtering(self):
        # Get all joint information
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot, i)
            parent_idx = joint_info[16]  # Parent link index
            child_idx = joint_info[0]    # Child link index
            
            # Disable collision between parent and child links
            if parent_idx >= 0:  # Skip if parent is base (-1)
                p.setCollisionFilterPair(self.robot, self.robot, parent_idx, child_idx, 0) 