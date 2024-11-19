import torch
import torch.optim as optim
import numpy as np
from model import PolicyNetwork
import math

class RobotArmEnv:
    def __init__(self):
        # Robot arm parameters
        self.l1 = 1.0  # Length of first link
        self.l2 = 1.0  # Length of second link
        self.theta1 = 0.0  # First joint angle
        self.theta2 = 0.0  # Second joint angle
        self.target_x = 1.5  # Target x position
        self.target_y = 1.0  # Target y position
        self.max_steps = 50  # Reduced max steps
        self.steps = 0
        
    def reset(self):
        # Smaller range for initial angles
        self.theta1 = np.random.uniform(-np.pi/4, np.pi/4)
        self.theta2 = np.random.uniform(-np.pi/4, np.pi/4)
        self.steps = 0
        return self._get_state()
    
    def _get_state(self):
        # Current end effector position
        x, y = self._forward_kinematics(self.theta1, self.theta2)
        # State: [end_effector_x, end_effector_y, target_x, target_y]
        return torch.FloatTensor([x, y, self.target_x, self.target_y])
    
    def _forward_kinematics(self, theta1, theta2):
        x1 = self.l1 * np.cos(theta1)
        y1 = self.l1 * np.sin(theta1)
        x2 = x1 + self.l2 * np.cos(theta1 + theta2)
        y2 = y1 + self.l2 * np.sin(theta1 + theta2)
        return x2, y2
    
    def step(self, action):
        # Update joint angles (action is delta theta)
        self.theta1 += action[0].item() * 0.1  # Scale action to smaller changes
        self.theta2 += action[1].item() * 0.1
        
        # Clip angles
        self.theta1 = np.clip(self.theta1, -np.pi, np.pi)
        self.theta2 = np.clip(self.theta2, -np.pi, np.pi)
        
        # Get current end effector position
        x, y = self._forward_kinematics(self.theta1, self.theta2)
        
        # Calculate distance to target
        distance = np.sqrt((x - self.target_x)**2 + (y - self.target_y)**2)
        
        # Calculate reward
        reward = -distance  # Negative distance as reward
        
        self.steps += 1
        done = (distance < 0.1) or (self.steps >= self.max_steps)
        
        if distance < 0.1:  # Bonus for reaching target
            reward += 10.0
        
        return self._get_state(), reward, done

def train():
    # Modified hyperparameters
    learning_rate = 0.0003  # Reduced learning rate
    episodes = 2000
    gamma = 0.99
    
    env = RobotArmEnv()
    model = PolicyNetwork()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add gradient clipping
    max_grad_norm = 0.5
    
    best_reward = float('-inf')
    
    for episode in range(episodes):
        state = env.reset()
        episode_rewards = []
        episode_log_probs = []
        
        done = False
        while not done:
            # Get action means from policy network
            action_means = model(state)
            
            # Reduced exploration noise
            action_std = torch.ones_like(action_means) * 0.05
            
            # Add try-except to catch any numerical instability
            try:
                action_distribution = torch.distributions.Normal(action_means, action_std)
                action = action_distribution.sample()
            except ValueError:
                print(f"Warning: Invalid values detected at episode {episode}")
                break
                
            # Take action in environment
            next_state, reward, done = env.step(action)
            
            # Store reward and log probability
            episode_rewards.append(reward)
            episode_log_probs.append(action_distribution.log_prob(action).sum())
            
            state = next_state
        
        if len(episode_rewards) == 0:
            continue
            
        # Calculate returns and loss
        returns = []
        R = 0
        for r in episode_rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns more safely
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate loss
        loss = []
        for log_prob, R in zip(episode_log_probs, returns):
            loss.append(-log_prob * R)
        loss = torch.stack(loss).sum()
        
        # Update model with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        # Track best model
        episode_total_reward = sum(episode_rewards)
        if episode_total_reward > best_reward:
            best_reward = episode_total_reward
            torch.save(model.state_dict(), "best_robot_model.pth")
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {loss.item():.4f}, Total Reward: {episode_total_reward:.4f}")

if __name__ == "__main__":
    train() 