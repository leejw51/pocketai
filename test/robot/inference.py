import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PolicyNetwork
from train import RobotArmEnv

def visualize_robot_arm(env, target_x, target_y):
    # Plot robot arm links
    x1 = env.l1 * np.cos(env.theta1)
    y1 = env.l1 * np.sin(env.theta1)
    x2, y2 = env._forward_kinematics(env.theta1, env.theta2)
    
    plt.clf()
    plt.plot([0, x1], [0, y1], 'b-', linewidth=2, label='Link 1')
    plt.plot([x1, x2], [y1, y2], 'g-', linewidth=2, label='Link 2')
    plt.plot(target_x, target_y, 'r*', markersize=15, label='Target')
    plt.plot(x2, y2, 'ko', markersize=10, label='End Effector')
    plt.grid(True)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.legend()
    plt.pause(0.1)

def evaluate(model_path="best_robot_model.pth"):
    # Load model
    model = PolicyNetwork()
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        return
    
    model.eval()
    
    # Create environment
    env = RobotArmEnv()
    state = env.reset()
    
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(8, 8))
    
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        # Visualize current state
        visualize_robot_arm(env, env.target_x, env.target_y)
        
        # Get action
        with torch.no_grad():
            action = model(state)
        
        # Take step in environment
        state, reward, done = env.step(action)
        total_reward += reward
        step_count += 1
        
        print(f"Step {step_count}: End effector position: ({state[0]:.2f}, {state[1]:.2f}), Reward: {reward:.2f}")
    
    print(f"\nEvaluation completed:")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    evaluate() 