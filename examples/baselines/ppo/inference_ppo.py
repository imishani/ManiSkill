#!/usr/bin/env python3
"""
PPO Inference Script for ManiSkill Humanoid Tasks

This script loads a pretrained PPO model and runs inference on the specified environment.
It supports both UnitreeG1PlaceAppleInBowl-v1 and UnitreeG1PushBowlToRegion-v1 tasks.

Usage:
    python inference_ppo.py --checkpoint path/to/ckpt_X.pt --env_id UnitreeG1PushBowlToRegion-v1 --num_episodes 10
"""

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

import mani_skill.envs


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO Agent architecture matching the training script"""
    
    def __init__(self, envs, hidden_dim=512):
        super().__init__()
        
        # Get observation and action space dimensions
        if hasattr(envs.single_observation_space, 'shape'):
            obs_shape = envs.single_observation_space.shape
        else:
            # Handle dict observation spaces
            obs_shape = None
            for key, space in envs.single_observation_space.spaces.items():
                if obs_shape is None:
                    obs_shape = space.shape
                else:
                    obs_shape = (obs_shape[0] + space.shape[0],)
        
        # Network architecture
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        if isinstance(x, dict):
            # Flatten dict observations
            x = torch.cat([v.flatten(1) for v in x.values()], dim=1)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if isinstance(x, dict):
            # Flatten dict observations
            x = torch.cat([v.flatten(1) for v in x.values()], dim=1)
            
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file in the directory"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Find all checkpoint files
    ckpt_files = list(checkpoint_path.glob("ckpt_*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Sort by step number and return the latest
    ckpt_files.sort(key=lambda x: int(x.stem.split('_')[1]))
    return str(ckpt_files[-1])


def load_model(checkpoint_path, envs, device):
    """Load the pretrained PPO model"""
    agent = Agent(envs).to(device)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # The checkpoint contains the full training state
    if 'model_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If it's just the model state dict
        agent.load_state_dict(checkpoint)
    
    agent.eval()
    print("Model loaded successfully!")
    return agent


def run_inference(agent, envs, num_episodes, device, render=False, save_video=False):
    """Run inference with the loaded model"""
    
    print(f"Running inference for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    
    for episode in range(num_episodes):
        obs, info = envs.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done:
            # Convert observation to tensor
            if isinstance(obs, dict):
                obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
            else:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Get action from the model
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy()[0]  # Remove batch dimension
            
            # Step the environment
            obs, reward, terminated, truncated, info = envs.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                envs.render()
                time.sleep(0.02)  # Small delay for visualization
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check if episode was successful
        success = info.get('success', False) if isinstance(info, dict) else False
        success_rates.append(success)
        
        print(f"  Reward: {episode_reward:.3f}, Length: {episode_length}, Success: {success}")
    
    # Print summary statistics
    print(f"\n=== Inference Summary ===")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success Rate: {np.mean(success_rates):.3f} ({sum(success_rates)}/{num_episodes})")


def main():
    parser = argparse.ArgumentParser(description="PPO Inference for ManiSkill Humanoid Tasks")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to checkpoint file or directory")
    parser.add_argument("--env_id", type=str, default="UnitreeG1PushBowlToRegion-v1",
                       choices=["UnitreeG1PlaceAppleInBowl-v1", "UnitreeG1PushBowlToRegion-v1"],
                       help="Environment to run inference on")
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of episodes to run")
    parser.add_argument("--render", action="store_true",
                       help="Render the environment")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run on (cpu/cuda/auto)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create environment
    print(f"Creating environment: {args.env_id}")
    envs = gym.make(args.env_id, num_envs=1, obs_mode="state", render_mode="human" if args.render else None)
    envs.reset(seed=args.seed)
    
    # Load checkpoint
    if os.path.isdir(args.checkpoint):
        checkpoint_path = find_latest_checkpoint(args.checkpoint)
    else:
        checkpoint_path = args.checkpoint
    
    agent = load_model(checkpoint_path, envs, device)
    
    # Run inference
    run_inference(agent, envs, args.num_episodes, device, args.render)
    
    envs.close()


if __name__ == "__main__":
    main()