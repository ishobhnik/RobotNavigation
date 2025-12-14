import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import wandb
from tqdm import tqdm
import yaml
from collections import deque
import random

from vint_train.models.sac.sac import SACAgent
from process_offline_data import OfflineRLDataset

class ReplayBuffer:
    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return {
            'context_vector': torch.FloatTensor(np.array(state)),
            'action': torch.FloatTensor(np.array(action)),
            'reward': torch.FloatTensor(np.array(reward)).unsqueeze(1),
            'next_context_vector': torch.FloatTensor(np.array(next_state)),
            'done': torch.FloatTensor(np.array(done)).unsqueeze(1)
        }
    
    def __len__(self):
        return len(self.buffer)

def compute_reward(waypoint: np.ndarray, next_waypoint: np.ndarray, goal_waypoint: np.ndarray) -> float:
    """Compute reward based on progress towards goal"""
    current_dist = np.linalg.norm(waypoint[:2] - goal_waypoint[:2])
    next_dist = np.linalg.norm(next_waypoint[:2] - goal_waypoint[:2])
    
    # Reward for getting closer to goal
    progress_reward = current_dist - next_dist
    
    # Penalty for large actions
    action_penalty = -0.1 * np.linalg.norm(next_waypoint[:2] - waypoint[:2])
    
    # Bonus for reaching goal
    goal_bonus = 10.0 if next_dist < 0.1 else 0.0
    
    return progress_reward + action_penalty + goal_bonus

def prepare_batch(batch: dict, goal_waypoint: np.ndarray) -> dict:
    """Prepare batch for training by computing rewards and next states"""
    context_vectors = batch['context_vector']
    actions = batch['action']
    waypoints = batch['waypoint']
    
    # Compute rewards
    rewards = []
    for i in range(len(waypoints) - 1):
        reward = compute_reward(waypoints[i], waypoints[i + 1], goal_waypoint)
        rewards.append(reward)
    rewards.append(0.0)  # Terminal state reward
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    
    # Prepare next states
    next_context_vectors = torch.cat([context_vectors[1:], context_vectors[-1:]], dim=0)
    
    # Prepare done flags
    dones = torch.zeros(len(waypoints), 1)
    dones[-1] = 1.0
    
    return {
        'context_vector': context_vectors,
        'action': actions,
        'reward': rewards,
        'next_context_vector': next_context_vectors,
        'done': dones
    }

def train_sac(args: argparse.Namespace):
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project="robot-navigation-sac", config=vars(args))
    
    # Load dataset
    with open(args.dataset_path, 'rb') as f:
        data = torch.load(f)
    dataset = data['dataset']
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Initialize SAC agent
    state_dim = dataset.episodes[0]['context_vectors'].shape[-1]
    action_dim = dataset.episodes[0]['actions'].shape[-1]
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        device=args.device
    )
    
    # Initialize replay buffer for online fine-tuning
    replay_buffer = ReplayBuffer(max_size=args.replay_size)
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_losses = []
        
        # Offline training phase
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs} - Offline"):
            # Get goal waypoint from the last waypoint in the episode
            goal_waypoint = batch['waypoint'][-1].numpy()
            
            # Prepare batch
            train_batch = prepare_batch(batch, goal_waypoint)
            
            # Update agent
            losses = agent.update_parameters(train_batch)
            epoch_losses.append(losses)
            
            # Log to wandb
            if args.use_wandb:
                wandb.log({f"offline_{k}": v for k, v in losses.items()})
        
        # Online fine-tuning phase
        if args.online_finetune:
            for _ in tqdm(range(args.online_steps), desc=f"Epoch {epoch + 1}/{args.epochs} - Online"):
                if len(replay_buffer) < args.batch_size:
                    continue
                    
                # Sample from replay buffer
                batch = replay_buffer.sample(args.batch_size)
                
                # Update agent
                losses = agent.update_parameters(batch)
                
                # Log to wandb
                if args.use_wandb:
                    wandb.log({f"online_{k}": v for k, v in losses.items()})
        
        # Compute average losses
        avg_losses = {
            k: np.mean([l[k] for l in epoch_losses])
            for k in epoch_losses[0].keys()
        }
        
        print(f"Epoch {epoch + 1}/{args.epochs}")
        for k, v in avg_losses.items():
            print(f"{k}: {v:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
    return agent, replay_buffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                      help="Path to processed dataset")
    parser.add_argument("--batch_size", type=int, default=256,
                      help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of workers for dataloader")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                      help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                      help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005,
                      help="Target network update rate")
    parser.add_argument("--alpha", type=float, default=0.2,
                      help="Entropy regularization coefficient")
    parser.add_argument("--hidden_dim", type=int, default=256,
                      help="Hidden dimension for networks")
    
    # Online fine-tuning arguments
    parser.add_argument("--online_finetune", action="store_true",
                      help="Whether to perform online fine-tuning")
    parser.add_argument("--online_steps", type=int, default=1000,
                      help="Number of online fine-tuning steps per epoch")
    parser.add_argument("--replay_size", type=int, default=100000,
                      help="Size of replay buffer for online fine-tuning")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save checkpoints")
    parser.add_argument("--save_freq", type=int, default=10,
                      help="Frequency of saving checkpoints")
    
    # Misc arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to train on")
    parser.add_argument("--use_wandb", action="store_true",
                      help="Whether to use wandb for logging")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_sac(args) 