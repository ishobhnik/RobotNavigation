import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import argparse

class OfflineRLDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.episodes = []
        self.load_episodes()
        
    def load_episodes(self):
        """Load all episodes from the data directory"""
        for episode_dir in sorted(os.listdir(self.data_dir)):
            if not episode_dir.startswith('episode_'):
                continue
                
            episode_path = os.path.join(self.data_dir, episode_dir)
            with open(os.path.join(episode_path, 'episode_data.pkl'), 'rb') as f:
                episode_data = pickle.load(f)
                
            # Convert lists to numpy arrays
            episode_data = {
                'context_vectors': np.array(episode_data['context_vectors']),
                'actions': np.array(episode_data['actions']),
                'predicted_noise': np.array(episode_data['predicted_noise']),
                'waypoints': np.array(episode_data['waypoints'])
            }
            
            self.episodes.append(episode_data)
            
    def __len__(self):
        """Return total number of transitions across all episodes"""
        return sum(len(ep['actions']) for ep in self.episodes)
    
    def __getitem__(self, idx):
        """Get a single transition"""
        # Find which episode and transition this index corresponds to
        episode_idx = 0
        transition_idx = idx
        while transition_idx >= len(self.episodes[episode_idx]['actions']):
            transition_idx -= len(self.episodes[episode_idx]['actions'])
            episode_idx += 1
            
        episode = self.episodes[episode_idx]
        
        # Get the transition data
        context_vector = episode['context_vectors'][transition_idx]
        action = episode['actions'][transition_idx]
        predicted_noise = episode['predicted_noise'][transition_idx]
        waypoint = episode['waypoints'][transition_idx]
        
        # Convert to tensors
        context_vector = torch.FloatTensor(context_vector)
        action = torch.FloatTensor(action)
        predicted_noise = torch.FloatTensor(predicted_noise)
        waypoint = torch.FloatTensor(waypoint)
        
        return {
            'context_vector': context_vector,
            'action': action,
            'predicted_noise': predicted_noise,
            'waypoint': waypoint
        }

def process_data(args: argparse.Namespace):
    """Process collected offline data and prepare it for RL training"""
    # Create dataset
    dataset = OfflineRLDataset(args.data_dir)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Save processed dataset
    output_path = os.path.join(args.output_dir, 'processed_dataset.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump({
            'dataset': dataset,
            'dataloader': dataloader
        }, f)
    
    print(f"Processed dataset saved to {output_path}")
    print(f"Total transitions: {len(dataset)}")
    print(f"Number of episodes: {len(dataset.episodes)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Directory containing collected offline data")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save processed dataset")
    parser.add_argument("--batch_size", type=int, default=256,
                      help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of workers for dataloader")
    args = parser.parse_args()
    
    process_data(args) 