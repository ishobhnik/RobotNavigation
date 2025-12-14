import os
import argparse
import torch
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Bool
import time
from tqdm import tqdm
import wandb

from vint_train.models.sac.sac import SACAgent
from utils import msg_to_pil, transform_images, load_model
from topic_names import IMAGE_TOPIC, WAYPOINT_TOPIC, GOAL_TOPIC

class SACEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = args.device
        
        # Load model parameters
        with open(args.model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)
            
        # Initialize vision encoder
        self.vision_encoder = load_model(
            args.vision_encoder_path,
            self.model_params,
            self.device
        ).to(self.device)
        self.vision_encoder.eval()
        
        # Initialize SAC agent
        state_dim = self.model_params["obs_encoding_size"]
        action_dim = 2  # x, y waypoint coordinates
        
        self.agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            device=self.device
        )
        self.agent.load(args.checkpoint_path)
        
        # Initialize ROS
        rospy.init_node("SAC_EVALUATOR", anonymous=False)
        self.rate = rospy.Rate(self.model_params["frame_rate"])
        
        # Subscribers
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback)
        self.goal_sub = rospy.Subscriber(GOAL_TOPIC, Bool, self.goal_callback)
        
        # Publishers
        self.waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
        
        # Initialize data collection
        self.context_queue = []
        self.context_size = self.model_params["context_size"]
        self.reached_goal = False
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def image_callback(self, msg):
        """Process incoming image and update context queue"""
        obs_img = msg_to_pil(msg)
        
        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(obs_img)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(obs_img)
            
    def goal_callback(self, msg):
        """Handle goal reached signal"""
        if msg.data:  # Goal reached
            self.reached_goal = True
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Log episode statistics
            if self.args.use_wandb:
                wandb.log({
                    "episode_reward": self.episode_rewards[-1],
                    "episode_length": self.episode_lengths[-1],
                    "mean_reward": np.mean(self.episode_rewards),
                    "mean_length": np.mean(self.episode_lengths)
                })
            
    def get_state_representation(self):
        """Get state representation from vision encoder"""
        if len(self.context_queue) <= self.context_size:
            return None
            
        obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)
        obs_images = obs_images.to(self.device)
        
        with torch.no_grad():
            # Get context vector from vision encoder
            context_vector = self.vision_encoder('vision_encoder', obs_img=obs_images)
            return context_vector.cpu().numpy()
        
    def evaluate(self, num_episodes: int):
        """Run evaluation episodes"""
        print(f"Starting evaluation for {num_episodes} episodes...")
        
        episode = 0
        while episode < num_episodes and not rospy.is_shutdown():
            if len(self.context_queue) <= self.context_size:
                self.rate.sleep()
                continue
                
            # Get state representation
            state = self.get_state_representation()
            if state is None:
                continue
                
            # Select action
            action = self.agent.select_action(state, evaluate=True)
            
            # Publish waypoint
            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = action
            self.waypoint_pub.publish(waypoint_msg)
            
            # Update episode tracking
            self.current_episode_length += 1
            
            # Check if episode ended
            if self.reached_goal:
                episode += 1
                self.reached_goal = False
                print(f"Episode {episode}/{num_episodes} completed")
                print(f"Reward: {self.episode_rewards[-1]:.2f}")
                print(f"Length: {self.episode_lengths[-1]}")
                
            self.rate.sleep()
            
        # Print final statistics
        print("\nEvaluation completed!")
        print(f"Mean reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
        print(f"Mean episode length: {np.mean(self.episode_lengths):.2f} ± {np.std(self.episode_lengths):.2f}")

def main(args: argparse.Namespace):
    evaluator = SACEvaluator(args)
    evaluator.evaluate(args.num_episodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_config_path", type=str, required=True,
                      help="Path to model config file")
    parser.add_argument("--vision_encoder_path", type=str, required=True,
                      help="Path to vision encoder checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                      help="Path to SAC agent checkpoint")
    
    # Evaluation arguments
    parser.add_argument("--num_episodes", type=int, default=10,
                      help="Number of evaluation episodes")
    parser.add_argument("--hidden_dim", type=int, default=256,
                      help="Hidden dimension for networks")
    
    # Misc arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run evaluation on")
    parser.add_argument("--use_wandb", action="store_true",
                      help="Whether to use wandb for logging")
    
    args = parser.parse_args()
    
    main(args) 