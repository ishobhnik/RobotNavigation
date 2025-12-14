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
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import matplotlib.pyplot as plt
from collections import defaultdict

from vint_train.models.sac.sac import SACAgent
from utils import msg_to_pil, transform_images, load_model, get_action
from topic_names import IMAGE_TOPIC, WAYPOINT_TOPIC, GOAL_TOPIC, SAMPLED_ACTIONS_TOPIC

class PolicyComparator:
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
        
        # Initialize diffusion policy
        self.diffusion_model = load_model(
            args.diffusion_model_path,
            self.model_params,
            self.device
        ).to(self.device)
        self.diffusion_model.eval()
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
        # Initialize SAC agent
        state_dim = self.model_params["obs_encoding_size"]
        action_dim = 2  # x, y waypoint coordinates
        
        self.sac_agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            device=self.device
        )
        self.sac_agent.load(args.sac_checkpoint_path)
        
        # Initialize ROS
        rospy.init_node("POLICY_COMPARATOR", anonymous=False)
        self.rate = rospy.Rate(self.model_params["frame_rate"])
        
        # Subscribers
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback)
        self.goal_sub = rospy.Subscriber(GOAL_TOPIC, Bool, self.goal_callback)
        
        # Publishers
        self.waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
        self.sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
        
        # Initialize data collection
        self.context_queue = []
        self.context_size = self.model_params["context_size"]
        self.reached_goal = False
        
        # Initialize metrics
        self.metrics = {
            'diffusion': defaultdict(list),
            'sac': defaultdict(list)
        }
        self.current_episode = {
            'diffusion': defaultdict(list),
            'sac': defaultdict(list)
        }
        
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
            
            # Save episode metrics
            for policy in ['diffusion', 'sac']:
                for metric, values in self.current_episode[policy].items():
                    self.metrics[policy][metric].append(np.mean(values))
            
            # Reset episode tracking
            self.current_episode = {
                'diffusion': defaultdict(list),
                'sac': defaultdict(list)
            }
            
            # Log episode statistics
            if self.args.use_wandb:
                for policy in ['diffusion', 'sac']:
                    wandb.log({
                        f"{policy}_episode_reward": self.metrics[policy]['reward'][-1],
                        f"{policy}_episode_length": self.metrics[policy]['length'][-1],
                        f"{policy}_mean_planning_time": self.metrics[policy]['planning_time'][-1],
                        f"{policy}_mean_action_smoothness": self.metrics[policy]['action_smoothness'][-1]
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
    
    def get_diffusion_action(self, state):
        """Get action from diffusion policy"""
        start_time = time.time()
        
        with torch.no_grad():
            # Initialize action from Gaussian noise
            noisy_action = torch.randn(
                (self.args.num_samples, self.model_params["len_traj_pred"], 2), 
                device=self.device
            )
            naction = noisy_action
            
            # Run diffusion steps
            self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.diffusion_model(
                    'noise_pred_net',
                    sample=naction,
                    timestep=k,
                    global_cond=state
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
            
            # Get final action
            naction = get_action(naction)
            action = naction[0].cpu().numpy()
            
        planning_time = time.time() - start_time
        return action, planning_time
    
    def get_sac_action(self, state):
        """Get action from SAC policy"""
        start_time = time.time()
        action = self.sac_agent.select_action(state, evaluate=True)
        planning_time = time.time() - start_time
        return action, planning_time
    
    def compute_action_smoothness(self, action, prev_action):
        """Compute smoothness of action sequence"""
        if prev_action is None:
            return 0.0
        return -np.linalg.norm(action - prev_action)
    
    def compare_policies(self, num_episodes: int):
        """Run comparison episodes"""
        print(f"Starting policy comparison for {num_episodes} episodes...")
        
        episode = 0
        prev_actions = {'diffusion': None, 'sac': None}
        
        while episode < num_episodes and not rospy.is_shutdown():
            if len(self.context_queue) <= self.context_size:
                self.rate.sleep()
                continue
                
            # Get state representation
            state = self.get_state_representation()
            if state is None:
                continue
            
            # Run both policies
            for policy in ['diffusion', 'sac']:
                # Get action
                if policy == 'diffusion':
                    action, planning_time = self.get_diffusion_action(state)
                else:
                    action, planning_time = self.get_sac_action(state)
                
                # Compute metrics
                action_smoothness = self.compute_action_smoothness(action, prev_actions[policy])
                prev_actions[policy] = action
                
                # Update current episode metrics
                self.current_episode[policy]['planning_time'].append(planning_time)
                self.current_episode[policy]['action_smoothness'].append(action_smoothness)
                self.current_episode[policy]['length'].append(1)
                
                # Publish waypoint
                waypoint_msg = Float32MultiArray()
                waypoint_msg.data = action
                self.waypoint_pub.publish(waypoint_msg)
                
                if policy == 'diffusion':
                    # Publish sampled actions for visualization
                    sampled_actions_msg = Float32MultiArray()
                    sampled_actions_msg.data = action.flatten()
                    self.sampled_actions_pub.publish(sampled_actions_msg)
            
            # Check if episode ended
            if self.reached_goal:
                episode += 1
                self.reached_goal = False
                print(f"Episode {episode}/{num_episodes} completed")
                
                # Print episode statistics
                for policy in ['diffusion', 'sac']:
                    print(f"\n{policy.upper()} Policy:")
                    print(f"Planning time: {np.mean(self.current_episode[policy]['planning_time']):.4f}s")
                    print(f"Action smoothness: {np.mean(self.current_episode[policy]['action_smoothness']):.4f}")
                    print(f"Episode length: {np.sum(self.current_episode[policy]['length'])}")
            
            self.rate.sleep()
        
        # Print final statistics
        print("\nComparison completed!")
        for policy in ['diffusion', 'sac']:
            print(f"\n{policy.upper()} Policy:")
            print(f"Mean planning time: {np.mean(self.metrics[policy]['planning_time']):.4f}s ± {np.std(self.metrics[policy]['planning_time']):.4f}s")
            print(f"Mean action smoothness: {np.mean(self.metrics[policy]['action_smoothness']):.4f} ± {np.std(self.metrics[policy]['action_smoothness']):.4f}")
            print(f"Mean episode length: {np.mean(self.metrics[policy]['length']):.1f} ± {np.std(self.metrics[policy]['length']):.1f}")
        
        # Plot comparison
        self.plot_comparison()
    
    def plot_comparison(self):
        """Plot comparison metrics"""
        metrics = ['planning_time', 'action_smoothness', 'length']
        labels = ['Planning Time (s)', 'Action Smoothness', 'Episode Length']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            diffusion_data = self.metrics['diffusion'][metric]
            sac_data = self.metrics['sac'][metric]
            
            axes[i].boxplot([diffusion_data, sac_data], labels=['Diffusion', 'SAC'])
            axes[i].set_title(label)
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, 'policy_comparison.png'))
        if self.args.use_wandb:
            wandb.log({"policy_comparison": wandb.Image(os.path.join(self.args.output_dir, 'policy_comparison.png'))})

def main(args: argparse.Namespace):
    comparator = PolicyComparator(args)
    comparator.compare_policies(args.num_episodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_config_path", type=str, required=True,
                      help="Path to model config file")
    parser.add_argument("--vision_encoder_path", type=str, required=True,
                      help="Path to vision encoder checkpoint")
    parser.add_argument("--diffusion_model_path", type=str, required=True,
                      help="Path to diffusion model checkpoint")
    parser.add_argument("--sac_checkpoint_path", type=str, required=True,
                      help="Path to SAC agent checkpoint")
    
    # Evaluation arguments
    parser.add_argument("--num_episodes", type=int, default=10,
                      help="Number of evaluation episodes")
    parser.add_argument("--num_samples", type=int, default=30,
                      help="Number of samples for diffusion policy")
    parser.add_argument("--hidden_dim", type=int, default=256,
                      help="Hidden dimension for networks")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save comparison results")
    
    # Misc arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run evaluation on")
    parser.add_argument("--use_wandb", action="store_true",
                      help="Whether to use wandb for logging")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args) 