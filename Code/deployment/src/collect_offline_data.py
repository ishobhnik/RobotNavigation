import os
import pickle
import numpy as np
import torch
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from utils import msg_to_pil, transform_images
from topic_names import IMAGE_TOPIC, WAYPOINT_TOPIC, SAMPLED_ACTIONS_TOPIC

class OfflineDataCollector:
    def __init__(self, output_dir, model_params):
        self.output_dir = output_dir
        self.model_params = model_params
        self.context_queue = []
        self.context_size = model_params["context_size"]
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize data storage
        self.episode_data = {
            'context_vectors': [],  # EfficientNet + Transformer embeddings
            'actions': [],         # Diffusion model actions
            'predicted_noise': [], # Diffusion model noise predictions
            'images': [],          # RGB images
            'waypoints': [],       # Selected waypoints
        }
        
        # ROS subscribers
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback)
        self.waypoint_sub = rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, self.waypoint_callback)
        self.sampled_actions_sub = rospy.Subscriber(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, self.sampled_actions_callback)
        
    def image_callback(self, msg):
        """Store RGB images and update context queue"""
        obs_img = msg_to_pil(msg)
        self.episode_data['images'].append(obs_img)
        
        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(obs_img)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(obs_img)
            
    def waypoint_callback(self, msg):
        """Store selected waypoints"""
        waypoint = np.array(msg.data)
        self.episode_data['waypoints'].append(waypoint)
        
    def sampled_actions_callback(self, msg):
        """Store diffusion model actions and noise predictions"""
        actions = np.array(msg.data)
        self.episode_data['actions'].append(actions)
        
    def store_context_vector(self, context_vector):
        """Store EfficientNet + Transformer embeddings"""
        self.episode_data['context_vectors'].append(context_vector)
        
    def store_predicted_noise(self, noise):
        """Store diffusion model noise predictions"""
        self.episode_data['predicted_noise'].append(noise)
        
    def save_episode(self, episode_num):
        """Save collected data for the current episode"""
        episode_dir = os.path.join(self.output_dir, f'episode_{episode_num}')
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
            
        # Save all data
        with open(os.path.join(episode_dir, 'episode_data.pkl'), 'wb') as f:
            pickle.dump(self.episode_data, f)
            
        # Save images separately
        for i, img in enumerate(self.episode_data['images']):
            img.save(os.path.join(episode_dir, f'image_{i}.jpg'))
            
        # Reset data storage
        self.episode_data = {
            'context_vectors': [],
            'actions': [],
            'predicted_noise': [],
            'images': [],
            'waypoints': [],
        }
        self.context_queue = [] 