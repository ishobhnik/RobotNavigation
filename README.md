## **NoMaD: Navigation with Goal-Masked Diffusion** 

This github repository presents our work on implementing and analyzing the training and
perception components of a visual navigation system based on diffusion policies, as
adapted from the NOMAD (Navigation with Goal Masked Diffusion) framework.

Robotic learning for navigation in unfamiliar environments requires the ability to perform
both task-oriented navigation (i.e., reaching a known goal) and task-agnostic exploration
(i.e., searching for a goal in a novel environment). Traditionally, these functionalities are
tackled by separate systems — for example, using subgoal proposals, explicit planning mod-
ules, or distinct navigation strategies for exploration and goal-reaching

The NoMaD (Navigation with Goal-Masked Diffusion) framework introduces a unified visual
navigation policy capable of both :
i) goal-conditioned navigation and 
ii) open-ended exploration
within a single architecture

**NoMaD** combines:
- EfficientNet-based encoders for image observations
- A Vision Transformer (ViNT) backbone for temporal modeling
- A Diffusion-based decoder for multi-step action prediction
  
![image](https://github.com/user-attachments/assets/b5e531c0-a6b0-4707-b335-f85aef93c979)

---
## Project Structure
RobotNavigation/ 
│ ├── train/ # Training scripts and utilities 
│ ├── train.py # Entry point for training 
│ ├── config/ # YAML config files 
│ ├── vint_train/ # Dataset, models, training utils 
│ ├── requirements.txt
│ └── README.md
