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

## Model Architecture

- **Encoder**: EfficientNet-B0
- **Temporal Context**: Transformer with 4 layers and 4 heads
- **Diffusion Decoder**: Conditional UNet1D
- **Objective**: MSE between predicted noise and sampled noise, plus auxiliary temporal distance loss

---
## Project Structure
```
RobotNavigation/ 
│ ├── train/ # Training scripts and utilities 
│ ├── train.py # Entry point for training 
│ ├── config/ # YAML config files 
│ ├── vint_train/ # Dataset, models, training utils 
│ ├── requirements.txt
│ └── README.md
```
---
## Set up
### Download datasets:
- RECON :
- SCAND :
- go_stanford:
- Sacson :

### clone this repo and setup the environment:
-We recommend to set up a conda env
-run the following commands in terminal 
```bash
conda env create -f train_environment.yml
conda activate nomad_env
```
---
## Data Preprocesing
First need to extract data from the .bag files.
Then, split the extracted data into train and test set. Default ratio is 80-20.

Run the following for example:
```bash
python train/data_split.py --data-dir go_stanford_extracted --dataset-name go_stanford
```

This creates a split inside:
```
train/vint_train/data/data_splits/go_stanford/
```



