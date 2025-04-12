## Authors

- Abhishek kumar Jha
- Shobnik kriplani
- Sehaj Ganjoo
- Namashivaaya V.



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
-Goal Masking strategy which enables flexible conditioning on optional goal images
- EfficientNet-based encoders for image observations
- A Vision Transformer (ViNT) backbone for temporal modeling
- A Diffusion-based decoder for multi-step action prediction
  
![image](https://github.com/user-attachments/assets/b5e531c0-a6b0-4707-b335-f85aef93c979)

---

## Model Architecture

1. **Visual Encoder**:
   - EfficientNet-B0 backbone processing RGB observations  
   - Output: 256-dimensional embeddings per image

2. **Goal Fusion Encoder**:
   - Processing current observation and goal image  
   - Implements attention masking for goal conditioning

3. **Transformer Decoder**:
   - 4-layer architecture with 4 attention heads (multi-headed attention layers) to obtain a sequence of context vectors that are concatenated to obtain the final context vector Ct. 

4. **Diffusion Policy**:
   - 15-layer 1D U-Net for noise prediction  
   - Square Cosine Noise Scheduler with K=10 steps

5.   **Objective**: MSE between predicted noise and sampled noise, plus auxiliary temporal distance loss

---
## Project Structure
```
RobotNavigation/ 
│ ├── train/ # Training scripts and utilities 
│ ├── train.py # Training the model  
│ ├── config/ # YAML config files(defaults.yaml,gnm.yaml,late_fusion.yaml .... though we configured nomad.yaml file)
│ ├── vint_train/ # Dataset(data/),models,processing data, training evaluation,training utils as well as visualisation
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
- tartan_drive:

### clone this repo and setup the environment:
-We recommend to set up a conda env
-run the following commands in terminal 
```bash
conda env create -f train/train_environment.yml  
conda activate nomad_env
```
- We recommend to install the vint_train packages
```bash
pip install -e train/
```
-Install the diffusion_policy package
```bash
git clone git@github.com:real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
```
---
## Data Preprocesing
First need to extract data from the .bag files(for SCAND).
Run process_bags.py with the relevant args, or process_recon.py for processing RECON HDF5s.
If you have downloaded the sacson and go_stanford dataset,  wthey will already  be in the correct format but in the other datasets or custom datasets, make sure it follows the following structure:
├── <dataset_name>
│   ├── <name_of_traj1>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_1.jpg
│   │   └── traj_data.pkl
│   ├── <name_of_traj2>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_2.jpg
│   │   └── traj_data.pkl
│   ...
└── └── <name_of_trajN>
    	├── 0.jpg
    	├── 1.jpg
    	├── ...
        ├── T_N.jpg
        └── traj_data.pkl


        
Then, split the extracted data into train and test set. Default ratio is 80-20.

Run the following for example:
```bash
python train/data_split.py --data-dir go_stanford_extracted --dataset-name go_stanford
```

This creates a split inside:
```
train/vint_train/data/data_splits/go_stanford/
```



