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
- goal-conditioned navigation and 
- open-ended exploration
within a single architecture

**NoMaD** combines:
- Goal Masking strategy which enables flexible conditioning on optional goal images
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

## Goal Masking Mechanism

| Mode         | Mask Value | Behavior                   |
|--------------|------------|----------------------------|
| Exploration  | m=1        | Ignores goal image         |
| Goal-Seeking | m=0        | Uses goal for conditioning |

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
- We recommend to set up a conda env for which run the following commands in terminal 
```bash
conda env create -f train/train_environment.yml  
conda activate nomad_env
```
- We recommend to install the vint_train packages
```bash
pip install -e train/
```
- Install the diffusion_policy package
```bash
git clone git@github.com:real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
```
---
## Data Preprocesing
- First need to extract data from the .bag files(for SCAND).

- Run process_bags.py with the relevant args, or process_recon.py for processing RECON HDF5s.

- If you have downloaded the sacson and go_stanford dataset,they will already  be in the correct format but in the other datasets or custom datasets, make sure it follows the following structure:
- 
## Dataset Structure
- `<dataset_name>/`
  - `<name_of_traj1>/`
    - `0.jpg`
    - `1.jpg`
    - `...` (additional frame images)
    - `T_1.jpg`
    - `traj_data.pkl`
  - `<name_of_traj2>/`
    - `0.jpg`
    - `1.jpg`
    - `...` (additional frame images)
    - `T_2.jpg`
    - `traj_data.pkl`
  - `...` (additional trajectories)
  - `<name_of_trajN>/`
    - `0.jpg`
    - `1.jpg`
    - `...` (additional frame images)
    - `T_N.jpg`
    - `traj_data.pkl`



- Then, split the extracted data into train and test set. Default ratio is 80-20.

Run the following for example:
```bash
python train/data_split.py --data-dir go_stanford_extracted --dataset-name go_stanford
```

## Data Split Output Structure
-  After running data_split.py on your dataset folder with the relevant arguments, the processed split should be located inside:
```bash 
vint_release/train/vint_train/data/data_splits/
```

- It will have the following structure now:
```markdown
- `<dataset_name>/`
  - `train/`
    - `traj_names.txt`
  - `test/`
    - `traj_names.txt
```



## Training the Model
- Run this command now
```bash
PYTHONPATH=.  python train/train.py -c train/config/nomad.yaml
```

### Training Configuration:
    - optimizer: adamw
    - lr: 1e-4
    - batch_size: 47
    - epochs: 30
    - goal_mask_prob: 0.5
    - eval_freq: 1

---

## Loss Function

NoMaD is trained end-to-end with supervised learning using the following loss function:

ℒₙₒₘₐ𝒹(ϕ, ψ, f, θ, fₙ) = MSE(εₖ, εₚ(εₜ, a⁰ₜ + εₖ, k)) + λ ⋅ MSE(d(oₜ, o₉), fₙ(cₜ))
Where:  
- ϕ, ψ → Visual encoders for observation and goal images  
- f → Transformer model
- θ → Parameters of the diffusion process 
- fₙ → Temporal distance predictor
- λ = 10⁻⁴ → Hyperparameter for auxiliary loss weight

This loss function optimizes for two objectives:  
1. **Noise prediction** — Ensures the diffusion model can accurately predict the noise injected in the input actions  
2. **Temporal alignment** — Encourages the model to estimate how far the current observation is from the goal in time, improving both exploration and goal-reaching capabilities  

### Training Strategy: Goal Masking

During training, we use a goal masking probability of p_m = 0.5, meaning:
- Half the training samples use goal images (goal-reaching)
- Half ignore goals (pure exploration)

## Losses Per Epoch(add values yourself) and Model Evaluation

-------------
-------------
-------------

## Some important visualisations from wandb depicting our Losses and how our model achieving accuracy

----------
-----------
-----------

##  Nomad vs other Baselines(Masked ViNT,ViB,AutoRegressive, Random Subgoals,Subgoal Diffusion) ,and how nomad outperforms all baselines

### Unified Policy for Navigation and Exploration
- **NoMaD**: Single unified diffusion policy handles both behaviors
- **Others**: Typically require separate models for navigation vs exploration

### Action Generation
- **NoMaD**: Diffusion model decoder enables multi-modal predictions
- **Others**: Often use autoregressive models with limited multi-modality

### Goal Conditioning
- **NoMaD**: Flexible goal masking mechanism
- **Others**: Usually require separate models for different behaviors

### Architecture
- **NoMaD**: Large-scale Transformer trained on multi-robot data
- **Others**: May use smaller architectures with limited generalization

### Performance
- **NoMaD**: Fewer collisions with smaller model size
- **Others**: Often need larger models or complex planners


---

## Acknowledgements

We extend our sincere gratitude to:

- **Adith Muralidharan** (Teaching Assistant, UMC 203 at IISc and RBCCPS student) for his unwavering technical and emotional support throughout this project.

- **Professor Chiranjib Bhattacharyya** and **Professor N.Y.K. Shishir** for providing the opportunity to explore this topic through a graded term-paper in their course.

## Code & Research References
This work builds upon several foundational resources:
- Code adapted from **[visualnav-transformer](https://github.com/robodhruv/visualnav-transformer.git)**
- Research papers:
  - [ViNT: Visual Navigation Transformer](https://arxiv.org/abs/2306.14846) (Backbone for NoMaD)
  - [Vision Transformers (ViT)](https://arxiv.org/abs/2010.11929)
  - [EfficientNet](https://arxiv.org/abs/1905.11946)
  - [DiffusionPolicy](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://arxiv.org/abs/2303.04137)

---




