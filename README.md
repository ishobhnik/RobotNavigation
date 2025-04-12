## Authors
- [Abhishek Kumar Jha](https://github.com/abhi-abhi-101)
- [Shobhnik Kriplani](https://github.com/ishobhnik)
- [Sehaj Ganjoo](https://github.com/Aureola21)
- [Namashivaaya V.](https://github.com/NamashivayaaV)

## **NoMaD: Navigation with Goal-Masked Diffusion** 

This github repository presents our work on implementing and analyzing the training and
perception components of a visual navigation system based on diffusion policies, as
adapted from the NoMaD (Navigation with Goal Masked Diffusion) framework.

Robotic learning for navigation in unfamiliar environments requires the ability to perform
both task-oriented navigation (i.e., reaching a known goal) and task-agnostic exploration
(i.e., searching for a goal in a novel environment). Traditionally, these functionalities are
tackled by separate systems — for example, using subgoal proposals, explicit planning 
modules, or distinct navigation strategies for exploration and goal-reaching

The NoMaD (Navigation with Goal-Masked Diffusion) framework introduces a unified visual
navigation policy capable of both :
- goal-conditioned navigation and 
- open-ended exploration
within a single architecture

**NoMaD** combines:
- Goal Masking strategy which enables flexible conditioning on optional goal images
- EfficientNet-based encoders for image observations
- A Visual Navigation Transformer (ViNT) backbone for temporal modeling
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
- [RECON](https://sites.google.com/view/recon-robot/dataset)
- [SCAND](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi%3A10.18738%2FT8%2F0PRYRH)
- [Sacson/HuRoN](https://sites.google.com/view/sacson-review/huron-dataset)
### clone this repo and setup the environment:
- We recommend to set up a conda env for which run the following commands in terminal 
```bash
conda env create -f train/train_environment.yml  
conda activate nomad_train 
```
- Install all the Dependencies
```bash
pip install -e train/
```
- Install the diffusion_policy package
```bash
git clone git@github.com:real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
```
---
## Data Preprocessing
- First need to extract data from the .bag files(for SCAND and RECON).

- Run process_bags.py with the relevant args, or process_recon.py for processing RECON HDF5s.

- If you have downloaded the sacson dataset, it's already in the correct format but in the other datasets or custom datasets, make sure it follows the following structure:
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
python train/data_split.py --data-dir #path of Downloaded Dataset --dataset-name scand
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
    - `traj_names.txt`
```

## Training the Model
- Run this command now
```bash
 python train/train.py -c train/config/nomad.yaml
```

### Training Configuration:
    - optimizer: adamw
    - lr: 1e-4
    - scheduler: "cosine"
    - batch_size: 47
    - epochs: 10
    - goal_mask_prob: 0.5
    - eval_freq: 1
    - num_diffusion_iters: 10
    - lamda : 1e-4
    
---

## Loss Function

The NoMaD model is trained using a combination of diffusion loss and temporal distance prediction loss. The total training objective is given by:
```math
\mathcal{L}_{\text{NoMaD}}(\phi, \psi, f, \theta, f_n) = \text{MSE}(\varepsilon_k, \varepsilon_{\theta}(c_t, a_t^0 + \varepsilon_k, k)) + \lambda \cdot \text{MSE}(d(o_t, o_g), f_n(c_t))
```

Where:
- $\phi$, $\psi$ are visual encoders for observation and goal images
- $f$ is the Transformer encoder
- $\theta$ are the parameters of the diffusion model
- $f_n$ is the temporal distance predictor
- $\lambda$ is a weighting coefficient
- $\varepsilon_k$ is noise sampled from $\mathcal{N}(0, I)$
- $c_t$ is the encoded context
- $a_t^0$ is the clean action at time $t$
- $d(o_t, o_g)$ is the temporal distance between observation and goal

This loss function optimizes for two objectives:  
1. **Noise prediction** — Ensures the diffusion model can accurately predict the noise injected in the input actions  
2. **Temporal alignment** — Encourages the model to estimate how far the current observation is from the goal in time, improving both exploration and goal-reaching capabilities  

### Training Strategy: Goal Masking

During training, we use a goal masking probability of Pm = 0.5, meaning:
- Half the training samples use goal images (goal-reaching)
- Half ignore goals (pure exploration)

## Some important visualisations from wandb depicting our Losses and how our model achieving accuracy
##  Nomad vs Other Baselines

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

### NoMaD (goal conditioned) vs ViNT

Total Loss
<table> <tr> <td><b>ViNT</b><br><img src="https://github.com/user-attachments/assets/642ea7d5-5e1f-419d-9441-7e570dfb1744" width="400"></td> <td><b>NoMaD</b><br><img src="https://github.com/user-attachments/assets/6bebcbfb-22cd-47d4-8f4e-235e63f6a733" width="400"></td> </tr> </table>
For Training
<table> <tr> <td><b>ViNT</b><br><img src="https://github.com/user-attachments/assets/83e4e442-2798-4a72-a10a-59b1491ab1bd" width="400"></td> <td><b>NoMaD</b><br><img src="https://github.com/user-attachments/assets/b2bb7cd6-cedb-4adb-b32d-8ac56adbd9d5" width="400"></td> </tr> </table>
For Test
<table> <tr> <td><b>ViNT</b><br><img src="https://github.com/user-attachments/assets/0d5fecb7-a904-4e22-ba73-93d75bdea347" width="400"></td> <td><b>NoMaD</b><br><img src="https://github.com/user-attachments/assets/e8cf3fdb-d2f7-413c-a834-7821a66a6284" width="400"></td> </tr> </table>
Action Loss
Training
<table> <tr> <td><b>ViNT</b><br><img src="https://github.com/user-attachments/assets/22e494eb-2141-4807-88e4-542d9a619af6" width="400"></td> <td><b>NoMaD</b><br><img src="https://github.com/user-attachments/assets/94482246-2ce4-425b-a33e-d87396a4a158" width="400"></td> </tr> </table>
Test
<table> <tr> <td><b>ViNT</b><br><img src="https://github.com/user-attachments/assets/cf5cc286-39d1-4c26-8a2b-f26df0a288ce" width="400"></td> <td><b>NoMaD</b><br><img src="https://github.com/user-attachments/assets/df87ecf4-69cd-4cf5-9c9a-17436ad893bc" width="400"></td> </tr> </table>
Multi-action Waypoints Cosine Similarity
Training
<table> <tr> <td><b>ViNT</b><br><img src="https://github.com/user-attachments/assets/4c80c311-ed25-474a-ab9b-afc45e34885f" width="400"></td> <td><b>NoMaD</b><br><img src="https://github.com/user-attachments/assets/47f97940-cc2a-412c-9067-41cab18ab905" width="400"></td> </tr> </table>
Test
<table> <tr> <td><b>ViNT</b><br><img src="https://github.com/user-attachments/assets/2a092199-7f53-4ebb-a771-e2d160be0e09" width="400"></td> <td><b>NoMaD</b><br><img src="https://github.com/user-attachments/assets/30d12d95-fc20-4242-957c-5f4740f99041" width="400"></td> </tr> </table>

## Acknowledgements

We extend our sincere gratitude to:

- **Adith Muralidharan** (Teaching Assistant, UMC 203) for his unwavering technical support throughout this project.

- **Professor N.Y.K. Shishir** and **Professor Chiranjib Bhattacharyya** for providing the opportunity to explore this topic through a graded term-paper in their course.

## Code & Research References
This work builds upon several foundational resources:
- Research papers:
  - [NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration](https://arxiv.org/pdf/2310.07896)
  - [ViNT: Visual Navigation Transformer](https://arxiv.org/abs/2306.14846) (Backbone of NoMaD)
  - [Vision Transformers (ViT)](https://arxiv.org/abs/2010.11929)
  - [EfficientNet](https://arxiv.org/abs/1905.11946)
  - [DiffusionPolicy](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://arxiv.org/abs/2303.04137)

- Code adapted from **[visualnav-transformer](https://github.com/robodhruv/visualnav-transformer.git)**

---
