# MARBLER: Multi-Agent RL Benchmark and Learning Environment for the Robotarium
Fork used for Generalization of Heterogeneous Multi-Robot Policies via Awareness and Communication of Capabilities 

## Installation Instructions
1. Activate the ca-gnn-marl Conda Environment: `conda activate ca-gnn-marl`. 
2. Download and Install the [Robotarium Python Simulator](https://github.com/robotarium/robotarium_python_simulator)
- The version of the Robotarium that this code was run with was commit 6bb184e. As of now, the code will run on the most recent version of the Robotarium but will not train.
3. Install our environment by running `pip install -e .` in this directory
4. To test successfull installation, run `python3 -m robotarium_gym.main` to run a pretrained model

## Usage
* To look at current scenarios or create new ones or to evaluate trained models, look at the README in robotarium_gym
* To upload the agents to the Robotarium, look at the README in robotarium_eval

## Training with our version of EPyMARL
1. Train agents normally using our gym keys
- For example: `python3 src/main.py with alg_yaml=qmix env_yaml=gymma env_args.time_limit=1000 env_args.key="robotarium_gym:PredatorCapturePrey-v0"`
- To train faster, ensure `robotarium` is False, `real_time` is False, and `show_figure_frequency` is large or -1 in the environment's `config.yaml`
- Known error: if `env_args.time_limit<max_episode_steps`, EPyMARL will crash after the first episode
2. Copy the trained weights to the models folder for the scenario that was trained
- Requires the agent.th file (location should be printed in the cout of the terminal the model was trained in, typically in EPyMARL/results/models/...)
- Requires the config.json file (typically in EPyMARL/results/algorithm_name/gym:scenario/...)
3. Update the scenario's config.yaml to use the newly trained agents


## Citing
If you use this fork of MARBLER, please cite:
* MARBLER:
> Reza Torbati, Shubham Lohiya, Shivika Singh, Meher Shashwat Nigam, & Harish Ravichandar. (2023). MARBLER: An Open Platform for Standarized Evaluation of Multi-Robot Reinforcement Learning Algorithms. 
* Our paper:
> Pierce Howell, Max Rudolph, Reza Torbati, Kevin Fu, & Harish Ravichandar (2023). Generalization of Heterogeneous Multi-Robot Policies via Awareness and Communication of Capabilities. In 7th Annual Conference on Robot Learning.
