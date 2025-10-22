# RL_STM_manipulation
## Automated STM manipulation using RL agents.

In this work, we develop an automated framework for scanning tunneling microscopy (STM) manipulation driven by reinforcement learning (RL). The workflow leverages RL agents to optimize key STM experimental parameters. To provide the agent with real-time feedback, we incorporate a YOLO-based object detection model capable of identifying carbon monoxide (CO) molecules on copper Cu(111) surface.

## Overview

This project implements automated STM (Scanning Tunneling Microscope) manipulation using reinforcement learning (RL) agents. The system leverages deep reinforcement learning techniques, specifically Soft Actor-Critic (SAC) algorithm, to enable intelligent manipulation of molecules and atoms at the nanoscale. The methodology combines computer vision for molecular detection using YOLO object detection models with multi-objective reinforcement learning for adaptive manipulation strategies.

The system operates by detecting molecular positions through automated image analysis, planning manipulation trajectories using path planning algorithms (including RRT* for obstacle avoidance), and executing precise manipulation actions through learned RL policies. 

We employ two RL agents:

1. Parameter agent: Learns to optimize manipulation parameters such as bias voltage, setpoint current, and tip speed based on real-time feedback from the STM environment.

2. Positioning agent: Learns and introduces target-positioning offsets according to the local molecular configuration described in the state variables.

- **[`Manipulation_MO_sac_gui_user_input.ipynb`](./Manipulation_MO_sac_gui_user_input.ipynb)** - The notebook serves as the primary interface to drive automated experiments, allowing users to input target configurations and initiate autonomous manipulation sequences through an intuitive GUI-based workflow.

## Key Components

- **[`Manipulation_env.py`](./Manipulation_env.py)** - Core RL environment implementing the multi-object manipulation gymnasium interface with state/action spaces and reward functions

- **[`sac_networks.py`](./sac_networks.py)** - Soft Actor-Critic neural network implementations including Actor, Critic networks and training algorithms

- **[`detect_molecules.py`](./detect_molecules.py)** - YOLO-based molecular detection and position tracking functionality for automated image analysis

- **[`Manipulation_coords.py`](./Manipulation_coords.py)** - Coordinate system transformations between STM space and manipulation coordinates

- **[`stm_manipulation_experiments.py`](./stm_manipulation_experiments.py)** - STM experimental routines for executing manipulation actions via Nanonis TCP communication

- **[`buffer.py`](./buffer.py)** - Experience replay buffer implementations including standard replay and Hindsight Experience Replay (HER)

- **[`drift_estimation.py`](./drift_estimation.py)** - Drift correction algorithms to compensate for thermal drift between successive STM scans

- **[`rrt_star_algo.py`](./rrt_star_algo.py)** - RRT* path planning algorithm implementation for collision-free manipulation trajectory generation

- **[`yolo_param_tuning.py`](./yolo_param_tuning.py)** - Automated hyperparameter optimization for YOLO detection models using Optuna

- **[`get_reward.py`](./get_reward.py)** - Reward function implementations for RL training including displacement and state-based rewards

- **[`get_target.py`](./get_target.py)** - Target selection and generation utilities for manipulation experiments

- **[`path_fns.py`](./path_fns.py)** - Path planning and coordinate transformation utilities for manipulation trajectories

- **[`nanonis_TCP.py`](./nanonis_TCP.py)** - TCP communication interface for Nanonis STM control software

- **[`experimental_routines.py`](./experimental_routines.py)** - High-level experimental workflow coordination and automation routines using Labview client.

- **[`expt_utils.py`](./expt_utils.py)** - General utility functions for experiment management and data processing

- **[`stm_utils.py`](./stm_utils.py)** - STM-specific utilities for image processing and data handling

- **[`env_functions.py`](./env_functions.py)** - Environment helper functions for RL training and evaluation

- **[`display_results.py`](./display_results.py)** - Visualization and result analysis tools for manipulation experiments

- **[`target_control.txt`](./target_control.txt)** - Configuration file for target manipulation parameters

- **[`model_buffer_saving_functions.py`](./model_buffer_saving_functions.py)** - Model checkpointing and buffer utilities

## Credits

The nanonisTCP repository is based on ['Julian Ceddia'/'nanonisTCP'](https://github.com/New-Horizons-SPM/nanonisTCP/tree/v1.0.2), doi:10.5281/zenodo.7402664