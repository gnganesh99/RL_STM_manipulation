
# Function to save and load RL agents and buffers


import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import h5py
import json


def save_sac_agent(agent, filename='sac_model.pth'):
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic1': agent.critic1.state_dict(),
        'critic2': agent.critic2.state_dict(),
        'value': agent.value.state_dict(),
        'actor_optimizer': agent.actor_optimizer.state_dict(),
        'critic1_optimizer': agent.critic1_optimizer.state_dict(),
        'critic2_optimizer': agent.critic2_optimizer.state_dict(),
        'value_optimizer': agent.value_optimizer.state_dict(),
    }, filename)





def load_sac_agent(agent, filename='sac_model.pth', eval_mode=False):
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic1.load_state_dict(checkpoint['critic1'])
    agent.critic2.load_state_dict(checkpoint['critic2'])
    agent.value.load_state_dict(checkpoint['value'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    agent.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
    agent.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
    agent.value_optimizer.load_state_dict(checkpoint['value_optimizer'])

    if eval_mode:
        agent.actor.eval()
        agent.critic1.eval()
        agent.critic2.eval()

    return agent




def save_replay_buffer(buffer, filename='replay_buffer.h5'):

    """
    Saves the replay buffer to an HDF5 file.
    Args:
        buffer (deque): The replay buffer containing transitions.
        filename (str): The name/path of the file to save the buffer to.
    """


    # Unzip the transitions
    states, actions, rewards, next_states, dones = zip(*buffer)

    with h5py.File(filename, 'w') as f:
        f.create_dataset('states', data=np.array(states))
        f.create_dataset('actions', data=np.array(actions))
        f.create_dataset('rewards', data=np.array(rewards))
        f.create_dataset('next_states', data=np.array(next_states))        
        f.create_dataset('dones', data=np.array(dones))





def load_replay_buffer(filename='replay_buffer.h5'):

    """
    Loads the replay buffer from an HDF5 file.
    Args:
        filename (str): The name/path of the file to load the buffer from.

    Returns:
        buffer: A deque containing the transitions from the replay buffer in the order (states, actions, rewards, next_states, dones).
                In case of a HER-buffer, the transitions will be in the form of (state, action, reward, next_state, dones).

    """


    buffer = deque()


    with h5py.File(filename, 'r') as f:

        states = f['states'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        next_states = f['next_states'][:]
        dones = f['dones'][:]

        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            buffer.append((s, a, r, ns, d))

    return buffer





def save_dict_to_h5(data_dict, filename, mode='w'):
    """
    Save a (possibly nested) dictionary into an HDF5 file.

    Parameters
    ----------
    data_dict : dict
        Dictionary with keys as dataset/group names and values as arrays, scalars, or dicts.
    filename : str
        Path to the HDF5 file to save.
    mode : str, optional
        File mode ('w' = overwrite, 'a' = append, etc.), by default 'w'.
    """
    def _recursively_save_dict(h5_group, dict_obj):
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                subgroup = h5_group.create_group(key)
                _recursively_save_dict(subgroup, value)
            else:
                # Overwrite existing dataset if it exists
                if key in h5_group:
                    del h5_group[key]
                h5_group.create_dataset(key, data=np.array(value))

    with h5py.File(filename, mode) as f:
        _recursively_save_dict(f, data_dict)




def load_h5_to_dict(filename):
    """
    Load an HDF5 file into a nested dictionary.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Returns
    -------
    dict
        Nested dictionary with numpy arrays/scalars as values.
    """
    def _recursively_load_dict(h5_group):
        result = {}
        for key, item in h5_group.items():
            if isinstance(item, h5py.Group):
                result[key] = _recursively_load_dict(item)
            elif isinstance(item, h5py.Dataset):
                data = item[()]  # read dataset into memory
                # Convert 0-dim arrays to scalars
                if isinstance(data, np.ndarray) and data.shape == ():
                    data = data.item()
                result[key] = data
        return result

    with h5py.File(filename, 'r') as f:
        return _recursively_load_dict(f)
    


def save_dict_to_txt(data_dict, filename, mode='w'):

    with open(filename, mode, encoding='utf-8') as f:
        for key, value in data_dict.items():
            f.write(f"{key}: {value}\n")


def save_dict_to_json(data_dict, filename, mode='w'):
    with open(filename, mode, encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

def load_json_to_dict(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)