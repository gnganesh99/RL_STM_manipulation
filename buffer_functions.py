import numpy as np
from env_functions import rescale_array


import d3rlpy
from d3rlpy.dataset import ReplayBuffer
import d3rlpy
from d3rlpy.algos import DiscreteCQL
from d3rlpy.algos import DiscreteSAC
from d3rlpy.algos import CQL
from d3rlpy.dataset import MDPDataset 


def save_buffer(buffer, buffer_name):

    with open(buffer_name, 'w+b') as f:
        buffer.dump(f)

def load_buffer(buffer_name):

    with open(buffer_name, 'rb') as f:
        buffer = ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())

    return buffer

def append_buffer(buffer_main, buffer_next):

    for episode in buffer_next.episodes:

        buffer_main.append_episode(episode)

    return buffer_main

def buffer_len(buffer):

    length = len(buffer.episodes)

    return length


def scaled_buffer_to_MDPdataset(buffer, rl_range_old, rl_range_new, old_action_range, new_action_range):

    """
    Rescales the action variables of the buffer and returns the MDPdataset

    Args:
        buffer: d3rlpy replay buffer
        rl_range1: old action space range eg: [[-1, 1], [-1, 1], [-1, 1]] 
        rl_range2: new action space range.
        old_action_range: old real action ranges, eg: [[0.01, 0.03],[40, 80], [0.5, 5]]
        new_action_range: new realaction ranges

    Returns:
        dataset: MDP dataset (d3rlpy buffer) with action variables rescaled to rl_range2
    """
    
    episodes = buffer.episodes
    n_episodes = len(episodes)

    for i in range(n_episodes):

        episode = episodes[i]
    
        observations = episode.observations
        actions = episode.actions
        new_actions =  rescale_episode_actions(actions, rl_range_old, rl_range_new, old_action_range, new_action_range)
        #print(new_actions)
        rewards = episode.rewards
        terminated = episode.terminated

        episode_len = len(episode)

        terminals = np.zeros((episode_len,1)).astype(bool)
        if terminated == True:
            terminals[-1] = True

        terminals = np.asarray(terminals)
        #print(terminals)

        if i == 0:
            all_observations = observations
            all_actions = new_actions
            all_rewards = rewards
            all_terminals = terminals

        else:

            all_observations = np.vstack((all_observations, observations))
            all_actions = np.vstack((all_actions, new_actions))
            all_rewards = np.vstack((all_rewards, rewards))
            all_terminals = np.vstack((all_terminals, terminals))


    all_observations = np.asarray(all_observations)
    all_actions = np.asarray(all_actions)
    all_rewards = np.asarray(all_rewards)
    all_terminals = np.asarray(all_terminals)
    #print(all_terminals)
        
    dataset = MDPDataset(all_observations, all_actions, all_rewards, all_terminals)

    return dataset




def rescale_episode_actions(episode_actions, rl_range1, rl_range2, old_action_range, new_action_range):

    """
    Rescales the action variables of for the episode actions

    Args:
        buffer: episode actions: buffer.episodes[i].actions
        rl_range1: old action space range eg: [[-1, 1], [-1, 1], [-1, 1]] 
        rl_range2: new action space range.
        old_action_range: old real action ranges, eg: [[0.01, 0.03],[40, 80], [0.5, 5]]
        new_action_range: new realaction ranges

    Returns:
        new episode actions: rescaled episode actions
    """
    
    new_episode_actions = []

    for action in episode_actions:

        action = np.ravel(np.asarray(action))

        # First rescale to real values
        action1 = rescale_array(rl_range1, action, old_action_range)
        action1 = np.ravel(action1)

        #Rescale to rl range based on new real range
        action2 = rescale_array(new_action_range, action1, rl_range2)
        action2 = np.ravel(action2)

        new_episode_actions.append(action2)

    new_episode_actions = np.asarray(new_episode_actions)

    return new_episode_actions




def add_observation_noise(buffer, noise_factor = 0.01, mean = 0, std_dev = 1, shuffle_episodes = False):

    """
    Returns MDP dataset after adding noise to the observation/state variables
    the noise is a normal distribution at mean = 0, stdev = 1.

    Args:
        buffer: input buffer
        noise_factor: (default = 0.01). Factor that is multiplied to the noise
        shuffle_episodes: (default = False) shuffles the episodes in the output buffer

    Returns:
        dataset: d3rlpy output dataset/replay buffer

    """
    
    episodes = buffer.episodes
    n_episodes = len(episodes)
    indices =  np.linspace(0, n_episodes-1, n_episodes).astype(int)

    if shuffle_episodes == True:
        np.random.shuffle(indices) 

    first_index = True

    for index in indices:

        episode = episodes[index]
    
        observations = episode.observations
        observations = np.asarray(observations)
        noise = np.random.normal(mean, std_dev, observations.shape)
        observations = observations + noise_factor*noise


        actions = episode.actions
        rewards = episode.rewards
        terminated = episode.terminated

        episode_len = len(episode)

        terminals = np.zeros((episode_len, 1)).astype(bool)
        if terminated == True:
            terminals[-1] = True

        terminals = np.asarray(terminals)
        #print(terminals)

        if first_index == True:
            all_observations = observations
            all_actions = actions
            all_rewards = rewards
            all_terminals = terminals
            
            first_index = False

        else:

            all_observations = np.vstack((all_observations, observations))
            all_actions = np.vstack((all_actions, actions))
            all_rewards = np.vstack((all_rewards, rewards))
            all_terminals = np.vstack((all_terminals, terminals))


    all_observations = np.asarray(all_observations)
    all_actions = np.asarray(all_actions)
    all_rewards = np.asarray(all_rewards)
    all_terminals = np.asarray(all_terminals)
    #print(all_terminals)
        
    dataset = MDPDataset(all_observations, all_actions, all_rewards, all_terminals)

    return dataset



def add_action_noise(buffer, noise_factor = 0.001, mean = 0, std_dev = 1, shuffle_episodes = False):

    """
    Returns MDP dataset after adding noise to the action variables
    the noise is a normal distribution at mean = 0, stdev = 1.

    Args:
        buffer: input buffer
        noise_factor: (default = 0.001). Factor that is multiplied to the noise
        shuffle_episodes: (default = False) shuffles the episodes in the output buffer

    Returns:
        dataset: d3rlpy output dataset/replay buffer

    """
    
    episodes = buffer.episodes
    n_episodes = len(episodes)
    indices =  np.linspace(0, n_episodes-1, n_episodes).astype(int)

    if shuffle_episodes == True:
        np.random.shuffle(indices) 

    first_index = True

    for index in indices:

        episode = episodes[index]
    
        observations = episode.observations

        actions = episode.actions
        actions = np.asarray(actions)
        noise = np.random.normal(mean, std_dev, actions.shape)
        actions = actions + noise_factor*noise
        
        rewards = episode.rewards
        terminated = episode.terminated

        episode_len = len(episode)

        terminals = np.zeros((episode_len, 1)).astype(bool)
        if terminated == True:
            terminals[-1] = True

        terminals = np.asarray(terminals)
        #print(terminals)

        if first_index == True:
            all_observations = observations
            all_actions = actions
            all_rewards = rewards
            all_terminals = terminals
            
            first_index = False

        else:

            all_observations = np.vstack((all_observations, observations))
            all_actions = np.vstack((all_actions, actions))
            all_rewards = np.vstack((all_rewards, rewards))
            all_terminals = np.vstack((all_terminals, terminals))


    all_observations = np.asarray(all_observations)
    all_actions = np.asarray(all_actions)
    all_rewards = np.asarray(all_rewards)
    all_terminals = np.asarray(all_terminals)
    #print(all_terminals)
        
    dataset = MDPDataset(all_observations, all_actions, all_rewards, all_terminals)

    return dataset



def terminate_done_episode(buffer, reward_th = 9, done_reward = 100):

    """
    terminates the episode length upon "done", decided by the reward value.
    This is required for buffers, where trajectory continues inspite of manipulation "done"

    Args:
        buffer: d3rlpy replay buffer
        reward_th: reward threshold. if reward > reward_th, the epsidoe is terminated here. 
        done_reward: the new done_reward. where the episode ends

    Returns:
        dataset: MDP dataset (d3rlpy buffer) with the done episodes terminated
    """
    
    episodes = buffer.episodes
    n_episodes = len(episodes)

    for i in range(n_episodes):

        episode = episodes[i]
        
        n_transitions =  len(episode)
        terminated = episode.terminated

        new_rewards = []
        for j in range(n_transitions):
            reward = episode.rewards[j]
            
            
            if reward[0] > reward_th:
                reward = [done_reward]           
                terminated = True
                
        

            new_rewards.append(reward)
            
            if reward[0] == done_reward:
                break
        
        new_rewards = np.array(new_rewards)
        new_episode_len =  len(new_rewards)       
        
        observations = episode.observations[0:new_episode_len, :]
        actions = episode.actions[0:new_episode_len, :]

        terminals = np.zeros((new_episode_len,1)).astype(bool)

        

        if terminated == True:
            terminals[-1] = True

        terminals = np.asarray(terminals)
        
        if i == 0:
            all_observations = observations
            all_actions = actions
            all_rewards = new_rewards
            all_terminals = terminals

        else:

            all_observations = np.vstack((all_observations, observations))
            all_actions = np.vstack((all_actions, actions))
            all_rewards = np.vstack((all_rewards, new_rewards))
            all_terminals = np.vstack((all_terminals, terminals))


    all_observations = np.asarray(all_observations)
    all_actions = np.asarray(all_actions)
    all_rewards = np.asarray(all_rewards)
    all_terminals = np.asarray(all_terminals)
    #print(all_terminals)
        
    dataset = MDPDataset(all_observations, all_actions, all_rewards, all_terminals)

    return dataset