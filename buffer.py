"""
Replay buffer implementations for reinforcement learning.
Includes standard experience replay and Hindsight Experience Replay (HER).

@author: Ganesh Narasimha
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from gym import Env

from collections import deque



class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    """

    def __init__(self, capacity="inf"):

        if isinstance(capacity, int):
            self.buffer = deque(maxlen=capacity)
        else:
            self.buffer = deque()

    def push(self, state, action, reward, next_state, done):

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        

        return self.format_batch(batch)
        
    
    def sample_recent(self, batch_size: int, recency_factor = 1):

        """
        Samples recent-experience-enhanced batch.

        Args:
            batch_size (int): Number of transitions to sample.
            recency_factor (float): Factor to weight recent experiences more heavily.

        Returns:
            states (torch.FloatTensor): -> shape (batch_size, state_dim)
            actions (torch.FloatTensor): -> shape (batch_size, action_dim)
            rewards (torch.FloatTensor): -> shape (batch_size, 1)
            next_states (torch.FloatTensor): -> shape (batch_size, state_dim)
            dones (torch.FloatTensor): -> shape (batch_size, 1)
        """

        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        N = len(self.buffer)
        probs = np.exp(recency_factor*(np.linspace(-1, 0, N)))
        probs /= np.sum(probs)
        indices = np.random.choice(np.arange(N), size=batch_size, p=probs)

        batch = [self.buffer[i] for i in indices]

        return self.format_batch(batch)


    def __len__(self):
        return len(self.buffer)
    
        
    
    def format_batch(self, batch):

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            self._fmt_size(states),         # -> (batch_size, state_dim)
            self._fmt_size(actions),        # -> (batch_size, action_dim)
            self._fmt_size(rewards),        # -> (batch_size, 1)
            self._fmt_size(next_states),    # -> (batch_size, state_dim)
            self._fmt_size(dones)           # -> (batch_size, 1)
        )
 
    @staticmethod   
    def _fmt_size(x):
        
        x = torch.FloatTensor(x)

        if x.dim() == 3 and x.size(1) == 1:  # If x is 3D with second dimension 1, squeeze it to 2D
            x = x.squeeze(1)
        
        if x.dim() == 1: # If x is 1D, make it 2D
            x = x.unsqueeze(1)

        return x
    




class HERReplayBuffer:
    """
    Hindsight Experience Replay (HER) buffer for goal-conditioned RL.

    Attributes:
        capacity (int): Maximum number of transitions to store.
        buffer (list): Circular buffer holding transitions.
        episode_buffer (list): Temporary storage for one episode's transitions.
        position (int): Current write position in the circular buffer.
        k_future (int): Number of future goals to sample per transition.
    """
    def __init__(self, k_future: int = 4, capacity = 'inf', env: Env = None, use_env_reward_fn = False):
        """
        Initialize the HER replay buffer.

        Args:
           
            state_dim (int): Dimensionality of state.
            action_dim (int): Dimensionality of action.
            goal_dim (int): Dimensionality of goal.
            k_future (int): Number of HER samples per transition.
            capacity (int): Max number of transitions (default is infinite capacity).
        """
        self.capacity = capacity
        if isinstance(self.capacity, int):
            self.buffer = deque(maxlen=capacity)
        else:
            self.buffer = deque()

        self.episode_buffer = []
        self.position = 0
        self.k_future = k_future
        self.env = env  # Environment to compute rewards if needed
        self.use_env_reward_fn = use_env_reward_fn

    def state_goal_combination(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Combine state and goal into a single vector.

        Args:
            state (np.ndarray): Current state.
            goal (np.ndarray): Goal for this transition.

        Returns:
            np.ndarray: Combined state-goal vector.
        """
        state = np.array(state).reshape(-1)
        goal = np.array(goal).reshape(-1)

        return np.concatenate((state, goal), axis=-1)
    

    def store_episode(self, episode: list, **kwargs):
        """
        Store one full episode with HER relabeling.

        Args:
            episode (list): List of transitions (state, action, reward, next_state, done).
        """
        episode_len = len(episode) 
        
        env = kwargs.get("env", None)
        if env is not None:
            self.env = env # update the timestamp of env

        for t in range(episode_len):
            state, action, _, next_state, goal, next_ag, orig_done = episode[t]


            #Recompute reward based on the designed reward function for goal-conditioned RL
            reward, done = self._compute_reward(next_ag, goal)

            done = bool(orig_done or done)  # Combine original done with goal-based done

            combined_state = self.state_goal_combination(state, goal)  # Combine state and goal
            combined_next_state = self.state_goal_combination(next_state, goal)        

            # Store original transition
            self._add(combined_state, action, reward, combined_next_state, done)


            # Store additional HER transitions
            # This is key in using the HER: Sample k future goals from the episode. Now assume that the agent has achieved the goal at t, and compute the reward for the new set of goals.
            # This way you are augmenting the experience of the agent with the future goals. THe size of the buffer samples increases by k_future

            if t + 1 < episode_len:

                future_idxs = np.random.randint(t + 1, episode_len, size=self.k_future)

                future_idxs = np.unique(future_idxs)  # Ensure unique future indices

                for future_t in future_idxs:
                    
                    future_goal = episode[future_t][5]

                    new_reward, new_done = self._compute_reward(next_ag, future_goal)

                    new_done = bool(orig_done or new_done)  # Combine original done with goal-based done

                    # Combine state and future goal
                    combined_state = self.state_goal_combination(state, future_goal)
                    combined_next_state = self.state_goal_combination(next_state, future_goal)

                    self._add(combined_state, action, new_reward, combined_next_state, new_done)



    def _add(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done:bool ) -> None:
        """
        Add a single transition to the circular buffer.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state observed.
            done: (terminated or truncated) for this transition.
        """

            
        reward = np.array(reward).reshape(-1)  # Ensure reward is a 1D array
        self.buffer.append((state, action, reward, next_state, done))


        # Increment position in circular buffer. this is not necessary!!!
        if isinstance(self.capacity, int):
            self.position = (self.position + 1) % self.capacity
        else:
            self.position += 1



    def sample(self, batch_size: int):
        """
        Sample a batch of transitions for training.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            states (torch.FloatTensor): -> shape (batch_size, state_dim)
            actions (torch.FloatTensor): -> shape (batch_size, action_dim)
            rewards (torch.FloatTensor): -> shape (batch_size, 1)
            next_states (torch.FloatTensor): -> shape (batch_size, state_dim)
            dones (torch.FloatTensor): -> shape (batch_size, 1)
        """

        batch_size = min(batch_size, len(self.buffer))

        batch = random.sample(self.buffer, batch_size)

        return self.format_batch(batch)


    def sample_recent(self, batch_size: int, recency_factor = 1):

        """
        Samples recent-experience-enhanced batch.

        Args:
            batch_size (int): Number of transitions to sample.
            recency_factor (float): Factor to weight recent experiences more heavily.

        Returns:
            states (torch.FloatTensor): -> shape (batch_size, state_dim)
            actions (torch.FloatTensor): -> shape (batch_size, action_dim)
            rewards (torch.FloatTensor): -> shape (batch_size, 1)
            next_states (torch.FloatTensor): -> shape (batch_size, state_dim)
            dones (torch.FloatTensor): -> shape (batch_size, 1)
        """

        batch_size = min(batch_size, len(self.buffer))
        
        N = len(self.buffer)
        probs = np.exp(recency_factor*(np.linspace(-1, 0, N)))
        probs /= np.sum(probs)
        indices = np.random.choice(np.arange(N), size=batch_size, p=probs)

        batch = [self.buffer[i] for i in indices]
        
        return self.format_batch(batch)


    def _compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray,
                        threshold: float = 0.05) -> float:
        """
        Compute sparse reward based on goal distance.

        Args:
            achieved_goal (np.ndarray): The goal the agent actually achieved.
            desired_goal (np.ndarray): The intended goal.
            threshold (float): Distance threshold for success.

        Returns:
            Calculates the distance from the desired goal.
            float: 0.0 if success (within threshold), else -1.0.
            
        """

        if self.env is not None and self.use_env_reward_fn:
            reward, done = self.env._compute_reward(gcrl_reward = True, achieved_goal = achieved_goal, goal = desired_goal, info=None)
            return reward, done
        

        # Compute the distance between achieved and desired goals
        # and return 0.0 if within threshold, else -1.0.
        distance = np.linalg.norm(achieved_goal - desired_goal)
        modified_reward = -distance  # Negative distance as reward
        
        done = distance < threshold

        return modified_reward, done
    
   
    def format_batch(self, batch):

        """
        Formats a batch of transitions into tensors.
        """

        states, actions, rewards, next_states, dones = zip(*batch)

        
        return (
            self._fmt_size(states),         # -> (batch_size, state_dim + goal_dim)
            self._fmt_size(actions),        # -> (batch_size, action_dim)
            self._fmt_size(rewards),  # -> (batch_size, 1)
            self._fmt_size(next_states),    # -> (batch_size, state_dim)
            self._fmt_size(dones)           # -> (batch_size, 1)
        )           
     



    @staticmethod   
    def _fmt_size(x):
        
        """
        Formats input to ensure correct tensor dimensions.
        """

        x = torch.FloatTensor(x)

        if x.dim() == 3 and x.size(1) == 1:  # If x is 3D with second dimension 1, squeeze it to 2D
            x = x.squeeze(1)
        
        if x.dim() == 1: # If x is 1D, make it 2D
            x = x.unsqueeze(1)

        return x
        

    def __len__(self):

        return len(self.buffer)
    