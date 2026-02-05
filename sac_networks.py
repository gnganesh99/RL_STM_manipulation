"""
Soft Actor-Critic (SAC) implementation in PyTorch - Networks and functions

@author: Ganesh Narasimha
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import deque
import random
from gym import Env


LOG_STD_MIN = -20
LOG_STD_MAX = 2

class Actor(nn.Module):
    """
    The Actor network is a policy approximator that takes in the state as input and outputs the
    action to be taken. It uses a Gaussian distribution to model the action space, and the mean and
    standard deviation of the Gaussian are learned by the network. The action is then sampled from
    the Gaussian distribution and squashed using the Tanh function to ensure that the action lies
    within the valid range. The log probability of the action is also computed for use in the
    policy gradient update.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, deterministic:bool = False):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        # Sample action

        if deterministic:  # use this for eval. Gives the mean of the action
            action = torch.tanh(mean)
            log_prob = None
        else:        
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()  # reparameterization trick, differentiable sample (vs normal.sample())
            action = torch.tanh(z)
    
            # Log prob correction for Tanh squashing
            log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
        mean_action = torch.tanh(mean)
        
        return action, log_prob, mean_action

class Critic(nn.Module):
    """
    The Critic network is a Q-function approximator that takes in the state and action as input
    and outputs the Q-value. It consists of two identical networks (q1 and q2) to mitigate the
    overestimation bias in Q-learning. The two networks are trained independently, and the minimum
    of the two Q-values is used to update the policy.

    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)

        q_val = self.q(x)
        return q_val.view(-1, 1)

class ValueNetwork(nn.Module):
    """
    The Value network is a V-function approximator that takes in the state as input and outputs
    the value of the state. It is used to compute the advantage function, which is the difference
    between the Q-value and the value of the state. The advantage function is used to update the
    policy in the actor-critic algorithm.
    """

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.v(state)



      
    
    

def soft_update(target, source, tau):
    """
    Soft update of the target network parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau*source_param.data + (1 - tau)*target_param.data)  

    return target
              

def hard_update(target, source):
    """
    Hard update of the target network parameters.
    θ_target = θ_local
    """
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data) 

    return target



def rescale_action(action, low, high):
    """
    Rescale the action from the range [-1, 1] to the range [low, high].
    """
    return low + (high - low) * (action + 1) / 2




def state_goal_combination(state, goal) -> np.ndarray:
    """
    Combine state and goal into a single vector.

    Args:
        state (np.ndarray): Current state.
        goal (np.ndarray): Goal for this transition.

    Returns:
        np.ndarray: Combined state-goal vector.
    """
    state = np.asarray(state).reshape(-1)
    goal = np.asarray(goal).reshape(-1)
    
    return np.concatenate((state, goal), axis=-1)


def rescale_array(old_range, old_element, new_range):
    """
    Rescale the elements of old_element from old_range to new_range.
    
    """

    new_array = []
    for i in range(len(old_element)):

        r_old = old_range[i]
        r_new = new_range[i]
        
        #Clip element in the given range
        new_element = np.clip(old_element[i], r_old[0], r_old[1])

        #First rescale in range(0-1)
        new_element = (new_element - (np.min(r_old)))/ (np.max(r_old) - np.min(r_old))

        #Rescale to new range 
        new_element = new_element*(np.max(r_new) - np.min(r_new)) + np.min(r_new)
        
        #Second clipping as a precaution
        new_element = np.clip(new_element, r_new[0], r_new[1])
        new_array.append([new_element])

    new_array = np.array(new_array).reshape(-1)

    return new_array


class SACAgent:

    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005, alpha=0.2,
                 target_entropy=None, automatic_entropy_tuning=True, a_lr=1e-4, c_lr=3e-4, cr_weight_decay = 1e-5, gcrl = False, goal_dim=0,
                 huber_loss = False, device = None):

        self.gcrl = gcrl  # If goal conditioned RL            
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self._moved_to_device = False

        # If goal conditioned RL, we append the goal dimension to include the goal
        if gcrl:
            state_dim += goal_dim


        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)

        # Initialize target value network. Copy weights from value network
        self.target_value = ValueNetwork(state_dim)
        self.target_value.load_state_dict(self.value.state_dict())

        self._ensure_device()  # move networks to device


        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=c_lr, weight_decay = cr_weight_decay)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=c_lr, weight_decay = cr_weight_decay)
        self.value_optimizer   = torch.optim.Adam(self.value.parameters(), lr=c_lr, weight_decay = cr_weight_decay)

        self.gamma = gamma
        self.tau = tau

        # Alpha is the temperature parameter that controls the entropy term.
        self.alpha = alpha

        ## If automatic_entropy_tuning is True, we will learn the temperature parameter
        self.automatic_entropy_tuning = automatic_entropy_tuning

        if self.automatic_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -action_dim  # heuristic
            else:
                self.target_entropy = target_entropy

            # log_alpha is the log of the temperature parameter. We optimize log_alpha instead of alpha
            # requires_grad=True means we want to optimize this parameter using gradient descent            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device) # log_alpha is initialized to 0, which corresponds to alpha = 1    
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=c_lr)

        self.huber_loss = huber_loss

    def _ensure_device(self):
        """Ensure that the networks are moved to the correct device."""
        if self._moved_to_device:
            return
        move_agent_to_device(self, self.device)
        self._moved_to_device = True
        
    
    def select_action(self, state, epsilon = 0, goal = None, deterministic:bool = True):
        """
        Select an action based on the current state. If gcrl is enabled, the goal is also passed.
        """
        
        if self.gcrl:
            state = state_goal_combination(state, goal)
            
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)


        if np.random.rand() < epsilon:
            # Epsilon-greedy action selection
            action = np.random.uniform(-1, 1, size=self.action_dim).astype(np.float32)

        else:
            with torch.no_grad():
                action, _, _ = self.actor(state, deterministic=deterministic)
                action = action.squeeze(0).detach().cpu().numpy().ravel()
                
        action = np.clip(action, -1, 1)

        return action
    
    def select_action_1(self, state, epsilon = 0, goal = None, deterministic:bool = True):
        """
        Use this variant while using states from buffer where state = state + goal.
        Useful while inputting states from her_buffer.
        """
        
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)


        if np.random.rand() < epsilon:
            # Epsilon-greedy action selection
            action = np.random.uniform(-1, 1, size=self.action_dim).astype(np.float32)

        else:
            with torch.no_grad():
                action, _, _ = self.actor(state, deterministic=deterministic)
                action = action.squeeze(0).detach().cpu().numpy().ravel()
        
        action = np.clip(action, -1, 1)

        return action
    
   
    def update(self, replay_buffer, batch_size=256, recency_factor = 0):
        """Update the networks based on the replay buffer."""

        self._ensure_device()
        
        if recency_factor > 0:
            state, action, reward, next_state, done = replay_buffer.sample_recent(batch_size, recency_factor = recency_factor)
        else:
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    
        state, action, reward, next_state, done = state.to(self.device), action.to(self.device), reward.to(self.device), next_state.to(self.device), done.to(self.device)

        #Determine the target Q-value for next_state
        with torch.no_grad():
            target_value = self.target_value(next_state)

            if self.gcrl:
                target_q_value = reward + self.gamma * target_value    
            else: 
                target_q_value = reward + (1 - done) * self.gamma * target_value        

        # Critic loss - optimizes only the 
        q1, q2 = self.critic1(state, action), self.critic2(state, action)


        loss_fn = F.smooth_l1_loss if self.huber_loss else F.mse_loss

        critic1_loss = loss_fn(q1, target_q_value)
        critic2_loss = loss_fn(q2, target_q_value)
        critic_loss  = critic1_loss + critic2_loss

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(         # clip gradients to prevent exploding gradients
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0
        )
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        
        td_error = ((q1 - target_q_value).detach().abs().mean() + (q2 - target_q_value).detach().abs().mean())/2



        # Value loss
        with torch.no_grad():
            new_action, log_prob, _ = self.actor(state)
            q1_new, q2_new = self.critic1(state, new_action), self.critic2(state, new_action) # Q-values for the new action at the same state
            q_min = torch.min(q1_new, q2_new)
            target_v = q_min - self.alpha * log_prob

        v = self.value(state)
        value_loss = loss_fn(v, target_v)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


        # Actor loss - update actor network 
        new_action, log_prob, _ = self.actor(state)
        # Compute the Q-values for the new action at the same same.
        # Here we require the critic to be differentiable, while at the same time we do not want to update the critic parameters.(no critic_optimizer.step())     
        q1_new, q2_new = self.critic1(state, new_action) , self.critic2(state, new_action)
        q_min = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_min).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # Entropy temperature update
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
           

        # Soft update
        self.target_value = soft_update(self.target_value, self.value, self.tau)

        training_log = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'value_loss': value_loss.item(),
            'alpha': self.alpha.item(),
            'alpha_loss': alpha_loss.item() if self.automatic_entropy_tuning else None,
            'q1': q1.mean().item(),
            'q2': q2.mean().item(),
            'q_min': q_min.mean().item(),
            'target_value': target_value.mean().item(),
            'td_error': td_error.item() if 'td_error' in locals() else None
        }

        return training_log


    def update_cql(self, replay_buffer, batch_size=256, alpha_cql=1.0, recency_factor = 0):
        """Update the networks based on the replay buffer using CQL."""

        self._ensure_device()
        
        if recency_factor > 0:
            state, action, reward, next_state, done = replay_buffer.sample_recent(batch_size, recency_factor = recency_factor)
        else:
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state, action, reward, next_state, done = state.to(self.device), action.to(self.device), reward.to(self.device), next_state.to(self.device), done.to(self.device)
    

        #Determine the target Q-value for next_state
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor(next_state)
            target_value = self.target_value(next_state)    
            if self.gcrl:
                target_q_value = reward + self.gamma * target_value
            else:        
                target_q_value = reward + (1 - done) * self.gamma * target_value        

        # Critic loss - optimizes only the 
        q1, q2 = self.critic1(state, action), self.critic2(state, action)

        loss_fn = F.smooth_l1_loss if self.huber_loss else F.mse_loss

        critic1_loss = loss_fn(q1, target_q_value)
        critic2_loss = loss_fn(q2, target_q_value)
        critic_loss  = critic1_loss + critic2_loss 

        td_error = ((q1 - target_q_value).detach().abs().mean() + (q2 - target_q_value).detach().abs().mean())/2


        # ------------    Add CQL conservative loss    ---------------
        

        # Sample random actions uniformly or from actor
        batch_size, action_dim = action.shape
        random_actions = torch.rand((batch_size, 10, action_dim), device=state.device) * 2 - 1  # Sample uniformly in range [-1, 1]

        # Repeat states for all sampled actions
        repeated_states = state.unsqueeze(1).repeat(1, 10, 1).reshape(-1, state.shape[-1])
        random_actions = random_actions.reshape(-1, action_dim)

        # Compute Q-values on random actions
        q1_rand, q2_rand = self.critic1(repeated_states, random_actions), self.critic2(repeated_states, random_actions)
        q1_rand = q1_rand.view(batch_size, 10)
        q2_rand = q2_rand.view(batch_size, 10)

        # Log-sum-exp for conservative penalty
        cql_q1 = torch.logsumexp(q1_rand, dim=1).mean()
        cql_q2 = torch.logsumexp(q2_rand, dim=1).mean()

        # Q-values on dataset actions (should be high)
        q1_data = q1
        q2_data = q2

        cql_penalty = (cql_q1 - q1_data.mean()) + (cql_q2 - q2_data.mean())

        # Add CQL loss to critic loss
        critic_loss += alpha_cql * cql_penalty


        # Update critic networks
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(         # clip gradients to prevent exploding gradients
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0
        )
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()


        # Value loss
        with torch.no_grad():
            new_action, log_prob, _ = self.actor(state)
            q1_new, q2_new = self.critic1(state, new_action), self.critic2(state, new_action) # Q-values for the new action at the same state
            q_min = torch.min(q1_new, q2_new)
            target_v = q_min - self.alpha * log_prob

        v = self.value(state)
        value_loss = loss_fn(v, target_v)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


        # Actor loss - update actor network 
        new_action, log_prob, _ = self.actor(state)
        # Compute the Q-values for the new action at the same same.
        # Here we require the critic to be differentiable, while at the same time we do not want to update the critic parameters.(no critic_optimizer.step())     
        q1_new, q2_new = self.critic1(state, new_action) , self.critic2(state, new_action)
        q_min = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_min).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # Entropy temperature update
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
           

        # Soft update
        self.target_value = soft_update(self.target_value, self.value, self.tau)


        training_log = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'value_loss': value_loss.item(),
            'alpha': self.alpha.item(),
            'alpha_loss': alpha_loss.item() if self.automatic_entropy_tuning else None,
            'q1': q1.mean().item(),
            'q2': q2.mean().item(),
            'q_min': q_min.mean().item(),
            'target_value': target_value.mean().item(),
            'td_error': td_error.mean().item()
        }

        return training_log


def move_agent_to_device(agent, device=None):
    """Move agent submodules to device and set agent.device."""
    if agent is None:
        return agent
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If the agent itself implements .to()
    if hasattr(agent, "to"):
        try:
            agent.to(device)
        except Exception:
            pass

    # Typical SAC bits
    for name in [
        "actor","critic","critic1","critic2",
        "target_actor","target_critic","target_critic1","target_critic2",
        "value","target_value","value_net","target_value_net",
        "q_net","q1_net","q2_net","policy"
    ]:
        m = getattr(agent, name, None)
        if m is not None and hasattr(m, "to"):
            try:
                m.to(device)
            except Exception:
                pass

    setattr(agent, "device", device)
    return agent

