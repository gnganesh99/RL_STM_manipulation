
"""
Programs to compute reward functions

@author: Ganesh Narasimha
"""

from stm_utils import Sxm_Image
from expt_utils import distance, euclidean_distance
from Manipulation_coords import get_manipulation_coords
import numpy as np
import pandas as pd






def get_state_reward_vector(corrected_initial, prev_inital, prev_final, d0_vec, X_current, X_target, obj_idx, tolerance = 0.05):

    """
    Compute reward vector for all objects based on their current positions relative to target positions.
    I/p:
        corrected_initial: corrected initial position of the manipulated object after action
        prev_inital: previous initial positions of all objects
        prev_final: previous final positions of all objects
        d0_vec: initial distances of all objects from their target positions
        X_current: current positions of all objects
        X_target: target positions of all objects
        obj_idx: index of the manipulated object
        tolerance: distance threshold to consider an object as successfully manipulated, default is 0.05

    O/p:
        reward_vector: array of rewards for each object. reward is in range [0, 10]
        delta_d: change in distance for the manipulated object (obj_idx), with respect to its previous distance
        complete_vector: array indicating if each object has been successfully manipulated
        d_space: current distance of the manipulated object (obj_idx) from its target position
    """

    
    _, d_current_vector = euclidean_distance(X_current[:, 0:2], X_target)

    n_objects = X_current.shape[0]

    complete_vector =  []
    reward_vector = []


    for i in range(n_objects):

        if i == obj_idx:
            d_prev, _ =  euclidean_distance(prev_inital[:, 0:2], prev_final)
            d_current, _ = euclidean_distance(corrected_initial, X_target[i])
            d_space = d_current
            
            delta_d = d_prev - d_current

        else:

            d_current = d_current_vector[i]


            
        reward = (1 -  d_current/(d0_vec[i]))

        done = False
        if d_current < tolerance: # d_current is within 10 % of the scan_frame distance, if tolerance = 0.1
        
            done = True
            reward = 10

        reward_vector.append(reward)
        complete_vector.append(done)

    reward_vector = np.asarray(reward_vector)

    return reward_vector, delta_d, complete_vector, d_space






def get_state_reward(current_initial, current_final, d_initial, prev_initial = None, prev_final = None, tolerance = 0.03):

    """
    Compute reward based on the distance of the manipulated object from its target position. This is for a single object manipulation.
    I/p:
        current_initial: current initial position of the manipulated object after action
        current_final: current final position of the manipulated object
        d_initial: initial distance of the manipulated object from its target position
        prev_initial: previous initial position of the manipulated object, optional
        prev_final: previous final position of the manipulated object, optional
        tolerance: distance threshold to consider the object as successfully manipulated, default is 0.05
    O/p:
        reward: computed reward. reward is in range[0, 10]
        delta_d: change in distance with respect to previous distance
        done: boolean indicating if the object has been successfully manipulated
    """

    
    if prev_initial is not None or prev_final is not None:
        d_prev, _ =  euclidean_distance(prev_initial[:, 0:2], prev_final)
    else:
        d_prev = d_initial

    d_current, _ = euclidean_distance(current_initial, current_final)

    delta_d = d_prev - d_current

    reward = (1 -  d_current/d_initial)

    done = False

    if d_current < tolerance: # d_current is within 10 % of the scan_frame distance, if tolerance = 0.1
    
        done = True
        reward = 10

    return reward, delta_d, done




def get_displacement_reward(current_initial, current_final, prev_inital, prev_final, d_initial, tolerance = 0.05):

    """
    Compute reward based on displacement improvement of the manipulated object.

    I/p:
        current_initial: current initial position of the manipulated object after action
        current_final: current final position of the manipulated object
        prev_inital: previous initial position of the manipulated object
        prev_final: previous final position of the manipulated object
        d_initial: initial distance of the manipulated object from its target position
        tolerance: distance threshold to consider the object as successfully manipulated, default is 0.05
    O/p:
        reward: computed reward, range[0, 10]
        delta_d: change in distance with respect to previous distance
        done: boolean indicating if the object has been successfully manipulated
    """

    d_prev, _ =  euclidean_distance(prev_inital[:, 0:2], prev_final)
    d_current, _ = euclidean_distance(current_initial, current_final)

    delta_d = d_prev - d_current

    reward = 10 * delta_d/d_initial

    done = False

    if reward > 0 and d_current < tolerance: # d_current is within 10 % of the scan_frame distance, if tolerance = 0.1
    
        done = True

    return reward, delta_d, done




# Reward based on distance.
def euclidean_reward(x_current, x_target, frame = 2):
    x = np.array([x_current[0], x_target[0]])
    y = np.array([x_current[1], x_target[1]])

    d_sq = ((y[1]-y[0])**2) + ((x[1]-x[0])**2)
    d = d_sq**0.5

    reward = 1 - d/frame
    
    return reward



































































# def get_image_reward(sxm_file, X_target):
    
#     asssigned_coords = manipulation_coords(sxm_file, X_target)
    
#     X_current =  asssigned_coords[0]
#     X_target = asssigned_coords[1]
           
#     num = X_current.shape[0]
    
#     sxm_file_directory = r"C:\Users\Administrator\Py_Scripts_ganesh\from laptop\1_CO on Cu images"
#     #sxm_file  = "Cu111_0234.sxm"
#     file_name = sxm_file.split('.')[-2]
#     file_path  = sxm_file_directory +'/'+ str(sxm_file)
#     scan = Sxm_Image(file_path)
    
#     scan_frame = scan.frame
    
#     reward = 0
#     reward_arr = []
    
#     for index in range(num):
        
#         p_curr = X_current[index]
        
#         p_target = X_target[index]
        
#         dis = distance(p_curr, p_target)
        
#         reward_i = 1  - dis/(2*scan_frame)
        
#         reward_arr.append(reward_i)
#         reward += reward_i
        
        


        
#     return reward_arr, reward




def get_reward_array(X_current, X_target, scan_frame):
     
    X_current = np.asarray(X_current)
    X_target = np.asarray(X_target)
         
    num = X_current.shape[0]

    
    reward = 0
    reward_arr = []
    
    for index in range(num):
        
        p_curr = X_current[index]
        
        p_target = X_target[index]
        
        dis = distance(p_curr, p_target)
        
        reward_i = 1  - dis/(2*scan_frame)
        
        reward_arr.append(reward_i)
        reward += reward_i
        
                
    return reward_arr



def get_net_reward(X_current, X_target, scan_frame):
    
    X_current = np.asarray(X_current)
    X_target = np.asarray(X_target)
    
           
    num = X_current.shape[0]
    
    
    reward = 0
    reward_arr = []
    
    for index in range(num):
        
        p_curr = X_current[index]
        
        p_target = X_target[index]
        
        dis = distance(p_curr, p_target)
        
        reward_i = 1  - dis/(2*scan_frame)
        
        reward_arr.append(reward_i)
        reward += reward_i
        
    #reward = 25
        
    return reward




def get_reward(p_curr, p_target, scan_frame):
    
    dis = distance(p_curr, p_target)
    
    reward_i = 1  - dis/(2*scan_frame)
        
    return reward_i


def dummy(a, b, c):
    
    return 25



