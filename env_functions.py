"""
Helper functions for the RL manipulation environment - to connect the environment with the rest of the code/routines.

@author: Ganesh Narasimha

"""



import gym

from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import time
from IPython.display import clear_output
import os
import numpy as np


from Manipulation_coords import get_manipulation_coords, get_manipulation_coords_MO
from expt_utils import transform_coord_array, translation_angle, write_log, get_latest_file, get_next_file_name, copy_files, get_sxm_filenames
from expt_utils import arr_to_linestring, euclidean_distance, saved_next_sxmfilenames, distance, path_exists, read_expt_log

from detect_molecules import get_latest_image
from stm_manipulation_experiments import manipulation_and_scan_tcp, rescan_tcp, do_manipulation_LV, rescan_LV, rescan_LV_dummy
from display_results import show_iter_image_results, show_iter_results
from drift_estimation import drift_correction_far_labels
from get_reward import get_displacement_reward, get_state_reward
from get_target import random_target, custom_target, compute_coordinates
from model_buffer_saving_functions import save_dict_to_h5

"""
Programs for the online RL manipulation.
"""

def get_state(expt_basename, expt_dir, X_target, max_len = False):

    sxm_file_name, _, _ =  saved_next_sxmfilenames(expt_dir, expt_basename)

    sxm_file_path = os.path.join(expt_dir, sxm_file_name)
    
    img_dir = os.path.join(expt_dir, 'images')

    # Convert the targets to a 2D array
    X_target = np.array([X_target])

    labels_exist = False
    while labels_exist == False:

        initial_coords, final_coords, scan_params, corrected_position, scan_drift, labels_exist = get_manipulation_coords(sxm_file_path, img_dir, X_target, max_len = max_len)


        if labels_exist == False:

            #rescan_LV(expt_dir)
            #rescan_tcp()
            rescan_LV_dummy(expt_dir)

        else:

            break
        
    

    #Add angle_info
    initial_states = []

    for i in range(initial_coords.shape[0]):

        theta = translation_angle(initial_coords[i], final_coords[i])
        
        # The state representaion has the shape (5,)
        
        initial_states.append([initial_coords[i][0], initial_coords[i][1], final_coords[i][0], final_coords[i][1], theta])

    initial_states = np.asarray(initial_states)

    #print(f'atget_states: scan_params = {scan_params}')
   


    return initial_states, final_coords, scan_params, corrected_position, scan_drift


def get_state_MO(expt_basename, expt_dir, X_target, iteration, obj_idx, label_margin = 0.0, anchor_tuple = None, detect_dict = None):    

    sxm_file_name, _, _ =  saved_next_sxmfilenames(expt_dir, expt_basename)
    
    print("sxm_file_name:", sxm_file_name)

    sxm_file_path = os.path.join(expt_dir, sxm_file_name)
    
    img_dir = os.path.join(expt_dir, 'images')

    X_target = np.array(X_target)
    

    labels_exist = False

    while labels_exist == False:

        initial_coords, final_coords, scan_params, corrected_position, scan_drift, labels_exist, avg_label_width, all_labels = get_manipulation_coords_MO(sxm_file_path, img_dir, X_target, iteration, obj_idx, label_margin = label_margin, anchor_tuple=anchor_tuple, detect_dict=detect_dict)

        if labels_exist == False:

            #rescan_LV(expt_dir)
            rescan_tcp()
            #rescan_LV_dummy(expt_dir)

        else:

            break
        
    

    #Add angle_info
    initial_states = []

    for i in range(initial_coords.shape[0]):

        theta = translation_angle(initial_coords[i], final_coords[i])
        
        # The state representaion has the shape (5,)
               
        
        initial_x = initial_coords[i][0]
        initial_y = initial_coords[i][1]
        
        initial_states.append([initial_x, initial_y, final_coords[i][0], final_coords[i][1], theta])

    initial_states = np.asarray(initial_states)

    #print(f'atget_states: scan_params = {scan_params}')

    current_file_name = sxm_file_name.split('.')[0]



    return initial_states, final_coords, scan_params, corrected_position, scan_drift, avg_label_width, all_labels, current_file_name





def rescale_action_params(action, old_range, new_range):

    old_range = np.asarray(old_range)
    new_range = np.array(new_range)

    #Convert to 1D array. else wrong predictions 
    action = np.ravel(np.asarray(action))
    
    rescaled_action = rescale_array(old_range, action, new_range)

    rescaled_action = np.ravel(rescaled_action)

    return rescaled_action




def save_offset_params(offset_vals, expt_dir, expt_name, basename, target = True):
    
    log_dir = os.path.join(expt_dir, 'expt_log', expt_name)
    os.makedirs(log_dir, exist_ok = True)       
    
    sxm_file_name,_, _ =  saved_next_sxmfilenames(expt_dir, basename)
    sxm_file_name = sxm_file_name.split('.')[0]

    if target:
        write_log(os.path.join(log_dir, 'offset_vals_target.txt'), sxm_file_name, arr_to_linestring(offset_vals))
    else:
        write_log(os.path.join(log_dir, 'offset_vals_initial.txt'), sxm_file_name, arr_to_linestring(offset_vals))


def log_parameters(expt_name, basename, expt_dir, state_vars, corrected_coords, action_params, drift, **kw):

    sxm_file_name,_, _ =  saved_next_sxmfilenames(expt_dir, basename)
    sxm_file_name = sxm_file_name.split('.')[0]

    log_dir = os.path.join(expt_dir, 'expt_log', expt_name)
    os.makedirs(log_dir, exist_ok = True)       
    
    
    
    write_log(os.path.join(log_dir, 'obs.txt'), sxm_file_name, arr_to_linestring(state_vars))

    write_log(os.path.join(log_dir, 'drift_corrected_obs.txt'), sxm_file_name, arr_to_linestring(corrected_coords))

    write_log(os.path.join(log_dir,'action_params.txt'), sxm_file_name, arr_to_linestring(action_params))
    
    write_log(os.path.join(log_dir,'drift_log.txt'), sxm_file_name, arr_to_linestring(drift))

    X_current = kw.get("X_current")

    if X_current is not None and X_current.any():
        write_log(os.path.join(log_dir,'global_states.txt'), sxm_file_name, arr_to_linestring(X_current)) 




def log_reward(expt_name, basename, expt_dir, d_initial, reward, reward_wdrift, disp, disp_wdrift, **kw):

    sxm_file_name,_, _ =  saved_next_sxmfilenames(expt_dir, basename)
    sxm_file_name = sxm_file_name.split('.')[0]

    log_dir = os.path.join(expt_dir, 'expt_log', expt_name)
    os.makedirs(log_dir, exist_ok = True)     
    
    enter_string = str(reward)+'\t'+str(reward_wdrift)+'\t'+str(disp)+'\t'+str(disp_wdrift)+'\t'+str(d_initial)
    write_log(os.path.join(log_dir,'reward_dinitial.txt'), sxm_file_name, enter_string) 

    reward_vec = kw.get("reward_vec")
    if reward_vec is not None and reward_vec.any():
        write_log(os.path.join(log_dir,'reward_vector.txt'), sxm_file_name, arr_to_linestring(reward_vec)) 


def log_tiprec_buffer(buffer, expt_name, basename, expt_dir):
    
    sxm_file_name,_, _ =  saved_next_sxmfilenames(expt_dir, basename)
    sxm_file_name = sxm_file_name.split('.')[0]
    
    log_dir = os.path.join(expt_dir, 'expt_log', expt_name)
    os.makedirs(log_dir, exist_ok = True)  


    channels = np.asarray(buffer.get("channel_indices"))
    data = np.asarray(buffer.get("data"))
    z_ind = np.squeeze(np.where(channels == 30))  # signal 30 is the Z-channel
    i_ind = np.squeeze(np.where(channels == 0))  #signal 0 is the i channel
    z_data = data[z_ind]
    i_data = data[i_ind]
    
    tip_buffer = {"i": i_data, "z": z_data}
    
    save_dict = {
        sxm_file_name: tip_buffer
        }
    
    buffer_file = os.path.join(log_dir, 'tiprec_buffer.h5')
    
    if not os.path.exists(buffer_file):
        save_dict_to_h5(save_dict, buffer_file, mode = 'w')
    else:
        save_dict_to_h5(save_dict, buffer_file, mode = 'a')



def rescale_Tuple(old_range, old_element, new_range):

    new_tuple = []
    for i in range(len(old_element)):

        r_old = old_range[i]
        r_new = new_range[i]
        
        #Clip element in the given range
        new_element = np.clip(old_element[i], r_old[0], r_old[1])

        #First rescale in range(0-1)
        new_element = (new_element - (np.min(r_old)))/ (np.max(r_old) - np.min(r_old))

        #Rescale to new range 
        new_element = new_element*(np.max(r_new) - np.min(r_new)) + np.min(r_new)
        new_tuple.append([new_element])

    return tuple(new_tuple)




def rescale_array(old_range, old_element, new_range):

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

    return new_array




# Reward based on distance.
def euclidean_reward(x_current, x_target, frame = 2):
    x = np.array([x_current[0], x_target[0]])
    y = np.array([x_current[1], x_target[1]])

    d_sq = ((y[1]-y[0])**2) + ((x[1]-x[0])**2)
    d = d_sq**0.5

    reward = 1 - d/frame
    
    return reward




def random_action(bias_range, setpoint_range, speed_range):
    
    bias_range = np.asarray(bias_range)
    setpoint_range =  np.asarray(setpoint_range)
    speed_range = np.asarray(speed_range)
    
    bias =  random.uniform(np.min(bias_range), np.max(bias_range))
    setpoint = random.uniform(np.min(setpoint_range), np.max(setpoint_range))
    speed = random.uniform(np.min(speed_range), np.max(speed_range))
                          
    ret_arr = []
    
    ret_arr.append(bias)
    ret_arr.append(setpoint)
    ret_arr.append(speed)
    
    return ret_arr




def point_within_margin(point, margin = 0.1):
    
    within_margin = False

    x_f, y_f = np.ravel(np.asarray(point))
    
    if x_f > margin and x_f < 1-margin:
        
        if y_f > margin and y_f < 1-margin:
            
            within_margin = True
            
    return within_margin
            




def get_nearest_coords(state, labels, n_near = 3, frame_length = 10E-9, cutoff_distance = 1E-9):


    """
    Outputs the relative coordinates of the nearest n_near points in labels to the state.
    Args:
        state: np.array of shape (1, 5) representing the current state (x, y, target_x, target_y, angle).
        labels: np.array of shape (N, 2) representing the coordinates of detected points.
        n_near: int, number of nearest points to find.
        frame_length: float, length of the frame in meters for distance normalization.

    Returns:
        np.array of shape (1, 6) representing the relative coordinates of 3 nearest points normalized to [0, 1].
        if labels is None or empty, returns random coordinates 0 or 1.
        if n_near<3, fills the remaining coordinates with random 0 or 1.
    """


    if labels is None or len(labels) == 0:

        near_coords = []
        for i in range(6):
            near_coords.append(random.choice([0, 1]))
        
        return np.array(near_coords).reshape(1, -1)


    max_distance = cutoff_distance/frame_length

    x_initial = state[0, 0:2]

    relative_positions = (labels - x_initial)

    distances = np.linalg.norm(relative_positions, axis=1)


    nearest_indices = np.argsort(distances)

    relative_positions = np.clip(np.asarray(relative_positions/max_distance), -1, 1)
    relative_positions = (relative_positions + 1)/2  #normalize to 0-1

    nearest_relative_coords = []

    for i in range(1, 4):  # Exclude the first one as it is the point itself
        if i > len(nearest_indices)-1 or i > n_near:
            nearest_relative_coords.append(random.choice([0, 1])); nearest_relative_coords.append(random.choice([0, 1]))
            
        else:
            nearest_relative_coords.extend(relative_positions[nearest_indices[i]].tolist())


    return np.array(nearest_relative_coords).reshape(1, -1)




























# def sxm_filename_LV(basename, sxm_dir):
    
#     filename = get_latest_file(sxm_dir)

#     return filename


# def do_manipulation_LV(initial_coords, final_coords, default_action_params, action_params, sxm_dir):

#     filename = get_latest_file(sxm_dir) 
    
#     origin_dir = r"C:\Users\ggn\Desktop\A_Research\Atom Manipulation\data\routine1"
#     next_filepath = get_next_file_name(filename, origin_dir)

#     copy_files(next_filepath, sxm_dir)

#     time.sleep(5)

#     return True

# def rescan_LV(sxm_dir):
#     filename = get_latest_file(sxm_dir) 
    
#     origin_dir = r"C:\Users\ggn\Desktop\A_Research\Atom Manipulation\data\routine1"
#     next_filepath = get_next_file_name(filename, origin_dir)

#     copy_files(next_filepath, sxm_dir)

#     time.sleep(5)

#     return True

