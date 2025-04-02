
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
from expt_utils import transform_coord_array, translation_angle, write_log, get_latest_file, get_next_file_name, copy_files, get_sxm_filenames, arr_to_linestring, euclidean_distance, saved_next_sxmfilenames
from expt_utils import arr_to_linestring, euclidean_distance, saved_next_sxmfilenames, distance, path_exists, read_expt_log

from detect_molecules import get_latest_image
from stm_manipulation_experiments import manipulation_and_scan_tcp, rescan_tcp, do_manipulation_LV, rescan_LV
from display_results import show_iter_image_results, show_iter_results
from drift_estimation import drift_correction_far_labels
from get_reward import get_displacement_reward, get_state_reward
from get_target import compute_coordinates

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

        initial_coords, final_coords, scan_params, corrected_position, scan_drift, labels_exist, avg_label_width = get_manipulation_coords(sxm_file_path, img_dir, X_target, max_len = max_len)


        if labels_exist == False:

            #rescan_LV(expt_dir)
            rescan_tcp()

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
   


    return initial_states, final_coords, scan_params, corrected_position, scan_drift, avg_label_width



def get_state_MO(expt_basename, expt_dir, X_target, iteration, obj_idx, label_margin = 0):

    sxm_file_name, _, _ =  saved_next_sxmfilenames(expt_dir, expt_basename)
    
    print("sxm_file_name:", sxm_file_name)

    sxm_file_path = os.path.join(expt_dir, sxm_file_name)
    
    img_dir = os.path.join(expt_dir, 'images')

    X_target = np.array(X_target)
    

    labels_exist = False

    while labels_exist == False:

        initial_coords, final_coords, scan_params, corrected_position, scan_drift, labels_exist, avg_label_width = get_manipulation_coords_MO(sxm_file_path, img_dir, X_target, iteration, obj_idx, label_margin=label_margin)

        if labels_exist == False:

            #rescan_LV(expt_dir)
            rescan_tcp()

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


    return initial_states, final_coords, scan_params, corrected_position, scan_drift, avg_label_width






def get_manipulation_offset(label_width, theta):
    
    offset = 0.18*label_width
    
    if (theta >= 0 and theta < 0.0625) or (theta >= 0.9375 and theta <=1):
        
        shift = [-offset, 0]
        
    elif theta >= 0.0625 and theta < 0.1875:
        shift = [-offset, -offset]
        
    elif theta >= 0.1875 and theta < 0.3125:
        shift = [0, -offset]
        
    elif theta >= 0.3125 and theta < 0.4375:
        shift = [offset, -offset]
        
    elif theta >= 0.4375 and theta < 0.5625:
        shift = [offset, 0]
        
    elif theta >= 0.5625 and theta < 0.6875:
        shift = [offset, offset]
        
    elif theta >= 0.6875 and theta < 0.8125:
        shift = [0, offset]
        
    elif theta >= 0.8125 and theta < 0.9375:
        shift = [-offset, offset]
        
    else:
        shift = [0, 0]
      
    extra_offset = [0*label_width, 0]
    
    shift = [shift[0] +extra_offset[0], shift[1]+extra_offset[1]]
    
    return shift
        
    

    
def compute_offset_coordinates(distance, theta, target = True):
    
    angle = theta*360
    
            
    if not target:
        angle += 180
        
    rad = np.radians(angle)

    x = distance * np.cos(rad)
    y = distance * np.sin(rad)

    return x, y





def compute_manipulation_offset(label_width, theta, delta_offset = 0.15):
    
    offset_distance = delta_offset*label_width
    offset = compute_offset_coordinates(offset_distance, theta, target=False)

    return offset





def compute_target_offset(label_width, theta, move_attempt = 0, delta_offset = 0.1):
    
    offset_distance = move_attempt*delta_offset*label_width
    offset = compute_offset_coordinates(offset_distance, theta, target = True)

    return offset




def object_stuck(current_state, expt_log_dir, prev_i = 1, threshold = 0.03):
    
    obs_path = os.path.join(expt_log_dir, 'obs.txt')

    stuck = False
    disp = 0
    if path_exists(obs_path):
        names, logs = read_expt_log(obs_path)
        
        if len(logs)>= prev_i:
            prev_state = logs[-1*prev_i][0:2]
            disp = distance(current_state, prev_state)
            if  disp < threshold:
                stuck = True
            
        else: 
            # prev_i = len(logs)
            # prev_state = logs[-1*prev_i][0:2]
            print("prev iteration {prev_i} > observations{len(logs)}")
            
        
        
        
        # print("At obj_stuck:",prev_state, disp)
        # time.sleep(2)
    return stuck, disp


def shake(x, y, move_attempt):
    
    fraction = min(0.1(move_attempt - 5), 0.5)
    
    if np.random.rand() > 0.5:
        x_offset, y_offset = -x(1-fraction), -y

    else:
        x_offset, y_offset = -x, -y(1-fraction)
        
    return x_offset, y_offset, fraction
    

def move_along_Cu_axis(current_state, final_coords, distance_fraction = 0.1, distance_fraction_shake = 1):


    initial_coords = current_state[0][0:2]
    final_coords = final_coords[0][0:2]
    manipulation_angle = current_state[0][-1]
    
    cu_angles = np.asarray([0, 60, 120, 180, 240, 300, 360])/360
    shake_angles = np.asarray([0, 90, 180, 270, 360])/360

    if distance_fraction < distance_fraction_shake:

        angle_diff = np.abs(cu_angles - manipulation_angle)
        idx = np.argmin(angle_diff)
    
        manipulation_angle = cu_angles[idx]
    
    else:
        manipulation_angle = random.choice(shake_angles)
        distance_fraction = 0.5*distance_fraction_shake

    dist = distance(initial_coords, final_coords)*distance_fraction

    offset_x, offset_y = compute_offset_coordinates(dist, manipulation_angle, target = True)
    target_coords = initial_coords + np.asarray([offset_x, offset_y])

    return target_coords, manipulation_angle    



def rescale_state_coords(coordinate_set, scan_params, expt_dir, expt_name, label_width = 0, manipulation_offset = None, current_state = None):

    coordinates = coordinate_set[:,0:2]
    manipulation_angle = coordinate_set[0][-1]
    #print(f'coordinates = {coordinates}')

    frame_size, center_offset, scan_angle = scan_params
    
    frame_rescaled_coords = []
    expt_log_dir = os.path.join(expt_dir, 'expt_log', expt_name)
    info =  None

    #Invert y-coordinate to correlate to real space
    # Transform to range 0-frame
    for coordinate in coordinates:
        norm_x, norm_y = coordinate
        
        # Add coordinate offset
        if manipulation_offset == "start_offset":
            
            x_offset, y_offset = compute_manipulation_offset(label_width, manipulation_angle, delta_offset=0.1)

        elif manipulation_offset == "end_offset":
            
            if current_state is None:
                return "Provide current_state"
            else:
                initial_stuck_point = current_state[:, 0:2][0]
                manipulation_angle = current_state[0][-1]
                
            prev_i = 1
            move_attempt = 0
            stuck = True

            while stuck:
                stuck, disp_struck = object_stuck(initial_stuck_point, expt_log_dir, prev_i = prev_i, threshold=0.03)
                if stuck:
                    move_attempt += 1
                info = "stuck:"+str(stuck)+"\tmove_attempt:"+str(move_attempt)+"\tdisp:"+ str(disp_struck)
                prev_i += 1
                
            
            x_offset, y_offset = compute_target_offset(label_width, manipulation_angle, move_attempt = move_attempt, delta_offset=0.3)
            
            # Move a bit either in vertical or horizontal direction if stuck            
            if move_attempt > 2:
                
                #x_offset, y_offset, distance_fraction = shake(norm_x, norm_y, move_attempt) 
                
                distance_fraction = min(0.3*(move_attempt - 0), 3.1)
                new_targets, new_mani_angle = move_along_Cu_axis(current_state, coordinates, distance_fraction=distance_fraction, distance_fraction_shake=3)
                    
                x_offset = new_targets[0]-norm_x
                y_offset = new_targets[1]-norm_y
                
                info = "stuck:"+str(stuck)+"\tmove_attempt:"+str(move_attempt)+"\tdisp:"+ str(disp_struck)+"\tfraction:"+str(distance_fraction)+"\tangle:"+str((1 -new_mani_angle)*360)
            
        else:
            x_offset, y_offset = 0, 0

        norm_x = norm_x + x_offset
        norm_y = norm_y + y_offset
            
        
        frame_rescaled_coords.append(np.asarray([norm_x, 1-norm_y])*frame_size)
        
    frame_rescaled_coords = np.asarray(frame_rescaled_coords)
    
    transformed_coords = transform_coord_array(frame_rescaled_coords, center_offset[0], center_offset[1], frame_size, scan_angle) 

    return transformed_coords, info




def rescale_initial_coords(coordinate_set, scan_params, expt_dir, expt_name, basename, offset_params = np.ones((2,))*-1, label_width = 0, manipulation_offset = None, current_state = None):

    coordinates = coordinate_set[:,0:2]

    manipulation_angle = coordinates[0][-1]
    
    frame_size, center_offset, scan_angle = scan_params
    
    info = {}
    frame_rescaled_coords = []
    expt_log_dir = os.path.join(expt_dir, 'expt_log', expt_name)

    
    old_offset_range = np.asarray([[-1, 1], [-1, 1]])
    new_offset_range = np.asarray([[0, label_width], np.radians([manipulation_angle*360-90, manipulation_angle*360+90])])
    offset_params = rescale_array(old_offset_range, np.ravel(offset_params), new_offset_range)
          
    

    #Invert y-coordinate to correlate to real space
    # Transform to range 0-frame
    
    for coordinate in coordinates:
        norm_x, norm_y = coordinate
        
        if manipulation_offset == "start_offset":

            offset_params = np.ravel(offset_params)
            x_offset, y_offset = compute_coordinates(offset_params[0], offset_params[1])
            info = f"Start Offset: {x_offset}, {y_offset}, offset_params = {offset_params}"
        
        elif manipulation_offset == "custom_start_offset":

            x_offset, y_offset = compute_manipulation_offset(label_width, manipulation_angle, delta_offset= 2*0.1)
            info = f" Start Offset: {x_offset}, {y_offset}"
        
        else:
            x_offset, y_offset = 0, 0

        norm_x = norm_x + x_offset
        norm_y = norm_y + y_offset
            
        
        frame_rescaled_coords.append(np.asarray([norm_x, 1-norm_y])*frame_size)
        
    frame_rescaled_coords = np.asarray(frame_rescaled_coords)
    
    offset_vals = [x_offset, y_offset]
    save_offset_params(offset_vals, expt_dir, expt_name, basename, target = False)
    
    transformed_coords = transform_coord_array(frame_rescaled_coords, center_offset[0], center_offset[1], frame_size, scan_angle) 

    return transformed_coords, info




def rescale_target_coords(coordinate_set, scan_params, expt_dir, expt_name, basename, offset_params = np.ones((2,))*-1, label_width = 0, manipulation_offset = None, current_state = None):

    coordinates = coordinate_set[:,0:2]

    manipulation_angle = current_state[0][-1]
    
    
    frame_size, center_offset, scan_angle = scan_params
    
    frame_rescaled_coords = []
    expt_log_dir = os.path.join(expt_dir, 'expt_log', expt_name)

    
    old_offset_range = np.asarray([[-1, 1], [-1, 1]])
    new_offset_range = np.asarray([[0, label_width], np.radians([0, 360])])
    offset_params = rescale_array(old_offset_range, np.ravel(offset_params), new_offset_range)
          
    

    #Invert y-coordinate to correlate to real space
    # Transform to range 0-frame

    for coordinate in coordinates:
        norm_x, norm_y = coordinate
        
        if manipulation_offset == "end_offset":

            offset_params = np.ravel(offset_params)
            x_offset, y_offset = compute_coordinates(offset_params[0], offset_params[1])
            info = f"Offset: {x_offset}, {y_offset}, offset_distance = {offset_params[0]}, angle = {360 -np.degrees(offset_params[1])}"
        
        elif manipulation_offset == "custom_end_offset":

            if current_state is None:
                return "Provide current_state"
            else:
                initial_stuck_point = current_state[:, 0:2][0]
                manipulation_angle = current_state[0][-1]
                
            prev_i = 1
            move_attempt = 0
            stuck = True

            while stuck:
                stuck, disp_struck = object_stuck(initial_stuck_point, expt_log_dir, prev_i = prev_i, threshold=0.03)
                if stuck:
                    move_attempt += 1
                info = "stuck:"+str(stuck)+"\tmove_attempt:"+str(move_attempt)+"\tdisp:"+ str(disp_struck)
                prev_i += 1
                
            
            x_offset, y_offset = compute_target_offset(label_width, manipulation_angle, move_attempt = move_attempt, delta_offset=0.2)
            
            # Move along Cu_axis if stuck            
            if move_attempt > 2:
                
                #x_offset, y_offset, distance_fraction = shake(norm_x, norm_y, move_attempt) 
                
                distance_fraction = min(0.3*(move_attempt - 0), 3.1)
                new_targets, new_mani_angle = move_along_Cu_axis(current_state, coordinates, distance_fraction=distance_fraction, distance_fraction_shake=3)
                    
                x_offset = new_targets[0]-norm_x
                y_offset = new_targets[1]-norm_y
                
                info = "stuck:"+str(stuck)+"\tmove_attempt:"+str(move_attempt)+"\tdisp:"+ str(disp_struck)+"\tfraction:"+str(distance_fraction)+"\tangle:"+str((1 -new_mani_angle)*360)
            

        else:
            x_offset, y_offset = 0, 0

        norm_x = norm_x + x_offset
        norm_y = norm_y + y_offset
        
        
        frame_rescaled_coords.append(np.asarray([norm_x, 1-norm_y])*frame_size)
        
    frame_rescaled_coords = np.asarray(frame_rescaled_coords)

    offset_vals = [x_offset, y_offset]
    save_offset_params(offset_vals, expt_dir, expt_name, basename, target = True)
    
    transformed_coords = transform_coord_array(frame_rescaled_coords, center_offset[0], center_offset[1], frame_size, scan_angle) 

    return transformed_coords, info



def rescale_action_params(action, old_range, new_range):

    old_range = np.asarray(old_range)
    new_range = np.array(new_range)

    #Convert to 1D array. else wrong predictions 
    action = np.ravel(np.asarray(action))
    
    rescaled_action = rescale_array(old_range, action, new_range)

    rescaled_action = np.ravel(rescaled_action)

    return rescaled_action



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
    
    

def save_offset_params(offset_vals, expt_dir, expt_name, basename, target = True):
    
    log_dir = os.path.join(expt_dir, 'expt_log', expt_name)
    os.makedirs(log_dir, exist_ok = True)       
    
    sxm_file_name,_, _ =  saved_next_sxmfilenames(expt_dir, basename)
    sxm_file_name = sxm_file_name.split('.')[0]

    if target:
        write_log(os.path.join(log_dir, 'offset_vals_target.txt'), sxm_file_name, arr_to_linestring(offset_vals))
    else:
        write_log(os.path.join(log_dir, 'offset_vals_initial.txt'), sxm_file_name, arr_to_linestring(offset_vals))
        

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



            
# def rescale_state_coords(coordinate_set, scan_params, label_width = 0, manipulation_offset = False):

    
#     coordinates = coordinate_set[:,0:2]
#     manipulation_angle = coordinate_set[0][-1]
#     #print(f'coordinates = {coordinates}')

#     frame_size, center_offset, scan_angle = scan_params
    
#     frame_rescaled_coords = []
    
#     #Invert y-coordinate to correlate to real space
#     # Transform to range 0-frame
#     for coordinate in coordinates:
#         norm_x, norm_y = coordinate
        
#         if manipulation_offset == True:
            
#             x_offset, y_offset = get_manipulation_offset(label_width, manipulation_angle)
#             norm_x = norm_x + x_offset
#             norm_y = norm_y + y_offset
            
        
#         frame_rescaled_coords.append(np.asarray([norm_x, 1-norm_y])*frame_size)
        
#     frame_rescaled_coords = np.asarray(frame_rescaled_coords)
    
#     transformed_coords = transform_coord_array(frame_rescaled_coords, center_offset[0], center_offset[1], frame_size, scan_angle) 

#     return transformed_coords



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

