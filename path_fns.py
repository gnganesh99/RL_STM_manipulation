"""
Functions to handle path planning for STM manipulation experiments.
This includes processing the coordinates, generating paths avoiding obstacles, and rescaling the spatial coordinates.

@author: Ganesh Narasimha
"""



#@author: Ganesh Narasimha
import gym

from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import time
from IPython.display import clear_output
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


from Manipulation_coords import get_manipulation_coords, get_manipulation_coords_MO
from expt_utils import transform_coord_array, translation_angle, write_log, get_latest_file, get_next_file_name, copy_files, get_sxm_filenames
from expt_utils import arr_to_linestring, euclidean_distance, saved_next_sxmfilenames, distance, path_exists, read_expt_log

from detect_molecules import get_latest_image
from stm_manipulation_experiments import manipulation_and_scan_tcp, rescan_tcp, do_manipulation_LV, rescan_LV, rescan_LV_dummy
from display_results import show_iter_image_results, show_iter_results
from drift_estimation import drift_correction_far_labels
from get_reward import get_displacement_reward, get_state_reward
from get_target import random_target, custom_target, compute_coordinates


from env_functions import *
from rrt_star_algo import run_rrt_star, Rect, segment_hits_rect




def modify_state_coords(coordinate_set, expt_dir, expt_name, label_width = 0, current_state = None,  manipulation_offset = None):

    """
    Modifies the coordinates based on the manipulation offset strategy.
    I/p:
        coordinate_set: array of normalized coordinates to be modified
        expt_dir: experiment directory
        expt_name: experiment name
        label_width: width of the label/object being manipulated
        current_state: current state of the manipulated object, used for certain offset strategies
        manipulation_offset: dictionary specifying the type and parameters of manipulation offset
            - type: type of manipulation offset ("start_offset", "end_offset_procedure", "custom_target_offset")
    O/p:
        modified_coords: array of modified coordinates
        info: additional information about the modification process

    """

    manipulation_offset_type = manipulation_offset.get("type", None)


    coordinates = coordinate_set[:,0:2]


    if current_state is not None:
        manipulation_angle = current_state[0][-1]
    else:
        return "Provide current_state"
    
    modified_coords = []
    expt_log_dir = os.path.join(expt_dir, 'expt_log', expt_name)
    info =  None

    #Invert y-coordinate to correlate to real space
    # Transform to range 0-frame
    for coordinate in coordinates:
        norm_x, norm_y = coordinate
        
        # Add coordinate offset
        if manipulation_offset_type == "start_offset":  # Simple offset at the start of manipulation based on manipulation angle. This applies a slight shift before the starting point.
            
            delta_offset = manipulation_offset.get("delta_offset", 0.1)
            
            x_offset, y_offset = compute_manipulation_offset(label_width, manipulation_angle, delta_offset=delta_offset)

        elif manipulation_offset_type == "end_offset_procedure":   # This is the previous path planning procedure. is very detailed, customized (and cumbersome !). Letting it be for now.
            
            
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
             
        elif manipulation_offset_type == "custom_target_offset": # Custom offset provided as input. The offset from the predicted actions is added here.
            
            x_offset, y_offset = manipulation_offset.get("offset", (0, 0))

            info = f"Custom offset: {x_offset}, {y_offset}"
    

        else:
            x_offset, y_offset = 0, 0

        norm_x = norm_x + x_offset
        norm_y = norm_y + y_offset
            
        modified_coords.append([norm_x, norm_y])
    

    
    return np.asarray(modified_coords), info


def get_path(x_initial, x_final, labels, exp_dir, exp_name, filename, orig_target = None, scan_params = None):

    """
    Generates a path from initial to final coordinates 
    Uses RRT* algorithm to avoid obstacles. If no obstacles are detected, a straight line path is returned.
    If the distance between initial and final coordinates is very small (below rrt_above_d), a straight line path is returned.

    I/p:

        - x_initial: initial coordinates
        - x_final: final coordinates
        - labels: object labels
        - exp_dir: experiment directory
        - exp_name: experiment name
        - filename: image filename
        - orig_target: original target coordinates (optional)
        - scan_params: scanning parameters (optional). required to compute rrt_above_d
    O/p:
        - norm_path: array of normalized path coordinates from initial to final positions

    """


    image_path = os.path.join(exp_dir, 'images', filename + '.jpg')
    
    x_initial = x_initial[:,0:2]
    x_final = x_final[:,0:2]
    
    pixels = 256
    img  =  cv2.imread(image_path)
    img = cv2.resize(img, (pixels, pixels), interpolation=cv2.INTER_NEAREST)



    dpi = 100.0
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    labels, bw, bh = labels

    num = labels.shape[0]

    obstacles = []
    obstacle_position = []
    
    
    for ind in range(num):
  
        xi  = labels[ind][0]
        yi = labels[ind][1]
        
        w = bw[ind]
        h = bh[ind]

        obstacle_position.append([xi, yi])

        r_x = np.clip((xi-w/2)*pixels, 0, pixels)
        r_y = np.clip((yi-h/2)*pixels, 0, pixels)
        w = np.clip(w*pixels, 0, pixels)
        h = np.clip(h*pixels, 0, pixels)
        
        rect = [r_x, r_y, w, h]
        obstacles.append(rect)

    
        s = patches.Rectangle((r_x, r_y), w, h, color = 'r', linewidth=1, fill=False)
                
        ax.add_patch(s)

    all_distances = np.linalg.norm(labels - x_initial[0], axis=1)
    closest_idx = np.argmin(all_distances)
    bw_selected = bw[closest_idx]
    bh_selected = bh[closest_idx]

    del obstacles[closest_idx]
    labels = np.delete(labels, closest_idx, axis=0)
    bw = np.delete(bw, closest_idx, axis=0) 
    bh = np.delete(bh, closest_idx, axis=0)


    # If the initial point overlaps with an obstacle, shrink the obstacle until they don't overlap
    all_distances = np.linalg.norm(labels - x_initial[0], axis=1)
    closest_idx = np.argmin(all_distances)
    init_rect = [x_initial[0][0], x_initial[0][1], bw_selected, bh_selected]

    x_io = labels[closest_idx][0]
    y_io = labels[closest_idx][1]
    bw_initial_obstacle = bw[closest_idx]
    bh_initial_obstacle = bh[closest_idx]
    obstacle_rect = [x_io, y_io, bw_initial_obstacle, bh_initial_obstacle]

    count_shrink = 0
    while intersection_area(init_rect, obstacle_rect) > 0:
        
        # shrink the obstcle a bit until they don't overlap
        bw_initial_obstacle *= 0.9
        bh_initial_obstacle *= 0.9

        obstacle_rect = [x_io, y_io, bw_initial_obstacle, bh_initial_obstacle]
        obstacles[closest_idx] = [(x_io - bw_initial_obstacle/2)*pixels, (y_io - bh_initial_obstacle/2)*pixels, bw_initial_obstacle*pixels, bh_initial_obstacle*pixels]

        count_shrink += 1
        if count_shrink > 50:
            print(f"Warning: Could not resolve overlap between intial object and obstacle after {count_shrink} iterations.")
            break



    # If the target overlaps with an obstacle, shrink the obstacle until they don't overlap
    all_distances = np.linalg.norm(labels - x_final[0], axis=1)
    closest_idx = np.argmin(all_distances)
    target_rect = [x_final[0][0], x_final[0][1], bw_selected, bh_selected]

    x_to = labels[closest_idx][0]
    y_to = labels[closest_idx][1]
    bw_target_obstacle = bw[closest_idx]
    bh_target_obstacle = bh[closest_idx]
    obstacle_rect = [x_to, y_to, bw_target_obstacle, bh_target_obstacle]

    count_shrink = 0
    while intersection_area(target_rect, obstacle_rect) > 0:
        
        # shrink the obstcle a bit until they don't overlap
        bw_target_obstacle *= 0.9
        bh_target_obstacle *= 0.9

        obstacle_rect = [x_to, y_to, bw_target_obstacle, bh_target_obstacle]
        obstacles[closest_idx] = [(x_to - bw_target_obstacle/2)*pixels, (y_to - bh_target_obstacle/2)*pixels, bw_target_obstacle*pixels, bh_target_obstacle*pixels]

        count_shrink += 1
        if count_shrink > 50:
            print(f"Warning: Could not resolve overlap between target and obstacle after {count_shrink} iterations.")
            break

    
    for ind in range(x_initial.shape[0]): # for loop is not necessary here, retained for convenience.
        
        xi  = x_initial[ind][0]
        yi = x_initial[ind][1]
        
        xf = x_final[ind][0]
        yf = x_final[ind][1]
        
        
        w = bw_selected
        h = bh_selected

        s_x = np.clip((xi)*pixels, 0, pixels)
        s_y = np.clip((yi)*pixels, 0, pixels)

        start =  [s_x, s_y]

        #Sqaure pathches for the intial coords
        s_i = patches.Rectangle(((xi-w/2)*pixels,(yi-h/2)*pixels), w*pixels, h*pixels, color = 'black', linewidth=1, fill=False)


   
        g_x = np.clip((xf)*pixels, 0, pixels)
        g_y = np.clip((yf)*pixels, 0, pixels)
        
        goal = [g_x, g_y]

        #Square patches for the target coords
        s_f = patches.Rectangle(((xf-w/2)*pixels,(yf-h/2)*pixels), w*pixels, h*pixels, color = 'black', linewidth=1, fill=False, linestyle = '--')
        
        
        
        # indicate original target
        s_ot = None
        if orig_target is not None:
            x_ot, y_ot = xi  = orig_target[ind][0], orig_target[ind][1]
            #Square patches for the target coords
            s_ot = patches.Rectangle(((x_ot-w/2)*pixels,(y_ot-h/2)*pixels), w*pixels, h*pixels, color = 'magenta', linewidth=1, fill=False)
            
                    
        
        ax.add_patch(s_i)
        ax.add_patch(s_f)
        if s_ot is not None:
            ax.add_patch(s_ot)
    
    if scan_params is None:
        rrt_above_d = 0
    else:
        rrt_above_d = (2E-9/scan_params[0])*pixels
    path, p_label = run_rrt_star(obstacles, start, goal, pixels  = pixels, max_iters = 10000, padding = (w+h)*pixels*0.25, rrt_above_d = rrt_above_d)


    px, py = zip(*path)
    ax.plot(px, py, 'r--', linewidth=2, label=p_label)
    ax.plot(start[0], start[1], 'og', markersize=6)
    ax.plot(goal[0], goal[1], 'xb', markersize=6)
    ax.legend()

    ax.set_axis_off()
    
    rrt_img_path = os.path.join(exp_dir, filename+'_path.jpg')
    plt.savefig(rrt_img_path, bbox_inches = 'tight', pad_inches = 0 )

    norm_path = path/pixels

    # Ensure the start and end points are exactly as specified
    if norm_path.shape[0] > 1:
        norm_path[0] = x_initial[0]
        norm_path[-1] = x_final[0]
    else:
        norm_path[0] = x_initial[0]


    return norm_path



def save_path_img(folder_name, file_name, labels, bw, bh, x_initial, x_final, bw_selected, bh_selected, obj_idx = 0): # this is currently not being used.
    
    """
    Saves the images with annotation of the intial points with assignment to the target points.

    Inputs
    folder: folder directory
    file_name: sxm file name
    x_initial: the array of assigned molecules
    x_final: array of the target points
    bw, bh: arrays of the width and height of the x_initial-labels  
    """

    image_path = folder_name + '/images/'+ file_name + '.jpg'
    
    
    pixels = 256
    img  =  cv2.imread(image_path)
    img = cv2.resize(img, (pixels, pixels), interpolation=cv2.INTER_NEAREST)



    dpi = 100.0
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    

    num = labels.shape[0]
    obstacles = []
    obstacle_position = []
    
    
    for ind in range(num):
  
        xi  = labels[ind][0]
        yi = labels[ind][1]
        
        w = bw[ind]
        h = bh[ind]

        obstacle_position.append([xi, yi])

        r_x = np.clip((xi-w/2)*pixels, 0, pixels)
        r_y = np.clip((yi-h/2)*pixels, 0, pixels)
        w = np.clip(w*pixels, 5, pixels)
        h = np.clip(h*pixels, 5, pixels)
        
        rect = [r_x, r_y, w, h]
        obstacles.append(rect)

    
        s = patches.Rectangle((r_x, r_y), w, h, color = 'r', linewidth=1, fill=False)
                
        ax.add_patch(s)

    

    num = x_initial.shape[0]
    
    color_arr =  plt.cm.jet(np.linspace(0, 1, num), alpha = 0.8)

    
    for ind in range(num):
        if ind != obj_idx:
            continue

        xi  = x_initial[ind][0]
        yi = x_initial[ind][1]
        
        xf = x_final[ind][0]
        yf = x_final[ind][1]
        
        w = bw_selected[ind]
        h = bh_selected[ind]

        s_x = np.clip((xi)*pixels, 0, pixels)
        s_y = np.clip((yi)*pixels, 0, pixels)

        start =  [s_x, s_y]

        #Sqaure pathches for the intial coords
        s_i = patches.Rectangle(((xi-w/2)*pixels,(yi-h/2)*pixels), w*pixels, h*pixels, color = color_arr[ind], linewidth=1, fill=False)


        #remove this object from the obstacles
        eps = (w+h)*0.25
        for i, obs in enumerate(obstacle_position):          # obs = [x, y] (example)
            ox, oy = obs[0], obs[1]
            if math.isclose(xi, ox, abs_tol=eps) and math.isclose(yi, oy, abs_tol=eps):
                del obstacles[i]
                break



        g_x = np.clip((xf)*pixels, 0, pixels)
        g_y = np.clip((yf)*pixels, 0, pixels)
        
        goal = [g_x, g_y]

        #Square patches for the target coords
        s_f = patches.Rectangle(((xf-w/2)*pixels,(yf-h/2)*pixels), w*pixels, h*pixels, color = color_arr[ind], linewidth=1, fill=False, linestyle = '--')
        
        ax.add_patch(s_i)
        ax.add_patch(s_f)
    
    
    path = run_rrt_star(obstacles, start, goal, pixels  = pixels, max_iters = 10000)


    if path is not None:
        path = np.unique(np.asarray(path), axis=0)
        p_label = "RRT* path"
    else:
        path = np.asarray([start, goal])
        p_label = "Displacement path"


    px, py = zip(*path)
    ax.plot(px, py, '-r', linewidth=2, label=p_label)
    ax.plot(start[0], start[1], 'og', markersize=8, label='Start')
    ax.plot(goal[0], goal[1], 'xb', markersize=8, label='Goal')
    ax.legend()


    ax.set_axis_off()

    
    rrt_img_path = folder_name + '/' + 'path.jpg'
    plt.savefig(rrt_img_path, bbox_inches = 'tight', pad_inches = 0 )







def transform_to_real_coords(coordinate_set, scan_params):

    """
    Transforms an array of normalized coordinates to real coordinates based on scan parameters.
    Args:
        coordinate_set (np.ndarray): An array of shape (N, 2) containing normalized coordinates (x, y) in the range [0, 1].
        scan_params (tuple): A tuple containing (frame_size, center_offset, scan_angle).
            - frame_size (float): The size of the scan frame.
            - center_offset (tuple): A tuple (x_offset, y_offset) representing the center offset in real coordinates.
            - scan_angle (float): The rotation angle of the scan in degrees.
    Returns:
        transformed_coords (np.ndarray): An array of shape (N, 2) containing the transformed real coordinates.
    """


    frame_size, center_offset, scan_angle = scan_params
    
    frame_rescaled_coords = []
    coordinates = coordinate_set[:,0:2]

    for coordinate in coordinates:
        coordinate =  np.clip(coordinate, 0, 1)
        norm_x, norm_y = coordinate
        frame_rescaled_coords.append(np.asarray([norm_x, 1-norm_y])*frame_size)
    
    frame_rescaled_coords = np.asarray(frame_rescaled_coords)
    
    transformed_coords = transform_coord_array(frame_rescaled_coords, center_offset[0], center_offset[1], frame_size, scan_angle) 
    
    return transformed_coords





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
    current_state = np.asarray(current_state).ravel()
    

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





def rescale_initial_coords(coordinate_set, state_action, scan_params, expt_dir, expt_name, basename, offset_params = np.ones((2,))*-1, label_width = 0, 
                           manipulation_offset = None, current_state = None):

    coordinates = coordinate_set[:,0:2]

    manipulation_angle = coordinates[0][-1] * 360
    
    frame_size, center_offset, scan_angle = scan_params
    
    frame_rescaled_coords = []
    expt_log_dir = os.path.join(expt_dir, 'expt_log', expt_name)

    
    old_offset_range = np.asarray([[-1, 1], [-1, 1]])
    new_offset_range = np.asarray([[0, label_width], np.radians([manipulation_angle-90, manipulation_angle+90])])
    offset_params = rescale_array(old_offset_range, np.ravel(offset_params), new_offset_range)
          
    

    #Invert y-coordinate to correlate to real space
    # Transform to range 0-frame

    for coordinate in coordinates:
        norm_x, norm_y = coordinate
        
        if manipulation_offset == "start_offset":

            offset_params = np.ravel(offset_params)
            x_offset, y_offset = compute_coordinates(offset_params[0], offset_params[1])
            info = f"Offset: {x_offset}, {y_offset}"
        
        elif manipulation_offset == "custom_start_offset":

            x_offset, y_offset = compute_manipulation_offset(label_width, manipulation_angle, delta_offset=0.18)
            

        else:
            x_offset, y_offset = 0, 0

        norm_x = norm_x + x_offset
        norm_y = norm_y + y_offset
            
        
        frame_rescaled_coords.append(np.asarray([norm_x, 1-norm_y])*frame_size)
        
    frame_rescaled_coords = np.asarray(frame_rescaled_coords)
    
    offset_vals = [x_offset, y_offset]
    save_offset_params(offset_vals, expt_dir, expt_name, basename, target = False)
    
    transformed_coords = transform_coord_array(frame_rescaled_coords, center_offset[0], center_offset[1], frame_size, scan_angle) 

    return transformed_coords




def rescale_target_coords(coordinate_set, state_action, scan_params, expt_dir, expt_name, basename, offset_params = np.ones((2,))*-1, label_width = 0, manipulation_offset = None, current_state = None):

    coordinates = coordinate_set[:,0:2]

    manipulation_angle = current_state[0][-1] * 360
    
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
            info = f"Offset: {x_offset}, {y_offset}"
        
        elif manipulation_offset == "custom_end_offset":

            prev_i = 1
            move_attempt = 0
            stuck = True

            while stuck:
                stuck = object_stuck(coordinate_set, expt_log_dir, prev_i = prev_i, threshold=0.03)
                if stuck:
                    move_attempt += 1
                prev_i += 1

            x_offset, y_offset = compute_target_offset(label_width, manipulation_angle, move_attempt = move_attempt, delta_offset=0.2)

            if move_attempt > 5:
                # Move a little bit either in vertical or horizontal direction
                fraction = min(0.1*(move_attempt-5), 0.5)

                if np.random.rand() > 0.5:
                    x_offset, y_offset = -norm_x*(1-fraction), -norm_y
                else:
                    x_offset, y_offset = -norm_x, -norm_y*(1-fraction)

                info = f"Fractional offset: {fraction}"

        else:
            x_offset, y_offset = 0, 0

        norm_x = norm_x + x_offset
        norm_y = norm_y + y_offset
        
        
        frame_rescaled_coords.append(np.asarray([norm_x, 1-norm_y])*frame_size)
        
    frame_rescaled_coords = np.asarray(frame_rescaled_coords)

    offset_vals = [x_offset, y_offset]

    save_offset_params(offset_vals, expt_dir, expt_name, basename, target = True)
    
    transformed_coords = transform_coord_array(frame_rescaled_coords, center_offset[0], center_offset[1], frame_size, scan_angle) 

    return transformed_coords


def rescale_state_coords(coordinate_set, scan_params, expt_dir, expt_name, label_width = 0, manipulation_offset = None):

    coordinates = coordinate_set[:,0:2]
    manipulation_angle = coordinate_set[0][-1]
    #print(f'coordinates = {coordinates}')

    frame_size, center_offset, scan_angle = scan_params
    
    frame_rescaled_coords = []
    expt_log_dir = os.path.join(expt_dir, 'expt_log', expt_name)

    #Invert y-coordinate to correlate to real space
    # Transform to range 0-frame
    for coordinate in coordinates:
        norm_x, norm_y = coordinate
        
        # Add coordinate offset
        if manipulation_offset == "start_offset":
            
            x_offset, y_offset = compute_manipulation_offset(label_width, manipulation_angle, delta_offset=0.18)

        elif manipulation_offset == "end_offset":

            prev_i = 1
            move_attempt = 0
            stuck = True

            while stuck:
                stuck = object_stuck(coordinate_set, expt_log_dir, prev_i = prev_i, threshold=0.03)
                if stuck:
                    move_attempt += 1
                prev_i += 1

            x_offset, y_offset = compute_target_offset(label_width, manipulation_angle, move_attempt = move_attempt, delta_offset=0.2)

            if move_attempt > 5:
                # Move a little bit either in vertical or horizontal direction
                fraction = min(0.1*(move_attempt-5), 0.5)

                if np.random.rand() > 0.5:
                    x_offset, y_offset = -norm_x*(1-fraction), -norm_y
                else:
                    x_offset, y_offset = -norm_x, -norm_y*(1-fraction)

                info = f"Fractional offset: {fraction}"



        else:
            x_offset, y_offset = 0, 0

        norm_x = norm_x + x_offset
        norm_y = norm_y + y_offset
            
        
        frame_rescaled_coords.append(np.asarray([norm_x, 1-norm_y])*frame_size)
        
    frame_rescaled_coords = np.asarray(frame_rescaled_coords)
    
    transformed_coords = transform_coord_array(frame_rescaled_coords, center_offset[0], center_offset[1], frame_size, scan_angle) 

    return transformed_coords


def target_avoid_collision(target_i, target_f, all_labels, margin, padding = 0.0, x_initial = None, eps = 0.01):
    
    """
    Avoids collisions with other objects by adjusting the target final coordinates.
    If a collision is detected, the target final coordinates are adjusted to retrace along the path to avoid the obstacle.

    Inputs:
        - target_i: initial target coordinates (numpy array of shape (1, 2))
        - target_f: final target coordinates (numpy array of shape (1, 2))
        - all_labels: tuple of (labels, bw, bh) where labels is a numpy array of shape (N, 2) containing the coordinates of other objects,
                      bw and bh are numpy arrays of shape (N,) containing the width and height of the objects respectively.
        - margin: margin to keep from the edges (float)
        - padding: additional padding to consider around the objects (float)
        - x_initial: initial coordinates of the object being manipulated (numpy array of shape (1, 2))
        - eps: small value to determine convergence (float)

    Outputs:
        - target_f: adjusted final target coordinates (numpy array of shape (1, 2))
        - info: string indicating whether collisions were avoided or not

    """
    
    labels, bw, bh = all_labels
    labels, bw, bh = np.asarray(labels).copy(), np.asarray(bw).copy(), np.asarray(bh).copy()
    

    target_i, target_f = target_i.ravel().copy(), target_f.ravel().copy()

    # Remove the label corresponding to the molecule being manipulated.
    if len(labels) > 0:
        all_distances = np.linalg.norm(labels - x_initial, axis=1)
        closest_idx = np.argmin(all_distances)
        bw_i = bw[closest_idx]
        bh_i = bh[closest_idx]

        labels = np.delete(labels, closest_idx, axis=0)
        bw = np.delete(bw, closest_idx, axis=0)
        bh = np.delete(bh, closest_idx, axis=0)

    if labels is None or len(labels) == 0 :
        target_f = np.clip(target_f, margin, 1-margin)
        target_f  = np.ravel(target_f).reshape(1, -1)
        return target_f, "No labels to avoid"

    

    # check for collision with other labels, give additioinal padding of (bw+bh)/4
    collision_free = False
    count = 0
    while not collision_free:
        collision_indices = get_collision_indices(target_i, target_f, labels, bw, bh, padding = ((bw_i+bh_i)/4 + padding))
        if len(collision_indices) == 0:
            break 
        idx = collision_indices[0]

        target_f = first_rect_intersection(target_i, target_f, [labels[idx][0], labels[idx][1], bw[idx], bh[idx]], padding = ((bw_i+bh_i)/4+padding))

        if target_f is None:
            target_f = target_i
            break
        
        # check is the target_f intersects with the obstacle, if yes, retrace a bit
        target_rect = [target_f[0], target_f[1], bw_i, bh_i]
        obstacle_rect = [labels[idx][0], labels[idx][1], bw[idx], bh[idx]]

        while intersection_area(target_rect, obstacle_rect) > 0:
            #retrace target_f a bit (10% of the way back to target_i)
            target_f = point_on_segment(target_i, target_f, 0.9)
            target_rect = [target_f[0], target_f[1], bw_i, bh_i]
            count = 0
            if distance(target_i, target_f) < eps:
                target_f = target_i
                break

        count += 1
        if count > 1:
            target_f = point_on_segment(target_i, target_f, 0.9)  #last resort to reduce the offset.
        if count > 50 or distance(target_i, target_f) < eps :
            print(f"{count} collision avoidance iterations")
            target_f = target_i
            break
            
    
    target_f = np.clip(target_f, margin, 1-margin)
    target_f  = np.ravel(target_f).reshape(1, -1)

    return target_f, "avoided collisions"





def target_avoid_collision_1(target_i, target_f, all_labels, margin, padding = 0.0, x_initial = None, eps = 0.01, retrace_fr = 0.99,
                             overlap_iou_th = 0.2):
    

    """
    Avoids collisions with other objects by adjusting the target final coordinates.
    If a collision is detected, the target final coordinates are adjusted to retrace along the path to avoid the obstacle.

    Inputs:
        - target_i: initial target coordinates (numpy array of shape (1, 2))
        - target_f: final target coordinates (numpy array of shape (1, 2))
        - all_labels: tuple of (labels, bw, bh) where labels is a numpy array of shape (N, 2) containing the coordinates of other objects,
                      bw and bh are numpy arrays of shape (N,) containing the width and height of the objects respectively.
        - margin: margin to keep from the edges (float)
        - padding: additional padding to consider around the objects (float)
        - x_initial: initial coordinates of the object being manipulated (numpy array of shape (1, 2))
        - eps: small value to determine convergence (float)
        - retrace_fr: fraction to retrace along the path when a collision is detected (float between 0 and 1)
        - overlap_iou_th: IOU threshold to consider overlap with initial target (float)

    Outputs:
        - target_f: adjusted final target coordinates (numpy array of shape (1, 2))
        - info: string indicating whether collisions were avoided or not

    """


    retrace_fr = np.clip(retrace_fr, 0, 0.99) #retract_fraction
    labels, bw, bh = all_labels
    labels, bw, bh = np.asarray(labels).copy(), np.asarray(bw).copy(), np.asarray(bh).copy()
    
    manipulation_angle = x_initial[0][4]
    x_initial = x_initial[:, 0:2].copy()
     
    
    pad_fac = 1.1  #additional padding_factor. 5 %

    target_i, target_f = target_i.ravel().copy(), target_f.ravel().copy()

    # Remove the label corresponding to the molecule being manipulated.
    if len(labels) > 0:
        all_distances = np.linalg.norm(labels - x_initial, axis=1)
        closest_idx = np.argmin(all_distances)
        bw_i = bw[closest_idx]
        bh_i = bh[closest_idx]

        labels = np.delete(labels, closest_idx, axis=0)
        bw = np.delete(bw, closest_idx, axis=0)
        bh = np.delete(bh, closest_idx, axis=0)

    if labels is None or len(labels) == 0 :
        target_f = np.clip(target_f, margin, 1-margin)
        target_f  = np.ravel(target_f).reshape(1, -1)
        return target_f, "No labels to avoid"

    
    # If the target_i marginally overlaps with an obstacle, shrink the obstacle until they don't overlap
    for i, label in enumerate(labels):
        obstacle = [label[0], label[1], bw[i], bh[i]]
        targeti_rect = [target_i[0], target_i[1], (bw_i+2*padding)*pad_fac, (bh_i+2*padding)*pad_fac]
        
        dx, dy = intersection_dxdy(targeti_rect, obstacle)
        init_area = dx*dy

        if calc_iou(targeti_rect, obstacle) < overlap_iou_th: #shrink the obstacle until they don't overlap
            while True:
                dx, dy = intersection_dxdy(targeti_rect, obstacle)
                if dx<=dy: #reduce the smaller overlaping dimension.
                    bw[i] *= 0.95
                else:
                    bh[i] *= 0.95
                obstacle = [label[0], label[1], bw[i], bh[i]]
                if calc_iou(targeti_rect, obstacle) == 0 or dx*dy < init_area/5:
                    break


    # # avoid collisions with target_i.
    # while True:
    #     collision_indices = get_collision_indices(target_i, target_i, labels, bw, bh, padding = ((bw_i+bh_i)/4 + padding)*pad_fac)
    #     if len(collision_indices) == 0:
    #         break
            
    #     coll_idx =  collision_indices[0]
    #     obs_rect = [labels[coll_idx][0], labels[coll_idx][1], bw[coll_idx], bh[coll_idx]]
    #     iou_i = calc_iou(targeti_rect, obs_rect)
        
    #     if iou_i < overlap_iou_th:
    #         bw[coll_idx] *= 0.95
    #         bh[coll_idx] *= 0.95
    #     else:
    #         break

        
        
    


    # check for collision and intersection with other labels, give additioinal padding of (bw+bh)/4
    count = 0
    entry = ''
    while True:
        
        collision_indices = get_collision_indices(target_i, target_f, labels, bw, bh, padding = ((bw_i+bh_i)/4 + padding)*pad_fac)
        
        rect_target_f = [target_f[0], target_f[1], bw_i*pad_fac, bh_i*pad_fac]
        intersection_indices = get_intersection_indices(rect_target_f, labels, bw, bh)
        
        collision_free  = True if len(collision_indices) == 0 else False
        intersection_free = True if len(intersection_indices) == 0 else False
        #intersection_free = True

        if collision_free and intersection_free:
            break
        target_f = point_on_segment(target_i, target_f, retrace_fr) #Retrace the target_f closer to target_i.
        count += 1
        
        # for idx in collision_indices:
        #     obs_rect = [labels[idx][0], labels[idx][1], bw[idx], bh[idx]]
        #     iou_i = calc_iou(targeti_rect, obs_rect)
        #     iou_f = calc_iou(rect_target_f, obs_rect)
        #     entry += str(labels[idx])+f"  iou_i:{iou_i}     iou_f:{iou_f}"'\n'
            
        # entry += f"intitial_target:{target_i}, final_target: {target_f}"+'\n'
        # with open("check_collision_avoidance1.txt", "a") as f:
        #     f.write(entry)
        
        
        print(f"Retraced by {count} steps")

        if distance(target_i, target_f) < eps/2:
            #target_f = target_i
            x_offset, y_offset = compute_target_offset((bw_i+bh_i)/2, manipulation_angle, move_attempt = 1, delta_offset=0.3)
            target_f = target_i+np.asarray([x_offset, y_offset])
            #print("Direct offset added to target")
            break
        
                   
    
    target_f = np.clip(target_f, margin, 1-margin)
    target_f  = np.ravel(target_f).reshape(1, -1)

    return target_f, "avoided collisions"






def get_collision_indices(initial, final, labels, bw, bh, padding = 0):

    """
    Check for collisions across the manipulation path from initial to final coordinates with given labels (obstacles).
    Inputs:
        - initial: initial coordinates
        - final: final coordinates
        - labels: array of shape (N, 2) containing the coordinates of the labels (obstacles)
        - bw: array of shape (N,) containing the width of the labels
        - bh: array of shape (N,) containing the height of the labels
        - padding: additional padding to consider around the obstacles (float)

    Outputs:
        - collision_indices: list of indices of labels that collide with the path from initial to final coordinates
    """



    collision_indices = []
    for i, label in enumerate(labels):
        obstacle = Rect(label[0], label[1], bw[i], bh[i], padding)
        if segment_hits_rect(initial, final, obstacle):
            collision_indices.append(i)
    
    return collision_indices




def get_intersection_indices(rect_target, labels, bw, bh):
    """
    Check for intersections between a target rectangle and given labels (obstacles).
    Inputs:
        - rect_target: [x_center, y_center, width, height] of the target rectangle
        - labels: array of shape (N, 2) containing the coordinates of the labels (obstacles)
        - bw: array of shape (N,) containing the width of the labels
        - bh: array of shape (N,) containing the height of the labels
    Outputs:
        - intersect_indices: list of indices of labels that intersect with the target rectangle
    """

    intersect_indices = []
    for i, label in enumerate(labels):
        obstacle = [label[0], label[1], bw[i], bh[i]]
        intersect_area = intersection_area(rect_target, obstacle)
        if intersect_area > 0:
            intersect_indices.append(i)
    
    return intersect_indices




def calc_iou(rect1, rect2):
    """
    Calculate the Intersection over Union (IoU) of two rectangles (xywh format).
    """
    intersect_area = intersection_area(rect1, rect2)
    area1 = rect1[2] * rect1[3]
    area2 = rect2[2] * rect2[3]
    union_area = area1 + area2 - intersect_area
    iou = intersect_area / union_area if union_area > 0 else 0
    return iou



def first_rect_intersection(x_initial, x_final, rect, padding = 0, eps=1e-12):
    """
    x_initial, x_final: (x, y) tuples for the segment endpoints (start is outside).
    rect: [x1, y1, x2, y2] axis-aligned (corners may be unordered).

    Returns:
        (xi, yi)  # closest intersection point from x_initial along the segment
        or None if no intersection within the segment.
    """
    x0, y0 = map(float, x_initial)
    x1, y1 = map(float, x_final)

    # normalize rectangle to xmin, ymin, xmax, ymax
    rx, ry, rw, rh = rect
    rx1, ry1, rx2, ry2 = rx-rw/2-padding, ry-rh/2-padding, rx+rw/2+padding, ry+rh/2+padding
    xmin, xmax = (rx1, rx2) if rx1 <= rx2 else (rx2, rx1)
    ymin, ymax = (ry1, ry2) if ry1 <= ry2 else (ry2, ry1)

    dx = x1 - x0
    dy = y1 - y0

    candidates = []

    # Intersect with x = xmin and x = xmax (vertical edges)
    if abs(dx) > eps:
        for x_edge in (xmin, xmax):
            t = (x_edge - x0) / dx
            if 0.0 - eps <= t <= 1.0 + eps:
                y_hit = y0 + t * dy
                if ymin - eps <= y_hit <= ymax + eps:
                    candidates.append((t, (x_edge, y_hit)))

    # Intersect with y = ymin and y = ymax (horizontal edges)
    if abs(dy) > eps:
        for y_edge in (ymin, ymax):
            t = (y_edge - y0) / dy
            if 0.0 - eps <= t <= 1.0 + eps:
                x_hit = x0 + t * dx
                if xmin - eps <= x_hit <= xmax + eps:
                    candidates.append((t, (x_hit, y_edge)))

    # Keep the closest forward intersection (smallest t >= 0 within [0,1])
    candidates = [(t, pt) for t, pt in candidates if t >= -eps and t <= 1.0 + eps]
    if not candidates:
        return None

    t_first, (xi, yi) = min(candidates, key=lambda z: z[0])
    return (xi, yi)





def point_on_segment(x_initial, x_final, fraction):
    """
    Returns the point fraction (0..1) along the segment x_initial -> x_final.
    inputs:
        x_initial, x_final: (x, y) tuples
        fraction: float in [0,1]
    Returns:
        (x, y) tuple at the given fraction along the segment
    """
    x0, y0 = x_initial
    x1, y1 = x_final
    t = float(fraction)
    return (x0 + t * (x1 - x0), y0 + t * (y1 - y0))




def intersection_dxdy(rect1, rect2):

    """
    Calculate the width and height of the intersection area of two rectangles.
    inputs:
        rect1, rect2: [x_center, y_center, width, height]
    Returns:
        overlap_w, overlap_h: width and height of the intersection area (float, float)
    """ 

    points_to_corners = lambda r: (r[0]-r[2]/2, r[0]+r[2]/2, r[1]-r[3]/2, r[1]+r[3]/2)  # x1, x2, y1, y2

    x1, x2, y1, y2 = points_to_corners(rect1)
    x3, x4, y3, y4 = points_to_corners(rect2)

    # normalize corners
    xa1, xa2 = min(x1, x2), max(x1, x2)
    ya1, ya2 = min(y1, y2), max(y1, y2)
    xb1, xb2 = min(x3, x4), max(x3, x4)
    yb1, yb2 = min(y3, y4), max(y3, y4)

    # intersection bounds
    xi1 = max(xa1, xb1)  # left
    yi1 = max(ya1, yb1)  # bottom
    xi2 = min(xa2, xb2)  # right
    yi2 = min(ya2, yb2)  # top

    overlap_w = max(0, xi2 - xi1)
    overlap_h = max(0, yi2 - yi1)

    return overlap_w, overlap_h



def intersection_area(rect1, rect2):
    """
    Calculate the intersection area of two rectangles.
    inputs:
        rect1, rect2: [x_center, y_center, width, height]
    Returns:
        Intersection area (float)
    """

    points_to_corners = lambda r: (r[0]-r[2]/2, r[0]+r[2]/2, r[1]-r[3]/2, r[1]+r[3]/2)  # x1, x2, y1, y2

    x1, x2, y1, y2 = points_to_corners(rect1)
    x3, x4, y3, y4 = points_to_corners(rect2)

    # normalize corners
    xa1, xa2 = min(x1, x2), max(x1, x2)
    ya1, ya2 = min(y1, y2), max(y1, y2)
    xb1, xb2 = min(x3, x4), max(x3, x4)
    yb1, yb2 = min(y3, y4), max(y3, y4)

    # intersection bounds
    xi1 = max(xa1, xb1)  # left
    yi1 = max(ya1, yb1)  # bottom
    xi2 = min(xa2, xb2)  # right
    yi2 = min(ya2, yb2)  # top

    overlap_w = max(0, xi2 - xi1)
    overlap_h = max(0, yi2 - yi1)

    return overlap_w * overlap_h
