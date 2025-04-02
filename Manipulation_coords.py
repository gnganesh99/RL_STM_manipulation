# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:53:39 2024

Funtions:

"""

import stmpy
from ultralytics import YOLO
import os

import IPython
from IPython.display import display, Image
from IPython import display

from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd


import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
    
from expt_utils import transform_coord_array, twoD_to_1D, distance, save_scan_img, norm_0to1, sort_by_order, filepath_to_filename, path_exists, interpolate_bw_points, read_log, write_anchor_log
from stm_utils import Sxm_Image
from detect_molecules import predict_labels, get_predicted_labels, get_latest_image, check_prediction_logged, get_basename


#from drift_estimation import drift_correction_far_labels, drift_correction_exclude_labels


def get_manipulation_coords(sxm_filepath, img_dir, X_target, max_len = False):

    """
    Provides the variables to define the state in the RL environment
    This involves using object detection, linear assignment and drift correction

    i/p:
        sxm_filepath: path of the scanned sxm file
        img_dir: image directory to save the corresponding images
        X_target: target coordinates for the manipulation. This is valued in the range [0-1]

    o/p:
        intial_coords: Coordinates of the molecules assigned for manipulation
        final_coords - Coordinates of the target position
        scan_params: scan params: [frame_size, center_offsets, scan_angle] of the scan
        drift_corrected_positions: These are molecule positions corrected for drift w.r.to the previous scan.
    """

    img_path, scan_params = save_sxm_image(sxm_filepath, img_dir)
    prediction_logged = check_prediction_logged(img_path)
    

    if prediction_logged == False:
        labels, bw, bh, label_path = predict_labels(img_path)
    else:
        labels, bw, bh, label_path = get_predicted_labels(img_path)

    
    labels_exist = False

    initial_coords = []
    final_coords = []
    corrected_position = []

    if label_path != 'no_detections':

        labels_exist = True

        sxm_name, file_name = filepath_to_filename(sxm_filepath)
        sxm_dir = sxm_filepath.replace(sxm_name, '')

        #Save annotated image
        save_annotated_img(sxm_dir, file_name, labels, bw, bh)
        print(labels, X_target)
        
        # Linear assignment
        X_target = np.asarray(X_target)
        
        if max_len == False:
            initial_coords, final_coords, bw_pos, bh_pos = linear_assignment(labels, bw, bh, X_target)
        else:
            initial_coords, final_coords, bw_pos, bh_pos = linear_assignment_maxlen(labels, bw, bh, X_target)

        
        label_width = (np.mean(bw_pos)+np.mean(bh_pos))/2
        #save the intial positions w.ro.to the filename

            
        # Save assigned image
        save_assigned_img(sxm_dir, file_name, initial_coords, final_coords, bw_pos, bh_pos)


        # Include drift correction!
        #drift, corrected_position = drift_correction_far_labels(sxm_filepath, img_dir, initial_coords, final_coords)
        drift, corrected_position = no_drift_correction(sxm_filepath, img_dir, initial_coords, final_coords)
    


    return initial_coords, final_coords, scan_params, corrected_position, drift, labels_exist, label_width


def get_manipulation_coords_MO(sxm_filepath, img_dir, X_target, iteration, obj_idx, label_margin = 0):

    """
    Provides the variables to define the state in the RL environment
    This involves using object detection, linear assignment and drift correction

    i/p:
        sxm_filepath: path of the scanned sxm file
        img_dir: image directory to save the corresponding images
        X_target: target coordinates for the manipulation. This is valued in the range [0-1]

    o/p:
        intial_coords: Coordinates of the molecules assigned for manipulation
        final_coords - Coordinates of the target position
        scan_params: scan params: [frame_size, center_offsets, scan_angle] of the scan
        drift_corrected_positions: These are molecule positions corrected for drift w.r.to the previous scan.
    """

    img_path, scan_params = save_sxm_image(sxm_filepath, img_dir)
    prediction_logged = check_prediction_logged(img_path)
    

    if prediction_logged == False:
        labels, bw, bh, label_path = predict_labels(img_path)
    else:
        labels, bw, bh, label_path = get_predicted_labels(img_path)

    
    labels_exist = False

    initial_coords = []
    final_coords = []
    corrected_position = []

    if label_path != 'no_detections':

        labels_exist = True

        sxm_name, file_name = filepath_to_filename(sxm_filepath)
        sxm_dir = sxm_filepath.replace(sxm_name, '')

        #Save annotated image
        save_annotated_img(sxm_dir, file_name, labels, bw, bh)
        #print(labels, X_target)
        

        
        # Linear assignment
        X_target = np.asarray(X_target)
        
        # use anchor labels that are a collection of prev_obj_labels and the current target
        #print("atmanipulation corrds")
        #print(X_target)

        if iteration == 0:
            anchor_labels = X_target

        else:
            #assign objects to previous object positions
            anchor_labels = read_log("anchor_log.txt")

            #assign current object to target position
            anchor_labels[obj_idx] = X_target[obj_idx]

        
        labels_within_margin = []
        
        for label in labels:
            if point_within_margin(label, margin=label_margin):
                labels_within_margin.append(label)
        labels = np.asarray(labels_within_margin)
        


        #initial_coords, anchor_coords, bw_pos, bh_pos = linear_assignment(labels, bw, bh, anchor_labels)
        initial_coords, anchor_coords, bw_pos, bh_pos = linear_assignment_by_index(labels, bw, bh, anchor_labels)
        
        
        label_width = (np.mean(bw_pos)+np.mean(bh_pos))/2
                

        write_anchor_log("anchor_log.txt", initial_coords)

        #save the intial positions w.ro.to the filename

        if iteration == 0:
            final_coords = anchor_coords
            
        else:
            final_coords = X_target
        
        # Save assigned image
        save_assigned_img(sxm_dir, file_name, initial_coords, final_coords, bw_pos, bh_pos)


        # Include drift correction!
        current_intial_coord =  np.asarray([initial_coords[obj_idx]])
        current_final_coord = np.asarray([final_coords[obj_idx]])
        
        #drift, corrected_position = drift_correction_far_labels(sxm_filepath, img_dir, current_intial_coord, current_final_coord)
        drift, corrected_position = drift_correction_anchor_label(sxm_filepath, img_dir, current_intial_coord, X_target, anchor_idx=0)

    return initial_coords, final_coords, scan_params, corrected_position, drift, labels_exist, label_width

def save_annotated_img(folder_name, file_name, labels, bw, bh):
    
    """
    Saves the annotated images using the same 'file_name' as the sxm image

    Inputs:

    folder_name: folder path to save the image
    file_name: nanonis sxm file name
    labels: labels of the images with values in the range 0-1
    bw, bh: arrays of the width and height of these labels  
    """

    image_path = folder_name + 'images\\'+ file_name + '.jpg'
    fig, ax = plt.subplots()
    
    pixels = 256
    img  =  cv2.imread(image_path)
    img = cv2.resize(img, (pixels, pixels))
    plt.imshow(img)
    
    num = labels.shape[0]
    
    
    for ind in range(num):
        xi  = labels[ind][0]
        yi = labels[ind][1]
        
        w = bw[ind]
        h = bh[ind]

    
        s = patches.Rectangle(((xi-w/2)*pixels,(yi-h/2)*pixels), w*pixels, h*pixels, color = 'r', linewidth=1, fill=False)
        
        
        ax.add_patch(s)
        
    
    ax.set_axis_off()
       

    annotated_img_path = folder_name + '/' + file_name+'_detectCO.jpg'
    plt.savefig(annotated_img_path, bbox_inches = 'tight', pad_inches = 0)

    

def save_assigned_img(folder_name, file_name, x_initial, x_final, bw, bh):
    
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
    fig, ax = plt.subplots()
    
    pixels = 256
    img  =  cv2.imread(image_path)
    img = cv2.resize(img, (pixels, pixels))
    plt.imshow(img)
    
    num = x_initial.shape[0]
    
    color_arr =  plt.cm.jet(np.linspace(0, 1, num), alpha = 0.8)
    
    for ind in range(num):
        xi  = x_initial[ind][0]
        yi = x_initial[ind][1]
        
        xf = x_final[ind][0]
        yf = x_final[ind][1]
        
        w = bw[ind]
        h = bh[ind]

        #Sqaure pathches for the intial coords
        s_i = patches.Rectangle(((xi-w/2)*pixels,(yi-h/2)*pixels), w*pixels, h*pixels, color = color_arr[ind], linewidth=1, fill=False)

        #Square patches for the target coords
        s_f = patches.Rectangle(((xf-w/2)*pixels,(yf-h/2)*pixels), w*pixels, h*pixels, color = color_arr[ind], linewidth=1, fill=False, linestyle = '--')
        
        ax.add_patch(s_i)
        ax.add_patch(s_f)
    
    ax.set_axis_off()
    
    assigned_img_path = folder_name + '/' + file_name+'_assigned.jpg'
    plt.savefig(assigned_img_path, bbox_inches = 'tight', pad_inches = 0)
    

    
def sort_assignment(costs, reverse = True) -> np.ndarray :
    '''
    Solves the linear assignment problem to reduce the cost.
    Sorts the rows, colomns and the costs in the decreasing order of the costs (reverse = True)
    
    provides wrong output if costs are same !!!
    
    Inputs
    costs: cost matrix. Here it is based on the distance between inital to final points
    if reverse = True, positions are sorted based on decreasing order of the cost

    Outputs:
    array of initial point-positions
    array of final point-positions
    associated cost(distance)
    '''
    row, col = linear_sum_assignment(costs)

    row_col_dict ={}
    col_sort = []

    for i in range(row.shape[0]):
        row_col_dict[row[i]] = col[i]

    # Sort row and costs in the decreasing order of the costs
    row_sort, cost_sort = sort_by_order(row, costs[row,col], reverse)

    # Sort colomn in as per row_sort
    for element in row_sort:
        col_sort.append(row_col_dict[element])

    return np.array(row_sort), np.array(col_sort), np.array(cost_sort)


    
def sort_assignment_col(costs, reverse = False):
    '''
    Solves the linear assignment problem to reduce the cost.
    Sorts the rows, colomns and the costs in the same order (increasing index) of the target positions (reverse = False)

    Inputs
    costs: cost matrix. Here it is based on the distance between inital to final points
    if reverse = True, positions are sorted based on decreasing order of the cost

    Outputs:
    array of initial point-positions
    array of final point-positions
    associated cost(distance)
    '''

    row, col = linear_sum_assignment(costs)

    row_cost_dict ={}
    cost_sort = []

    for i in range(row.shape[0]):
        row_cost_dict[row[i]] = costs[row[i], col[i]]

    # Sort row and col in the increasing col-index
    row_sort, col_sort = sort_by_order(row, col, reverse)

    # Sort cost in as per row_sort
    for element in row_sort:
        cost_sort.append(row_cost_dict[element])

    return np.array(row_sort), np.array(col_sort), np.array(cost_sort)


def distance_cost_matrix(X_initial, X_final):
    cost = []
       
    for initial_position in X_initial:
        # distance matrix for a given initial_position
        distance_i = []
        
        for final_position in X_final:
            
            #Compute Euclidean distance
            dis =  distance(initial_position, final_position)
            distance_i.append(dis)
            
        cost.append(distance_i)
    
    cost = np.asarray(cost)

    return cost


def distance_collision_cost_matrix(X_initial, X_final, bw, bh):
    
    cost = []
    X_intial = np.asarray(X_initial)
    X_final = np.asarray(X_final)
    
    n_objects = X_initial.shape[0]
    
    for idx in range(n_objects):
        # distance matrix for a given initial_position
        distance_i = []
        initial_position = X_intial[idx]
        avg_bw = (bw[idx]+bh[idx])/2
        
        for final_position in X_final:
            
            #Compute Euclidean distance
            dis =  distance(initial_position, final_position)
            collision_cost = path_collision(initial_position, final_position, X_initial, avg_bw)
            distance_i.append(dis + collision_cost)
            
        cost.append(distance_i)
    
    cost = np.asarray(cost)

    return cost

def distance_collision_cost_matrix_1(X_initial, X_final, bw, bh):
    
    cost = []
    X_intial = np.asarray(X_initial)
    X_final = np.asarray(X_final)
    
    n_objects = X_initial.shape[0]
    
    for idx in range(n_objects):
        # distance matrix for a given initial_position
        distance_i = []
        initial_position = X_intial[idx]
        avg_bw = (bw[idx]+bh[idx])/2
        
        final_object_i = 0 #len(X_final)
        for final_position in X_final:
            initial_position = [0.5, 0.5]
            #Compute Euclidean distance
            dis = distance(initial_position, final_position)
            collision_cost =  0 #final_object_i*1 #path_collision(initial_position, final_position, X_initial, avg_bw)
            distance_i.append(dis + collision_cost)
            final_object_i += 1
        cost.append(distance_i)
    
    cost = np.asarray(cost)

    return cost


def path_collision(initial_coord, final_coord, labels, avg_bw):

    exclude_points = [initial_coord, final_coord]
    mani_path = interpolate_bw_points(initial_coord, final_coord)
    n_collision = 0

    for object_pos in labels:

        if (object_pos[0] != initial_coord[0]) and (object_pos[1] != initial_coord[1]):

            collision = False

            for point in mani_path:

                if distance(point, object_pos) < 2*avg_bw:

                    collision = True

            if collision == True:
                n_collision += 10

    return n_collision


def linear_assignment(x_current, bw, bh, x_target) -> np.ndarray:
    
    '''
    Computes linear assignment to minimise cost.
    The cost here is the net-distance between x_current and x_target
    here x_current and x-targets are tensors in the range 0-1


    Output:
    array of positons conrrespondin to initial (current) atom-positions
    array of target positions
    array of the widths of the atom-labels
    array of the heights of the atom-labels
    '''
    
    cost = distance_cost_matrix(x_current, x_target)
    #cost = distance_collision_cost_matrix(x_current, x_target, bw, bh)  
    #cost = distance_collision_cost_matrix_1(x_current, x_target, bw, bh) 
    
    # Sorted linear assignment, with descending value of the distance:
        
    #row, col, sorted_costs = sort_assignment(cost)
    
    # Sorted linear assignment, with respect to index-value of the target atom:
        
    row, col, sorted_costs = sort_assignment_col(cost)
    
    # Rearrage atom-positions based on sorted assignment
    initial_atoms = []
    final_atoms = []

    
    for index in row:
        initial_atoms.append(x_current[index])
        
    bw_atom = bw[row]
    bh_atom = bh[row]
        
    for index in col:
        final_atoms.append(x_target[index])
    
    initial_atoms = np.asarray(initial_atoms)
    final_atoms = np.asarray(final_atoms)
    
    
    return initial_atoms, final_atoms, bw_atom, bh_atom


def linear_assignment_by_index(x_current, bw, bh, x_target):

    x_current = np.asarray(x_current)
    x_target = np.asarray(x_target)
    

    initial_atoms = []
    bw_atoms = []
    bh_atoms = []

    for i in range(len(x_target)):
        cost = distance_cost_matrix(x_current, x_target[i:i+1])
        row, col, sorted_costs = sort_assignment_col(cost) 
        row = np.squeeze(row)
        
        initial_atoms.append(x_current[row])
        bw_atoms.append(bw[row])
        bh_atoms.append(bh[row])

        x_current = np.delete(x_current, row, axis=0)
        bw = np.delete(bw, row, axis = 0)
        bh = np.delete(bh, row, axis = 0)
        

    initial_atoms = np.asarray(initial_atoms)
    
    bw_atoms = np.asarray(bw_atoms)
    bh_atoms = np.asarray(bh_atoms)


    return initial_atoms, x_target, bw_atoms, bh_atoms


def linear_assignment_maxlen(x_current, bw, bh, x_target) -> np.ndarray:
    
    '''
    Computes linear assignment to minimise cost.
    The cost here is the net-distance between x_current and x_target
    here x_current and x-targets are tensors in the range 0-1


    Output:
    array of positons conrrespondin to initial (current) atom-positions
    array of target positions
    array of the widths of the atom-labels
    array of the heights of the atom-labels
    '''
    
    cost = distance_cost_matrix(x_current, x_target)
       
    
    # Sorted linear assignment, with descending value of the distance:
        
    #row, col, sorted_costs = sort_assignment(cost)
    
    # Sorted linear assignment, with respect to index-value of the target atom:
        
    row, col, sorted_costs = sort_assignment(cost)
    
    # Rearrage atom-positions based on sorted assignment
    initial_atoms = []
    final_atoms = []

    
    for index in row:
        initial_atoms.append(x_current[index])
        
    bw_atom = bw[row]
    bh_atom = bh[row]
        
    for index in col:
        final_atoms.append(x_target[index])
    
    initial_atoms = np.asarray(initial_atoms)
    final_atoms = np.asarray(final_atoms)
    
    
    return initial_atoms[0,:], final_atoms[0,:], [bw_atom[0]], [bh_atom[0]]


def save_sxm_image(sxm_filepath, img_dir):

    """
    Saves an image for the sxm scan file.

    i/p: 
        sxm_filepath: path of scanned sxm file
        img_dir: directory for saving the images

    o/p:
        saved image_path
        array of scan paramters: frame_size, center_offsets, scan_angle
    """

    # Unpack sxm file
    scan = Sxm_Image(sxm_filepath)
    image = scan.image()
    scan_frame = scan.frame
    scan_offset =  scan.scan_offset
    scan_angle = scan.scan_angle
    
           
    # Save the image as a jpg
    image,_,_ = norm_0to1(image)
    image = cv2.resize(image, (256,256))
    
    _, primary_name = filepath_to_filename(sxm_filepath)
    img_name = primary_name + '.jpg'
    image_path = os.path.join(img_dir, img_name)
    save_scan_img(image_path, image)

    scan_parameters = [scan_frame, scan_offset, scan_angle]

    return image_path, scan_parameters


def get_prev_img_name(filename, img_dir, precede = 1):

    """
    Returns the previous image name based on the numerical suffix.
    i/p:
        filename: filename string w/ or w/o the extension - sxm or jpg
        img-dir: directory of the image.

    O/p:
        prev_imag_path: image path of the previous image file.
        prev_img_logged: True if yolo prediction logged
    """

    _,basename , img_no = get_basename(filename)

    #Get previous scan-image
    prev_img_name = get_basename(basename+str(img_no - precede))[1] + str(get_basename(basename+str(img_no - precede))[2]) + '.jpg'    
    prev_img_path =  os.path.join(img_dir, prev_img_name)

    # Check if the the previous image was logged with yolo-prediction
    prev_img_logged = check_prediction_logged(prev_img_path)

    return prev_img_path, prev_img_logged




# copied here to avoid circular import

def drift_correction_far_labels(sxm_filepath, img_dir, X_initial, X_final):

    """
    Computes drift in the scan w.r.to the previous scan
    The computation is by calculating the average distance moved by the "un-manipulated" molecules
    The "unmanipulated labels" are those which are far from the line of manipulation
    Linear assignment is used to correlate the molecules between the scan frames

    i/p:
        sxm_filepath: path of the scanned sxm file
        img_dir: image directory to save the corresponding images
        assigned_coords: coordinates of the molecules assigned for manipulation.(i.e., likely manipulated in the previous state)
    """

    sxm_name, file_name = filepath_to_filename(sxm_filepath)
    sxm_dir = sxm_filepath.replace(sxm_name, '')

    # Get prev_image which has valid detections.
    while True:
        count = 1
        prev_img_path, prev_img_logged = get_prev_img_name(file_name, img_dir, precede = count)

        #Save the previous image if it doesn't exist
        if path_exists(prev_img_path) == False:
        
            prev_sxm_file = filepath_to_filename(prev_img_path)[1]+'.sxm'
            prev_sxm_path = os.path.join(sxm_dir, prev_sxm_file)
            prev_img_path,_ = save_sxm_image(prev_sxm_path, img_dir)

        
        # Get prev labels
        if prev_img_logged == False:
            prev_labels, bw, bh, prev_label_path = predict_labels(prev_img_path)
        else:
            prev_labels, bw, bh, prev_label_path = get_predicted_labels(prev_img_path)

        if prev_label_path == "no_detections":            
            count += 1
        else:
            break
    
    
    
    # Get current labels
    current_img_path = os.path.join(img_dir, file_name+'.jpg')
    current_labels, _, _, _= get_predicted_labels(current_img_path)
    

    # Static labels excludes coords assigned for manipulation. 
    # Likely they were manipulated since previous state
    
    current_static_labels = get_static_labels(X_initial, X_final, current_labels)
    prev_static_labels = get_static_labels(X_initial, X_final, prev_labels)

    
    # Compute distance cost matrix
    drift_cost = distance_cost_matrix(prev_static_labels, current_static_labels)

    prev_idx, current_idx, cost = sort_assignment_col(drift_cost)
    
    

    #Calculate drift in x and y directions
    drift_arr = []

    for i in range(len(current_idx)):     

        current_coord = current_static_labels[current_idx[i]]
        prev_coord = prev_static_labels[prev_idx[i]]
        
        drift_arr.append(current_coord - prev_coord)
        
    
    
    drift = np.mean(np.asarray(drift_arr), axis = 0)
          
    corrected_coords = X_initial - np.asarray([drift])

    #print(X_initial, drift, corrected_coords)

    return drift, corrected_coords



def no_drift_correction(sxm_filepath, img_dir, X_initial, X_final):
    
    
    drift = np.asarray([0, 0])
          
    corrected_coords = X_initial - np.asarray([drift])

    #print(X_initial, drift, corrected_coords)

    return drift, corrected_coords


def drift_correction_anchor_label(sxm_filepath, img_dir, X_initial, X_final, anchor_idx = 0):

    """
    Computes drift in the scan w.r.to the previous scan
    The computation is by calculating the average distance moved by the "un-manipulated" molecules
    The "unmanipulated labels" are those which are far from the line of manipulation
    Linear assignment is used to correlate the molecules between the scan frames

    i/p:
        sxm_filepath: path of the scanned sxm file
        img_dir: image directory to save the corresponding images
        assigned_coords: coordinates of the molecules assigned for manipulation.(i.e., likely manipulated in the previous state)
    """

    sxm_name, file_name = filepath_to_filename(sxm_filepath)
    #sxm_dir = sxm_filepath.replace(sxm_name, '')
    
    
    
    # Get current labels
    current_img_path = os.path.join(img_dir, file_name+'.jpg')
    current_labels, _, _, _= get_predicted_labels(current_img_path)
    


    
    # Compute distance cost matrix
    drift_cost = distance_cost_matrix(current_labels, X_final[anchor_idx: anchor_idx+1])

    row, col, cost = sort_assignment_col(drift_cost)

    #sample index since row is a list    
    anchor_labels = np.asarray([current_labels[row_i] for row_i in row])


    drift = anchor_labels - X_final[anchor_idx]
    
    drift = np.mean(drift, axis = 0)

    corrected_coords = X_initial - np.asarray([drift])

    return drift, corrected_coords



def get_static_labels(X_initial, X_final, label_points, d_th = 0.2):

    """
    Includes the static label points that are far from the line of manipulation

    Args:
        X_intial: intial coordinate [x, y] of the manipulation  in range [0-1]
        X_final: final coordinate [x,y] of the manipulation in range [0-1]
        label_points are the list of label points of object prediction
        d_th : (default = 0.2) is the minimum distance fromany point on the line of manipulation
        
    Returns:
        static_label_points: 2D np.ndarray of the labels_points [x, y]
    """

    static_label_points = []
    m_line_points = interpolate_bw_points(X_initial[0], X_final[0])

    for l_point in label_points:
            
        is_static = True
        
        for m_line_point in m_line_points:
            if distance(l_point, m_line_point) <= d_th:
                is_static = False
                break

        if is_static == True:
            static_label_points.append(l_point)

    
    static_label_points = np.asarray(static_label_points)

    return static_label_points






def drift_correction_exclude_labels(sxm_filepath, img_dir, assigned_coords, target_coords):

    """
    Computes drift in the scan w.r.to the previous scan
    The computation is by calculating the average distance moved by the "un-manipulated" molecules
    Linear assignment is used to correlate the molecules between the scan frames

    i/p:
        sxm_filepath: path of the scanned sxm file
        img_dir: image directory to save the corresponding images
        assigned_coords: coordinates of the molecules assigned for manipulation.(i.e., likely manipulated in the previous state)
    """

    sxm_name, file_name = filepath_to_filename(sxm_filepath)
    sxm_dir = sxm_filepath.replace(sxm_name, '')

    # Get prev_image which has valid detections.
    while True:
        count = 1
        prev_img_path, prev_img_logged = get_prev_img_name(file_name, img_dir, precede = count)

        #Save the previous image if it doesn't exist
        if path_exists(prev_img_path) == False:
        
            prev_sxm_file = filepath_to_filename(prev_img_path)[1]+'.sxm'
            prev_sxm_path = os.path.join(sxm_dir, prev_sxm_file)
            prev_img_path,_ = save_sxm_image(prev_sxm_path, img_dir)

        
        # Get prev labels
        if prev_img_logged == False:
            prev_labels, bw, bh, prev_label_path = predict_labels(prev_img_path)
        else:
            prev_labels, bw, bh, prev_label_path = get_predicted_labels(prev_img_path)

        if prev_label_path == "no_detections":            
            count += 1
        else:
            break
    
    
    
    # Get current labels
    current_img_path = os.path.join(img_dir, file_name+'.jpg')
    current_labels, _, _, _= get_predicted_labels(current_img_path)

    # Static labels excludes coords assigned for manipulation. 
    # Likely they were manipulated since previous state
    
    static_labels =  []
    for coord in current_labels:
        if coord not in assigned_coords:
            static_labels.append(coord)
    
    static_labels = np.asarray(static_labels)

    # Compute distance cost matrix
    drift_cost = distance_cost_matrix(prev_labels, static_labels)

    prev_idx, current_idx, cost = sort_assignment_col(drift_cost)


    #Calculate drift in x and y directions
    drift_arr = []

    for i in range(len(current_idx)):     

        current_coord = static_labels[current_idx[i]]
        prev_coord = prev_labels[prev_idx[i]]
        drift_arr.append(current_coord - prev_coord)

    drift = np.mean(np.asarray(drift_arr), axis = 0)
    #print("drift", drift)
    # changing the convention of y since this is inverse for real space.
    
    
    corrected_coords = assigned_coords - drift

    #print(assigned_coords, drift, corrected_coords)

    return drift, corrected_coords


def point_within_margin(point, margin = 0.1):
    
    within_margin = False

    x_f, y_f = np.ravel(np.asarray(point))
    
    if x_f > margin and x_f < 1-margin:
        
        if y_f > margin and y_f < 1-margin:
            
            within_margin = True
            
    return within_margin









    














# def manipulation_coords_offline(sxm_file, x_target): # Not required
    
#     """
#     Provides the real initial and final coordinates for tip manipulation

#     Inputs:
#     sxm file corresponding with the molecules
#     X_target: 2D arrray of target positions with values normalized in range [0-1]
#     """    
      
    
#     # x_target is point in range 0-frame; provide array data - scalable to many atoms. This is a tensor
    

#     sxm_file_directory = r"E:\Ganesh\Manipulation_expts\routine1"
#     #sxm_file  = "Cu111_0246.sxm"
#     sxm_file = str(sxm_file)
#     file_name = sxm_file.split('.')[-2]
#     file_path  = sxm_file_directory +'/'+ str(sxm_file)
    
    
        
#     # Get scan_params
#     scan = Sxm_Image(file_path)
#     image = scan.image()
#     scan_frame = scan.frame
#     scan_offset =  scan.scan_offset
#     scan_angle = scan.scan_angle
    
    
        
#     # Save the image as a jpg
#     image, _, _ = norm_0to1(image)
#     image = cv2.resize(image, (256,256))
    
#     image_path = sxm_file_directory + '/images/'+ file_name + '.jpg'
#     save_scan_img(image_path, image)
    
    
#     # Predict using yolo
#     img_dir = sxm_file_directory + '/images'
#     #labels, bw, bh = predict_labels(image_path, img_dir)
#     labels, bw, bh, label_path = get_predicted_labels(image_path, img_dir)
#     detected_atoms = labels.shape[0]
    
#     #Save annotated image
#     save_annotated_img(sxm_file_directory, file_name, labels, bw, bh)
    
      
    
#     # Linear assignment
#     x_target = np.asarray(x_target)/scan_frame
#     initial_pos, final_pos, bw_pos, bh_pos = linear_assignment(labels, bw, bh, x_target)
    
    
#     # Save assigned image
#     save_assigned_img(sxm_file_directory, file_name, initial_pos, final_pos, bw_pos, bh_pos)
    
#     #save_observations(sxm_directory, file_name, initial_pos, final_pos)
#     #save_reward_value(sxm_directory, file_name, initial_pos, final_pos)
    
#     # Positions transformed to sample space
#     initial_positions = [] 
#     final_positions = []
    
#     for i in range(initial_pos.shape[0]):
        
#         initial_pos_i = initial_pos[i]
#         final_pos_i = final_pos[i]
        
#         #if y_convention is reversed
#         initial_pos_i[1] = 1 - initial_pos_i[1]
#         final_pos_i[1] = 1 - final_pos_i[1]
        
#         # Transform to range 0-frame
#         initial_pos_i = initial_pos_i* scan_frame
#         final_pos_i = final_pos_i* scan_frame
              

        
#         x_i = transform_point_array(initial_pos_i, scan_offset[0], scan_offset[1], scan_frame, scan_angle)        
#         x_f = transform_point_array(final_pos_i, scan_offset[0], scan_offset[1], scan_frame, scan_angle)
        
#         initial_positions.append(x_i)
#         final_positions.append(x_f)
        
           
#     return initial_positions, final_positions, detected_atoms, label_path






































# def dummy(sxm_file, x_target):
#     sxm_file = str(sxm_file)
    
#     #trial
#     x_target1 = np.asarray(x_target)
#     x_target1 = x_target1.tolist()  # Convert numpy array to list with python scalars
#     num1 = x_target1[0]
    
#     # x_target is point in range 0-frame; provide array data - scalable to many atoms. This is a tensor
    
#     sxm_file_directory = r"C:\Users\Administrator\Py_Scripts_ganesh\from laptop\1_CO on Cu images"
#     sxm_file  = "Cu111_0246.sxm"
#     sxm_file = str(sxm_file)
#     file_name = sxm_file.split('.')[-2]
#     file_path  = sxm_file_directory +'/'+ str(sxm_file)
    
#     # Get scan_params
#     scan = Sxm_Image(file_path)
#     image = scan.image()
#     scan_frame = scan.frame
#     scan_offset =  scan.scan_offset
#     scan_angle = scan.scan_angle
    
#     # Save the image as a jpg
#     image, _, _ = norm_0to1(image)
#     image = cv2.resize(image, (256,256))
    
#     image_path = sxm_file_directory + '/images/'+ file_name + '.jpg'
#     save_scan_img(image_path, image)
    
#     img_dir = sxm_file_directory + '/images'
#     #labels, bw, bh = predict_labels(image_path, img_dir)
#     labels, bw, bh = get_predicted_labels(image_path, img_dir)
#     detected_atoms = labels.shape[0]
    
#     #Save annotated image
#     save_annotated_img(sxm_file_directory, file_name, labels, bw, bh)
    
      
    
#     # Linear assignment
#     x_target = np.asarray(x_target)/scan_frame
#     initial_pos, final_pos, bw_pos, bh_pos = linear_assignment(labels, bw, bh, x_target)
    
    
#     # Save assigned image
#     save_assigned_img(sxm_file_directory, file_name, initial_pos, final_pos, bw_pos, bh_pos)
    
#     # Positions transformed to sample space
#     initial_positions = [] 
#     final_positions = []
    
#     for i in range(initial_pos.shape[0]):
        
#         x_i = transform_point_array(initial_pos[i], scan_offset[0], scan_offset[1], scan_frame, scan_angle)        
#         x_f = transform_point_array(final_pos[i], scan_offset[0], scan_offset[1], scan_frame, scan_angle)
        
#         initial_positions.append(x_i)
#         final_positions.append(x_f)
    
    
#     # Embed into the return array
#     return_array = []
#     #return_array.append(initial_positions)
#     #return_array.append(final_positions)
    
#     #num = x_target.shape[0]
#     #x_target = np.asarray(x_target)
    
#     #x_target = x_target.tolist()  # Convert numpy array to list with python scalars
    
#     num = [1,2,3]
#     #return_array.append(num)
#     return_array.append(num)
#     return num1


# #img_path = r"C:\Users\Administrator\Py_Scripts_ganesh\from laptop\1_CO on Cu images\Cu111_0246.sxm"
# #x_target = [[5E-9,5E-9], [2E-9, 2E-9], [7E-9, 8E-9]]

# #x_target = [[5E-9,5E-9]]

# #op = manipulation_coords("Cu111_0247.sxm", x_target)
# #print(op)


# #a, b, c, d = extract_labels(path)
# #print(a)





























