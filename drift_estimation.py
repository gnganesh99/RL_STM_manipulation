import os
import time
import numpy as np


from Manipulation_coords import save_sxm_image, distance_cost_matrix, get_prev_img_name, get_predicted_labels, sort_assignment

from expt_utils import interpolate_bw_points, distance, filepath_to_filename, path_exists, sort_by_order
#from Manipulation_coords import save_sxm_image, distance_cost_matrix, get_prev_img_name, get_predicted_labels, sort_assignment
from detect_molecules import predict_labels, get_predicted_labels, get_latest_image, check_prediction_logged, get_basename






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
    prev_static_labels = get_static_labels(X_initial, X_final, current_labels)

    # Compute distance cost matrix
    drift_cost = distance_cost_matrix(prev_static_labels, current_static_labels)

    prev_idx, current_idx, cost = sort_assignment(drift_cost)


    #Calculate drift in x and y directions
    drift_arr = []

    for i in range(len(current_idx)):     

        current_coord = current_static_labels[current_idx[i]]
        prev_coord = prev_labels[prev_idx[i]]
        drift_arr.append(current_coord - prev_coord)

    drift = np.mean(np.asarray(drift_arr), axis = 0)
    
    
    # changing the convention of y since this is inverse for real space.
   
    
    corrected_coords = X_initial - drift

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
        
        for m_line_point in m_line_points[0]:
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

    prev_idx, current_idx, cost = sort_assignment(drift_cost)


    #Calculate drift in x and y directions
    drift_arr = []

    for i in range(len(current_idx)):     

        current_coord = static_labels[current_idx[i]]
        prev_coord = prev_labels[prev_idx[i]]
        drift_arr.append(current_coord - prev_coord)

    drift = np.mean(np.asarray(drift_arr), axis = 0)
    
    
    
    corrected_coords = assigned_coords - drift
    
    print(assigned_coords, drift, corrected_coords)

    return drift, corrected_coords