# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:41:22 2024

@author: Administrator


"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy.spatial.distance import cosine
import glob
import shutil
import pathlib



def Rot_meshgrid_points(coordinate_point, frame_half, x_offset, y_offset, Angle):
    """
    Maps a point [x, y] to a rotated and translated meshgrid

    i/p:
      coordinate_point: coodinate [x,y]-point in range [0-frame]
      frame_half:  frame/2
      x_offset: x-offset center coordinates of the grid
      y_offset: y-offset center coordinates of the grid
      Angle: rotation in deg

    o/p:
        point_rot: transformed point [x,y]
    """    
    x_point = coordinate_point[0]
    y_point = coordinate_point[1]

    Rad = np.pi*Angle/180.
        
    Rot_Op = np.array([[np.cos(Rad), np.sin(Rad)], [-np.sin(Rad), np.cos(Rad)]])
    
    #X,Y = np.meshgrid(x_array, y_array)
    
    point_rot = []
         

    X_new = x_point - frame_half
    Y_new = y_point - frame_half
    X_element, Y_element = np.dot(Rot_Op, [X_new, Y_new])
    X_element = X_element + x_offset
    Y_element = Y_element + y_offset
    #X_new = x_array[i] + x_offset - get_frame_half
    #Y_new = y_array[i] + y_offset - get_frame_half
    #X_element, Y_element = np.dot(Rot_Op, [X_new, Y_new])
    
    point_rot.append(X_element)
    point_rot.append(Y_element)
    
    return point_rot



def transform_coord_array(point_arr, xc, yc, frame_length, rot_angle):
    
    """
    Transforms a point [x,y] from [0-frame] --> STM coords

    Outputs the transformed array
    """
    
    transform_point_array = []

    for point in point_arr:

        transform_point = Rot_meshgrid_points(point, frame_length/2, xc, yc, rot_angle)      
    
        transform_point_array.append(transform_point)
    
    transform_point_array = np.asarray(transform_point_array)

    return transform_point_array




def remove_nan_elements(array1):

    """
    Removes the nan elements from the array
    """
    array2 = []
    for element in array1:
        if np.isnan(element) == False:
            array2.append(element)
    return(array2)


def reverse_2D_y(img):
    
    """
    Reverses the y_axis of an image. 
    useful to correlate labview coords with real coords
    """
    
    img_yr = np.zeros(np.shape(img))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_yr[i, j] = img[(img.shape[0]-1)-i, j]
            
    return img_yr


def generate_xyz_arrays(data_f):
    """
    Outputs first three colomns of a pd dataframe
    """
    
    x = data_f.loc[:,0]
    y = data_f.loc[:,1]
    z = data_f.loc[:,2]
   

    x1 = np.asarray(x)
    y1 = np.asarray(y)
    z1 = np.asarray(z)
   

    xdata_f = remove_nan_elements(x1)
    ydata_f = remove_nan_elements(y1)
    zdata_f = remove_nan_elements(z1)
    
    return xdata_f, ydata_f, zdata_f


def twoD_to_1D(array_2D):
    array_op = np.asarray(array_2D)
    length = int(len(array_op)**2)
    
    array_op = np.reshape(array_op, (1, length))
    array_op = array_op[0]
    
    return array_op

def oneD_to_2D(array_1D):

    length = int(len(array_1D)**0.5)
    array_op = np.reshape(array_1D, (length,length))
    array_op = np.asarray(array_op)
    
    return array_op


def distance(p0, p1):
    """
    Outputs the distance between two points p0 and p1
    """
    x =  np.asarray([p0[0], p1[0]])
    y =  np.asarray([p0[1], p1[1]])                                     #x and y are array of length 2
    
    r_sq = ((y[1]-y[0])**2) + ((x[1]-x[0])**2)
    r = r_sq**0.5
    return r


# Reward based on distance.
def euclidean_distance(X_intial, X_final):

    """
    Computes the Euclidean distance vector of a multi-point coordinates
    i/p:
        X_initial: Array of X_intial coordinates of type [[x1, y1], [x2, y2]...]
        X_final: Array of X_final coordinates of type [[x1, y1], [x2, y2]...]

    o/p:
        d: averaged Euclidean distance of the points
        d_vector: vector of the euclidean distance for each of the point-pairs
    """

    Xi = np.asarray(X_intial)
    Xf = np.asarray(X_final)

    d_vector =  np.sqrt(np.sum((Xf - Xi)**2, axis = -1))
    d = np.mean(d_vector)
    

    return d, d_vector


def norm_0to1(arr):
    
    """
    Normalizes an array in the range 0-1

    Return: normalized array, factor, offset
    """
    arr = np.asarray(arr)
    factor = (np.max(arr) - np.min(arr))
    offset = np.min(arr)
    norm_arr = (arr - offset)/factor

    return norm_arr, factor, offset


def save_scan_img(img_path, image_array):
    """
    Saves the image_array in the image path.
    The image is saved w/o axis and in 'tight' configuration.
    """
    plt.figure(frameon=False)
    plt.axis('off')
    plt.imshow(image_array, origin= 'lower')
    plt.savefig(img_path, bbox_inches = 'tight', pad_inches=0)
    
    
def sort_by_order(y, order, reverse = False):
    '''
    Sorts both arrays, while sorting the first array based on the order of the second array.

    i/p:
        y: the secondary array
        order: the primary array that is ordered either in ascending or descending order
                np. array of int or float
        reverse = defaut(False) for ascending order

    O/p: Returns two arrays, the second array is ordered, while the first array follows the order of the second array.
        y_sorted: sorted based on ordering of the primary 
        order_sorted: ordered primary array
   
    '''

    sample_dict = {}
    y_sorted = []

    y = np.asarray(y)
    order = np.asarray(order)

    for i in range(order.shape[0]):
        sample_dict[order[i]] = y[i]
    
    if reverse == True: 
        #Sort Array in descending order 
        order_sorted =  np.sort(order)[::-1]  # Ascending followed by reversing the array
    else:
        #Sort Array in ascending order
        order_sorted =  np.sort(order)
    

    for key in order_sorted:
        y_sorted.append(sample_dict[key])

    
    return y_sorted, order_sorted

def translation_angle(X_initial, X_final):

    """
    Computes the angle of tip movement for manipulation from points X_initial to X_final
    Outputs degress in range [0-360] normalized to range [0-1]

    i/p:
        X_intiial: initial point coordinate [x, y]
        X_final: final point coordinate [x, y]

    o/p:
        norm_angle: float value in range [0-1]. Can be rescaled to range [0-360]

    """

    v =  np.asarray(X_final) - np.asarray(X_initial)
    v_x = [abs(v[0]), 0]

    # Cosine distance gives 1 - cosine similarity
    cos_sim = 1 - cosine(v, v_x)
    
    # Convert cosine similarity to angle in radians
    angle_radians = np.arccos(np.clip(cos_sim, -1.0, 1.0))
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    if v[1] < 0:
        angle_degrees  = 360 - angle_degrees

    norm_angle  = angle_degrees/360

    return norm_angle


def write_log(log_filepath, key_name, value):

    """
    Enters the log of parameter with the corresponding keyname

    i/p:
        log_filepath: path of the log file
        key_name: image name to record state
        value: state or action params
    """
    

    f = open(log_filepath, "a")

    entry = f'{key_name}\t{value}\n'
    f.write(str(entry))

    f.close()


def get_basename(filename, number_length = 4):

    """
    Separates the basename and the numeric from a given filename:

    i/p:
        filename: with or without the extension (eg: .sxm)
        number_legth: lengh of numbers in the number string

    O/p:
        basename: only the basename string
        basename_corr: appends "0" string to have 4 units in the numeric string
        number: number associated with the filename
    """
    
    arr = filename.split('.')[0]
    number_string = arr.split('_')[-1]
    basename = arr.replace(number_string, "")

    
    # add '0' id len(number_string)<4

    basename_corr = basename
    l = len(str(int(number_string)))
    if l < number_length:
        for i in range(number_length-l):
            basename_corr = basename_corr+str(0)



    return basename, basename_corr, int(number_string)


def get_sxm_filenames(dir) -> list:
    
    "Outputs the list of all the .sxm files in the given directory"

    file_paths = dir +'\*.sxm'
    saved_sxm_files = []

    for file_path in glob.glob(file_paths):
        filename  =  os.path.basename(file_path).split('/')[-1]
        #print(img_name)
        saved_sxm_files.append(filename)

    return saved_sxm_files



def get_latest_file(dir):
    
    """
    Returns the latest sxm file in the dir depending the highest value of the number_suffix of the filename

    o/p:
        latest_filename: string
    """

    saved_files = get_sxm_filenames(dir)

    file_index = []

    for file_name in saved_files:
        _, _, idx = get_basename(file_name)
        file_index.append(idx)

    file_names, file_index =  sort_by_order(saved_files, file_index, reverse = True) # reverse = True for desending value of the index

    return file_names[0]



def get_next_file_name(filename, dir):

    """
    Returns the next file name based on the numerical suffix.
    i/p:
        filename: filename string w/ or w/o the extension - sxm or jpg
        img-dir: directory of the image.

    O/p:
        prev_imag_path: image path of the previous image file.
        prev_img_logged: True if yolo prediction logged
    """

    _,basename , file_no = get_basename(filename)

    #Get previous scan-image
    next_file_name = get_basename(basename+str(file_no+1))[1] + str(get_basename(basename+str(file_no+1))[2]) + '.sxm'    
    next_file_path =  os.path.join(dir, next_file_name)


    return next_file_path


def copy_files(origin_file_path, target_dir):

    """
    Creates the copy of the file to the target directory

    i/p:
        origin_file_path:
        target_dir
    """

    file_name,_ = filepath_to_filename(origin_file_path)
    target_filepath = os.path.join(target_dir, file_name)

    shutil.copy(origin_file_path, target_filepath)


def filepath_to_filename(filepath):
    
    """
    Extracts the filename and primary filename from a filepath

    i/p: filepath

    o/p:
        filename: filename w/ the extension
        primary_filename: filename w/o the extension
    """

    filename = filepath.split('\\')[-1]
    primary_filename = filename.split('.')[0]

    return filename, primary_filename


def path_exists(path):
    """Test whether a path exists.  Returns False for broken symbolic links"""

    try:
        st = os.stat(path)
    except os.error:
        return False
    return True


def arr_to_linestring(arr, delimiter = '\t'):
    
    """
    Converts an array to a 1D array, finally into a linestring
    
    Args:
        arr : input array
        delimiter: default = '\t'
    
    Returns:
        linestring: str output seperated by delimiter
    """
    
    arr = np.asarray(arr)
    arr = np.ravel(arr)
    
    str_arr = list(map(str, arr))
    linestring = delimiter.join(str_arr)
    
    return linestring



def get_latest_folder(path):
    
    """
    Outputs the latest folder within a directory
    
    """
    
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    if not folders:
        return None

    latest_folder =  max(folders, key=lambda f: os.path.getmtime(os.path.join(path, f)))

    if latest_folder:
        return latest_folder
    else:
        return "No folders found in the directory"
    
    

def read_expt_log(log_path, delimit = '\t'):
    
    """
    Reads the expt log files
    
    Args:
        log_path of the experimental entries
        delimit: default = '\t'
        
    Returns:
        names: files names of the corresponding sxm files
        logs: the rest of the entries as float into a 2D nparray
        
    """
    
    f = open(log_path, "r")
    names = []
    logs = []

    
    for entry in f.readlines():
        
        # Rid of the end of string character
        entry = entry.replace('\n', "")

        #image_name
        img_name = entry.split(delimit)[0]

        #Rest of the entry
        log = entry.split(delimit)[1:]
        
        #Convert the strings to list
        log = list(map(float, log))

        names.append(img_name)
        logs.append(log)

    f.close()
    
    logs = np.asarray(logs)

    return names, logs


def interpolate_bw_points(X_0, X_1, n_points = 20):
    
    """
    Outputs interpolated points including the given coordinates

    Args:
        X_0: initial coordinate [x,y]
        X_1: final coordinate [x, y]
        n_points: (default = 20) is the number of interpolated points
    """


    x0, y0 = X_0
    x1, y1 = X_1

    x_points =  np.linspace(x0, x1, n_points)
    y_linspace =  np.linspace(y0, y1, n_points)

    itp_points = []
    
    if x1 == x0:
        infinite_slope = True
    else:
        infinite_slope = False
    
    
    
    for i in range(len(x_points)):
        
        if infinite_slope == False:
            y_point = y0 + ((y1-y0)/(x1-x0))*(x_points[i] - x0)
            itp_points.append([x_points[i], y_point])
            
        else:
            itp_points.append([x_points[i], y_linspace[i]])

    itp_points = np.asarray(itp_points)

    return itp_points



def read_log(log_path, delimit = '\t'):
    
    """
    Reads the log files
    
    Args:
        path of the log file
        delimit: default = '\t'
        
    Returns:
        logs: entries as float into a 2D nparray
        
    """
    
    f = open(log_path, "r")
    logs = []

    
    for entry in f.readlines():
        
        # Rid of the end of string character
        entry = entry.replace('\n', "")


        #Rest of the entry
        log = entry.split(delimit)
        
        #Convert the strings to list
        log = list(map(float, log))

        logs.append(log)

    f.close()
    
    logs = np.asarray(logs)

    return logs



def write_anchor_log(log_filepath, coords):

    """
    Enters the log of parameter with the corresponding keyname
    Overwrites if for the same filename

    i/p:
        log_filepath: path of the log file
        coords is a 2D array of the coordinate points [x, y]
    """
    
    coords = np.asarray(coords)

    f = open(log_filepath, "w")
    
    for coord in coords:
        #print(coord)
        entry = arr_to_linestring(coord)+'\n'
        #print(entry)
        f.write(entry)

    f.close()
    
    
    
    
def basename_exists(basename, filename, index_len = 4):

    filename = filename.split('.')[0]
    exists  =  False
    
    if basename in filename:

        num_idx = filename.replace(basename, '')
        

        if len(num_idx) == index_len:
            exists =  True
        
    return exists



def saved_next_sxmfilenames(expt_dir, basename, index_len = 4):
    filenames = get_sxm_filenames(expt_dir)
    
    max_idx = 0

    valid_filenames = []

    for filename in filenames:
        if basename_exists(basename, filename, index_len) == True:
            valid_filenames.append(filename)

    

    if len(valid_filenames) > 0:


        for filename in valid_filenames:
            filename = filename.split('.')[0]
        
            num_string = filename.replace(basename, '')
            idx = int(num_string)

            if idx > max_idx:
                max_idx = idx
                max_num_string = num_string

    else:
        max_num_string = ''
        for i in range(index_len):
            max_num_string += str(0)


    saved_filename = basename+max_num_string+'.sxm'

    next_num = int(max_idx+1)
    

    basename_corr = basename
    l = len(str(next_num))
    if l < index_len:
        for i in range(index_len-l):
            basename_corr = basename_corr+str(0)

    next_filename = basename_corr+str(next_num)+'.sxm'

    return saved_filename, next_filename, next_num


def sort_and_get_indices(array, reverse = False):

    """
    Sorts and returns a 1D array with its reordered original indices

    Args:
    array: input 1D array of numerical values
    reverse: (default = False) indicates ascending order
    """

    array = np.asarray(array)
    indexed_arr = [(value, index) for index, value in enumerate(array)]

    # Sort the array based on the values
    sorted_indexed_arr = sorted(indexed_arr, key=lambda x: x[0], reverse = reverse)

    # Extract the sorted values and their original indices
    sorted_values = [x[0] for x in sorted_indexed_arr]
    original_indices = [x[1] for x in sorted_indexed_arr]

    sorted_values = np.asarray(sorted_values)
    original_indices = np.asarray(original_indices)

    return sorted_values, original_indices
