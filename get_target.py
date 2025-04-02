# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:56:18 2024

@author: Administrator
"""


"""
programs to get target coordinates
"""


# target functions
import numpy as np
import random
import os
from expt_utils import distance, sort_by_order, sort_and_get_indices

def random_target(size = 1):

    """
    Returns target coordinates [x,y] in range [0-1]

    Args:
        size: default (size = 1), determines the no of coordinates per target state.
    """
    
    X_target = []
    
    if size < 1:
        size = 1
    
    size = int(size)
    
    for i in range(size):
        
        x =  random.uniform(0, 1)
        y = random.uniform(0, 1)
        
        X_target.append([x, y])
    
    X_target = np.asarray(X_target)

    return X_target



    
def custom_target(file_name):

    """
    Reads and outputs the target coordinates saved in a file.
    I/p:
        file_name: filename that contains custom target coordinates 
    """

    #i_p = r"E:\labView\Ganesh\LV_programs_2018\v5_infinity\Py_Scripts\target_structures"

    #target_file =  os.path.join('.', file_name)
    
    f = open(file_name, "r")
    
    target = []
    
    for line in f.readlines():
        
        point = line.split('\t')
    
        x= float(point[0])
        y = float(point[1])
    
        target.append([x,y])
    
    f.close()
    target =  np.asarray(target)
    
    return target




def clear_region(n_objects = 12, margin = 0.3):

    
    n_sides = int(np.ceil((n_objects+4)//4))
    
    divs = np.linspace(margin, 1-margin, n_sides)
    

    points = []
    for element in divs:
        points.append([element, margin])
        points.append([element, 1-margin])
        points.append([margin, element])
        points.append([1-margin, element])    

    points = np.asarray(points)

    target_points =  np.unique(points, axis = 0)

    if n_objects < target_points.shape[0]:
        target_points = target_points[0:n_objects, :]

    return target_points


def clear_region_levels(n_objects, limit = 7, margin = 0.1):
    
    n_left = n_objects

    level = 0

    level_margin  = margin

    while n_left > 0:
        
        level += 1
    
        n_obj_level = 4*limit - 4  

        #print(n_obj_level)
        
        if n_obj_level > n_left:
            n_obj_level = n_left

        
        
        level_targs = clear_region(n_obj_level, margin = level_margin)

        limit -= 2
    
        level_margin += margin


        n_left = n_left - n_obj_level


        if level == 1:

            targs = level_targs
        else:

            targs = np.vstack((targs, level_targs))
    
        #print(limit, n_left, targs, margin)
    targs =  np.asarray(targs)[0:n_objects, :]

    return targs



def sort_target_to_center(X_target, to_center = True):
    
    center_coord = np.asarray([0.5, 0.5])
    dist_arr = []

    for point in X_target:

        dist = distance(point, center_coord)
        dist_arr.append(dist)

    dist_arr = np.asarray(dist_arr)
    
    sorted_dist_arr, sorted_indices = sort_and_get_indices(dist_arr, reverse = not to_center)

    sorted_target = [X_target[idx] for idx in sorted_indices]

    sorted_target = np.asarray(sorted_target)

    return sorted_target




def compute_coordinates(distance, theta):

    x = distance * np.cos(theta)
    y = distance * np.sin(theta)

    return x, y

def translate(points, offset):
    new_points = []

    for point in points:
        new_point = [point[0]+offset[0], point[1]+offset[1]]
        new_points.append(new_point)
    
    new_points = np.asarray(new_points)
    return new_points

def hexagon_set(point, a):
    angles = np.radians([0, 60, 120, 180, 240, 300])
    
    hex_points = []
    for angle in angles:
        x, y = compute_coordinates(a, angle)
        new_point = [point[0]+x, point[1]+y]
        hex_points.append(new_point)
    
    hex_points = np.asarray(hex_points)
    # Add the central point at the beginning
    hex_points = np.vstack((point, hex_points))
    hex_points = np.asarray(hex_points)
    return hex_points


def hexagon_ring_points(rings = 1, a = 1, center = [0, 0], frame_size = 1):

    seed_point = [0, 0]

    all_hex_points = []

    for ring in range(rings):
        if ring == 0:
            all_hex_points = hexagon_set(seed_point, a)
        else:
            new_hex_points = []
            for point in all_hex_points:
                new_hex_points.append(hexagon_set(point, a))
            
            all_hex_points = np.concatenate(new_hex_points, axis = 0)
            all_hex_points = np.round(all_hex_points, 4)
            all_hex_points = np.unique(all_hex_points, axis = 0)
  
    all_hex_points = all_hex_points/frame_size
    all_hex_points = translate(all_hex_points, center)

    return all_hex_points
    


























def y_inv(X_target):

    points  = []

    for point in X_target:
        x, y = point
        points.append([x, 1-y])

    points = np.asarray(points)

    return points




def square_target(size = 4, width = 0.5):
        
    x = random.uniform(0, 1)
    
    pass

def circular_target(size = 4):
    pass

# test_def = True




# if test_def == True:   
    
#     op = random_target(2)
#     print(op)
