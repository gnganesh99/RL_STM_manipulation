"""
Programs to get target coordinates


@author: Ganesh Narasimha
"""


# target functions
import numpy as np
import random
import os
from expt_utils import distance, sort_by_order, sort_and_get_indices


class Random_Target():
    """
    Class to generate random target coordinates within a unit square frame.
    I/p:
        size: number of target points to generate, default is 1
        margin: margin from the edges of the frame to avoid placing points too close to the borders, default is 0.0
    """

    def __init__(self, size = 1, margin = 0.0):

        self.size = int(max(1, size))
        self.margin = margin

    def generate_point(self):
        """
        Generate a single random point within the unit square frame, considering the margin.
        """

        while True:
            x =  random.uniform(0, 1)
            y = random.uniform(0, 1)

            point = [x, y]

            if point_within_margin(point, margin = self.margin):
                return point

    def __call__(self):
        
        X_target = []
        for _ in range(self.size):
            point = self.generate_point()
            X_target.append(point)
        X_target = np.asarray(X_target)

        return X_target
    
    def min_distance(self, x_ref = [0.5, 0.5], dmin = 0):
        """
        Generate target points ensuring a minimum distance from a reference point.
        I/p:
            x_ref: reference point to maintain minimum distance from, default is center [0.5, 0.5]
            dmin: minimum distance to maintain from the reference point, default is 0
        O/p:
            X_target: array of target points satisfying the minimum distance condition
        """

        
        x_ref = np.ravel(np.asarray(x_ref))

        X_target = []

        for _ in range(self.size):
            while True:
                point = self.generate_point()
                dist = np.linalg.norm(point - x_ref)
                if dist >= dmin:
                    break
            X_target.append(point)
        X_target = np.asarray(X_target)

        return X_target
    
    def constant_distance(self, x_ref = [0.5, 0.5], d = 0.3):

        """
        Generate target points at a constant distance from a reference point. Here the angles are sampled randomly.
        I/p:
            x_ref: reference point to maintain constant distance from, default is center [0.5, 0.5]
            d: constant distance to maintain from the reference point, default is 0.3
        O/p:
            X_target: array of target points at the specified constant distance
        """
        
        x_ref = np.ravel(np.asarray(x_ref))

        X_target = []

        for _ in range(self.size):
            while True:
                angle = random.uniform(0, 2*np.pi)
                x = x_ref[0] + d * np.cos(angle)
                y = x_ref[1] + d * np.sin(angle)
                point = [x, y]
                if point_within_margin(point, margin = self.margin):
                    break
            X_target.append(point)

        X_target = np.asarray(X_target)
 
        return X_target



def point_within_margin(point, margin = 0.1):
    """
    Check if a point is within the specified margin from the edges of the unit square frame.
    I/p:
        point: coordinates of the point to check
        margin: margin from the edges of the frame, default is 0.1
    O/p:
        within_margin: boolean indicating if the point is within the margin    
    """
    
    within_margin = False

    x_f, y_f = np.ravel(np.asarray(point))
    
    if x_f > margin and x_f < 1-margin:
        
        if y_f > margin and y_f < 1-margin:
            
            within_margin = True
            
    return within_margin

    




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

    """
    Generate target points to clear a region by placing points near the edges of the unit square frame.
    I/p:
        n_objects: number of target points to generate
        margin: margin from the edges of the frame, default is 0.3

    O/p:
        target_points: array of generated target points
    """

    
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
    """
    Generate target points to clear a region by placing points across multiple levels starting from the edges and moving inward.
    I/p:
        n_objects: number of target points to generate
        limit: initial limit for the outermost level, default is 7. subsequent levels reduce this limit by 2
        margin: margin from the edges of the frame for each level, default is 0.1

    O/p:
        targs: array of generated target points across multiple levels
    """



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



def sort_target_to_center(X_target, to_center = True, center_coord = [0.5, 0.5]):

    """
    Sorts target points based on their distance to a center coordinate.
    I/p:
        X_target: array of target points
        to_center: boolean indicating sorting order; True for closest to center first, False for farthest first. Default is True
        center_coord: coordinates of the center point, default is [0.5, 0.5]

    O/p:
        sorted_target: array of sorted target points
    """

    
    
    center_coord = np.asarray(center_coord)
    dist_arr = []

    for point in X_target:

        dist = distance(point, center_coord)
        dist_arr.append(dist)

    dist_arr = np.asarray(dist_arr)
    
    sorted_dist_arr, sorted_indices = sort_and_get_indices(dist_arr, reverse = not to_center)

    sorted_target = [X_target[idx] for idx in sorted_indices]

    sorted_target = np.asarray(sorted_target)

    return sorted_target



def y_inv(X_target):

    points  = []

    for point in X_target:
        x, y = point
        points.append([x, 1-y])

    points = np.asarray(points)

    return points


def translate(points, offset):
    new_points = []

    for point in points:
        new_point = [point[0]+offset[0], point[1]+offset[1]]
        new_points.append(new_point)
    
    new_points = np.asarray(new_points)
    return new_points

def compute_coordinates(distance, theta):

    x = distance * np.cos(theta)
    y = distance * np.sin(theta)

    return x, y


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






















def square_target(size = 4, width = 0.5):
        
    x = random.uniform(0, 1)
    
    pass

def circular_target(size = 4):
    pass

# test_def = True




# if test_def == True:   
    
#     op = random_target(2)
#     print(op)
