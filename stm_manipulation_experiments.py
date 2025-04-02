# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:38:14 2024

@author: Administrator
"""

import os
import numpy as np

from experimental_routines import LV_STM
import time
from detect_molecules import get_latest_image
from expt_utils import transform_coord_array, translation_angle, write_log, get_latest_file, get_next_file_name, copy_files, get_sxm_filenames, read_expt_log



from nanonis_TCP import Nanonis_TCP

def sxm_filename_LV(basename, sxm_dir):
    
    e1 = LV_STM()
    saved_filename, new_file_name = e1.get_filename(sxm_dir, basename) 

    #saved_filename = sxm_filename_LV_dummy(basename, sxm_dir)
    
    return saved_filename


def do_manipulation_LV(initial_coords, final_coords, action_params, default_params, sxm_dir):

    e1 = LV_STM()
    
    e1.manipulate_and_scan(initial_coords, final_coords, action_params, default_params)
    
    time.sleep(2)
    
    #do_manipulation_LV_dummy(initial_coords, final_coords, default_action_params, action_params, sxm_dir)
    
    return True

def rescan_LV(sxm_dir):
    
    e1 = LV_STM()
    
    e1.scan()
    
    time.sleep(1)
    
    #rescan_LV_dummy(sxm_dir)

    return






# Functions using Nanonis TCP


def rescan_tcp():
    e1 = Nanonis_TCP()
    s_done = False
    
    print("\n#####\tScanning....\t #####")
    
    while not s_done:
        
        try:
            s_done = e1.scan()
        except:
            pass
    
    print("\tscan complete\n")
    return s_done



def manipulation_and_scan_tcp(initial_coords, final_coords, manipulation_params, default_params, 
                              sxm_dir, scan_params, drift, expt_name,
                              drift_compensation = False, target_reached = False):
    
    if target_reached:
        
        print("\nTarget reached !!!\n")
        return True
    
    # Manipulate if target not reached
    m_tb, m_tsp, m_ts = manipulation_params
    d_tb, d_tsp, d_ts = default_params
    
    modified_action_params = [m_tb, m_tsp*1E-9, m_ts*1E-9]
    
    modified_default_params =  [d_tb, d_tsp*1E-9, d_ts*1E-9]
    
    
    


    e1 = Nanonis_TCP()
    print("\n#####\tManipulation in progress....\t#####")
    
    m_done = False
    
    while not m_done:
        
        try:
            #print(initial_coords[0], final_coords[0], modified_action_params, modified_default_params)
            
            # point coords are converted from 2D to 1D for manipulation function.
            m_done = e1.manipulation(initial_coords = initial_coords[0], final_coords=final_coords[0], manipulation_params=modified_action_params, 
                           default_params = modified_default_params, bias_slew_rate=0.033, setpoint_slew_rate=80E-9)
        except:
            pass
        
        
    if drift_compensation == True:
        
        frame_size, frame_center_coords, scan_angle = scan_params
        
        
        # image coordinate has y- inverse relation to real space
        delta_x, delta_y = drift
        
        drift_yinv = np.asarray([delta_x, -delta_y])        

        
        frame_drift = drift_yinv*frame_size
        
        #print(drift, frame_drift)
        
        # Add the drift to the tip position. This is because tip /drame direction is opposite. if the tip drifts in -x, objects drift in x.
        center_drift = np.asarray([frame_size/2, frame_size/2]) + frame_drift
        
        new_center_coords = transform_coord_array([center_drift], frame_center_coords[0], frame_center_coords[1], frame_size, scan_angle)
        
        
        # Get average drift vals
        drift_log_path = os.path.join(sxm_dir, 'expt_log', expt_name, 'drift_log.txt')
        _, drift_log = read_expt_log(drift_log_path)
        
        if len(drift_log) > 1:
            avg_x_drift = np.mean(abs(drift_log[:, 0]))
            avg_y_drift = np.mean(abs(drift_log[:, 1]))
            
        else:
            avg_x_drift = 0
            avg_y_drift = 0
            
            
        # if drift is < 1.5 * average drift.
        if abs(delta_x) < 3*avg_x_drift and abs(delta_y) < 3*avg_y_drift:            
            
                       
            try:
                _, _, frame_x, frame_y, angle = e1.scan_Frameget()
                
    
                
            finally:
                time.sleep(1)
                
            
            print("Compensating drift, moving frame to:  ", new_center_coords)
            
            center_x, center_y = np.ravel(new_center_coords)
            try:
                e1.scan_Frameset(center_x, center_y, frame_x, frame_y, angle)
                    
            except:
                pass
            
            finally:
                time.sleep(2)


    print("\n#####\tScanning....\t#####")
    s_done = False

    while (not s_done) and m_done:

        try:    
            s_done = e1.scan()
        except:
            pass
    
    
    print("\tScanning complete\n")

    return s_done
    






















def sxm_filename_LV_dummy(basename, sxm_dir):
    
    filename = get_latest_file(sxm_dir)

    return filename


def do_manipulation_LV_dummy(initial_coords, final_coords, default_action_params, action_params, sxm_dir):

    filename = get_latest_file(sxm_dir) 
    
    origin_dir = r"E:\Ganesh\Manipulation_expts\routine1"
    next_filepath = get_next_file_name(filename, origin_dir)

    copy_files(next_filepath, sxm_dir)

    time.sleep(5)

    return True

def rescan_LV_dummy(sxm_dir):
    filename = get_latest_file(sxm_dir) 
    
    origin_dir = r"E:\Ganesh\Manipulation_expts\routine1"
    next_filepath = get_next_file_name(filename, origin_dir)

    copy_files(next_filepath, sxm_dir)

    time.sleep(5)

    return True