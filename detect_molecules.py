# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:52:44 2024

@author: Administrator
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
import glob

import time
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
    
from expt_utils import transform_coord_array, twoD_to_1D, distance, save_scan_img, norm_0to1, sort_by_order, get_basename, filepath_to_filename, path_exists
from stm_utils import Sxm_Image

trained_model = r"C:\Users\Administrator\Py_Scripts_ganesh\manipulation_scripts_2\yolo_models\train40\weights\best.pt"


def predict_labels(img_path):
    
    """
    Predicts and outputs the object labels in the image.

    I/p:
        image path of the image on which the yolo-prediction is computed

    O/p:
        box_labels: 2D array of the coordinates [x, y] corresponding to each detected object
        bw: width array of the bounding box
        bh: height array of the bounding box
    """


    # get labels of the object labels!

    bx = []
    by = []
    bw = []
    bh = []    

    # Detect CO-molecule
    
    model_test = YOLO(trained_model)
    result = model_test.predict(source = img_path, save = True, save_txt =  True, iou = 0.1, conf = 0.4)
    
    
    #Extract labels
    image_name  =  os.path.basename(img_path).split('/')[-1]
    file_name = image_name.split('.')[-2]
    
    label_path = result[0].save_dir + "\\labels\\"+ file_name+ ".txt"
    
    print('label path is here:', label_path)
    
    labels_exist = path_exists(label_path)

    box = []
    bw = []
    bh = []

    if labels_exist == True:
    
        bx, by, bw, bh = extract_labels(label_path)   
    
        # converts to a 2D array where each position is an array element [[1], [2]], [3]]
        bx  =  np.reshape(bx, (len(bx), 1))  
        by  =  np.reshape(by, (len(by), 1))
        
        box = np.hstack((bx, by))

    else:
        label_path = "no_detections"

    # Save the label_path to label_log file
    dir = img_path.replace(image_name, "")
    write_label_log(dir, image_name, label_path)

    return box, bw, bh, label_path

def extract_labels(label_path):

    """
    Extracts the label parameters from the label file after the yolo prediction

    i/p:
        path of the label file
    o/p:
        bx, by: array of the center coordinates of the bounding boxes
        bw, bh: array of the width and height of the bounding boxes
    """
    
    #i_p = r"C:\Users\Administrator\Py_Scripts_ganesh\from laptop"
    #i_p = r"E:\labView\Ganesh\LV_programs_2018\v5_infinity\Py_Scripts"
    
    #label_path = os.path.join(i_p, label_path)
    
    f = open(label_path, "r")

    bx = []
    by = []
    bw = []
    bh = []

    for label in f.readlines():
        delimit = ' '
        x  = float(label.split(delimit)[1])
        y = float(label.split(delimit)[2]) 
        w = float(label.split(delimit)[3])
        h = float(label.split(delimit)[4])

        bx.append(x)
        by.append(y)
        bw.append(w)
        bh.append(h)
   
    f.close()

    bx = np.asarray(bx)
    by = np.asarray(by)
    bw = np.asarray(bw)
    bh = np.asarray(bh)

    return bx, by, bw, bh


def get_predicted_labels(img_path):
    
    """
    Extracts and outputs the object labels in the image given by img_path.
    In contrast to the prediction, this extracts the labels parameters from a previously predicted image by accessing the log of images and labels

    I/p:
        img_path: image path of the image on which the yolo-prediction is completed
        img_dir: directory of the images, where the label_log is present

    O/p:
        box_labels: 2D array of the coordinates [x, y] corresponding to each detected object
        bw: width array of the bounding box
        bh: height array of the bounding box
        label_path: associated with the image
    """
    

    img_name  =  os.path.basename(img_path).split('/')[-1]
    img_dir = img_path.replace(img_name, "")

    log_file = img_dir + 'label_log.txt'
    
    # get labels of the object labels!

    bx = []
    by = []
    bw = []
    bh = []    
    box = []
    
        
    search_log = True
    
    while search_log == True:

        log_images, log_labelpath = read_label_log(log_file)
        label_dict = {}
        
        for i in range(len(log_images)):
            
            label_dict[log_images[i]] = log_labelpath[i][:-1]  # removes the next line character
        
            label_path = label_dict.get(img_name)
        
        if label_path == None:
            
            time.sleep(2)
            
            
        else:
            search_log = False
    
   
    #print('label path is here:', label_path)

    if label_path != "no_detections":

        bx, by, bw, bh = extract_labels(label_path)   
    
        # converts to a 2D array where each position is an array element [[coord_1], [coord_2]], [coord_3]]
        bx  =  np.reshape(bx, (len(bx), 1))  
        by  =  np.reshape(by, (len(by), 1))
        
        box = np.hstack((bx, by))
    
    
    return box, bw, bh, label_path


def write_label_log(dir, img_name, label_path):

    """
    Enters the image name and the corresponding label_path into the log file

    i/p:
        log_file: path of the log file
        img_name: image file name on which the prediction is performed
        label_path: path of the label_file after prediction
    """
    log_file = dir + '\label_log.txt'

    f = open(log_file, "a")

    entry = f'{img_name}\t{label_path}\n'
    f.write(str(entry))

    f.close()


def read_label_log(log_file) -> tuple[list, list]:

    """
    Reads the label log file and outputs the list of the image-names and the label-paths
    """

    f = open(log_file, "r")
    images = []
    labels = []

    
    for entry in f.readlines():
        delimit = '\t'
        
        img_name = entry.split(delimit)[0]
        label_path = entry.split(delimit)[1]

        images.append(img_name)
        labels.append(label_path)

    f.close()

    return images, labels


def get_saved_img_names(dir) -> list:
    
    "Outputs the list of all the .jpg images in the given directory"

    img_dir = dir +'\*.jpg'
    saved_images = []

    for img_path in glob.glob(img_dir):
        img_name  =  os.path.basename(img_path).split('/')[-1]
        #print(img_name)
        saved_images.append(img_name)

    return saved_images





def get_missing_images(ls1, ls2) -> list:

    """
    Compares lists and outputs the set of missing elements that are present in ls1 and not in ls2
    ls1 should be the superset
    """
    
    m_ls = []
    for element in ls1:
        if element not in ls2:
            m_ls.append(element)

    return m_ls






def get_latest_image(dir):
    
    """
    Returns the latest image in the dir depending the highest value of the number_suffix of the filename

    o/p:
        latest_filename: string
        predcition_logged: True if yolo-prediction is logged within the label_log
    """

    log_file = dir + '\label_log.txt'

    saved_images = get_saved_img_names(dir)
    logged_images, _ = read_label_log(log_file)

    img_index = []

    for image_name in saved_images:
        _, _, img_idx = get_basename(image_name)
        img_index.append(img_idx)

    img_names, img_index =  sort_by_order(saved_images, img_index, reverse = True) # reverse = True for desending value of the index

    # Check if prediction is logged for the latest image
    prediction_logged = img_names[0] in logged_images

    return img_names[0], prediction_logged




    
def check_prediction_logged(img_path):

    """
    Checks if the yolo-prediciton of the image_path is logged into the label_log file

    Returns True if logged
    """

    image_name,_ =  filepath_to_filename(img_path)
    img_dir = img_path.replace(image_name, '')

    log_file = img_dir + 'label_log.txt'
    logged_images, _ = read_label_log(log_file)

    prediction_logged = image_name in logged_images

    return prediction_logged