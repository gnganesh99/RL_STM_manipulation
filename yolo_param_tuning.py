"""
Tune YOLO model parameters (confidence threshold and IoU threshold) for molecule detection using Optuna.

@author: Ganesh Narasimha
"""



import numpy as np
import optuna
import os
from PIL import Image
from scipy.optimize import linear_sum_assignment
from expt_utils import get_basename



def tune_conf_iou_for_image(yolo_model, image, n_trials=40, timeout=None, seed=None, label_dict = None):
    """
    Tune YOLO model parameters (confidence threshold and IoU threshold) for molecule detection using Optuna.
    i/p:
        yolo_model: Pre-trained YOLO model for molecule detection
        image: Path to the image file for tuning
        n_trials: Number of optimization trials to run, default is 40
        timeout: Maximum time (in seconds) for optimization, default is None (no timeout)
        seed: Random seed for reproducibility, default is None
        label_dict: Optional dictionary containing 'target_n' (expected number of molecules) and 'use_prev' (boolean to use previous labels), default is None
    O/p:
        best_params: Dictionary containing the best 'conf' and 'iou' values found
        best_value: Best loss value achieved during optimization
    """

    # Do a initial prediction to get target_n, if label_dict not provided.
    results = yolo_model.predict(source = image, save = False, save_txt =  False, iou = 0.1, conf = 0.2)
    target_n = len(results[0].boxes)


    prev_label_path = get_prev_label_path(image)
    if label_dict is not None:
        target_n = label_dict.get('target_n', target_n)
        use_prev = label_dict.get('use_prev', False)    

    def objective(trial):
        conf = trial.suggest_float("conf", 0.01, 0.85)
        iou  = trial.suggest_float("iou", 0.1,  0.85)
        return per_image_loss(detect_fn, yolo_model, image, conf, iou, target_n, prev_label_path, use_prev)



    print("Optimizing YOLO parameters for detecting {} molecules...".format(target_n))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params, study.best_value




def get_prev_label_path(image_path):
    """
    Get the path to the previous label file corresponding to the given image.
    The previous label file is determined based on the numbering of the image files.
    i/p:
        image_path: Path to the current image file
    O/p:
        prev_label_path: Path to the previous label file, or None if not found
    """

    img_dir = os.path.dirname(image_path)
    image_name  =  os.path.basename(image_path).split('/')[-1]


    basename, _, num = get_basename(image_name)
    num =  num - 1 if num > 0 else 0
    prev_name = basename + str(num)
    prev_name = get_basename(prev_name)[1] + str(get_basename(prev_name)[2]) + ".jpg"


    # Read the label_log file
    log_file = os.path.join(img_dir, "label_log.txt")

    log_images, log_labelpath = read_label_log(log_file)
    label_dict = {}
    
    for i in range(len(log_images)):
        
        label_dict[log_images[i]] = log_labelpath[i][:-1]  # removes the next line character
    
    prev_label_path = label_dict.get(prev_name)

    return prev_label_path



def per_image_loss(detect_fn, yolo_model, image, conf, iou, target_n, prev_label_path=None, use_prev=False):
    """
    Computes the loss for a single image based on YOLO detections.
    Loss components include count error, overlap penalty, distance penalty from previous labels, and center molecule penalty.
    i/p:
        detect_fn: Function to perform detection using the YOLO model
        yolo_model: Pre-trained YOLO model for molecule detection
        image: Path to the image file
        conf: Confidence threshold for detection
        iou: IoU threshold for detection
        target_n: Expected number of molecules in the image
        prev_label_path: Path to the previous label file, default is None
        use_prev: Boolean indicating whether to use previous labels for distance penalty, default is False
    O/p:
        loss: Computed loss value (float)
    
    """
    boxes = detect_fn(yolo_model, image, conf=conf, iou=iou)  # your model call; return Nx4
    k = len(boxes)
    count_err = (k - target_n)**2
    dup_pen = overlap_penalty(boxes)

    distance_penalty = distance_pen(prev_label_path, boxes, target_n)
    ctr_pen = center_molecule_pen(boxes)

    return count_err + 2*dup_pen + 0.5*float(use_prev)*distance_penalty + 0.5*ctr_pen




def overlap_penalty(boxes_xyxy, iou_thresh=0.6):
    """Penalize highly-overlapping pairs (duplicate counting)."""

    n = len(boxes_xyxy)                   # Number of bounding boxes
    if n <= 1:                            # If there's 0 or 1 box, no overlap possible
        return 0.0

    # Ensure float for safe division operations
    b = boxes_xyxy.astype(float)

    # Compute area of each bounding box: (x2 - x1) * (y2 - y1)
    areas = np.clip(b[:,2] - b[:,0], 0, None) * np.clip(b[:,3] - b[:,1], 0, None)

    pen = 0.0                             # Accumulator for overlap penalty

    # Loop through each unique box pair (i, j>i) to compute pairwise IoU
    for i in range(n - 1):
        # Compute intersection rectangle between box i and all boxes after it
        xx1 = np.maximum(b[i, 0], b[i+1:, 0])   # left edge = max of x1s
        yy1 = np.maximum(b[i, 1], b[i+1:, 1])   # top edge  = max of y1s
        xx2 = np.minimum(b[i, 2], b[i+1:, 2])   # right edge = min of x2s
        yy2 = np.minimum(b[i, 3], b[i+1:, 3])   # bottom edge = min of y2s

        # Compute intersection area
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)

        # Compute Intersection-over-Union (IoU) between box i and each later box
        iou = inter / (areas[i] + areas[i+1:] - inter + 1e-9)

        # Count how many IoUs exceed the threshold (strong overlaps)
        pen += (iou > iou_thresh).sum()

    # Normalize by the number of unique box pairs (n choose 2)
    denom = n * (n - 1) / 2
    return pen / max(denom, 1)            # Return average fraction of overlapping pairs




def center_molecule_pen(boxes_xyxy):
    """Penalize molecules detected near the center of the image."""
    ctr_pt = np.asarray([[0.5, 0.5]])
    x_center = (boxes_xyxy[:,0] + boxes_xyxy[:,2]) / 2.0
    y_center = (boxes_xyxy[:,1] + boxes_xyxy[:,3]) / 2.0
    curr_labels = np.asarray([x_center, y_center]).T
    
    ctr_lbl, ctr_pt = linear_assignment_labels(curr_labels, ctr_pt)

    dist = ctr_lbl - ctr_pt
    dist = np.sqrt( dist[:,0]**2 + dist[:,1]**2 )
    return np.mean(dist)



def detect_fn(yolo_model, image, conf, iou):
    """Perform detection on the image using the YOLO model with specified confidence and IoU thresholds."""
    results = yolo_model.predict(source = image, save = False, save_txt =  False, iou = iou, conf = conf) 
    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()  # Nx4 array    
    boxes_xyxy = normalize_boxes(boxes_xyxy, image) # normalize to [0,1]

    return boxes_xyxy




def distance_pen(prev_label_path, boxes_xyxy, target_n):
    """Compute distance penalty from previous labels to current detections."""

    if prev_label_path is None or os.path.exists(prev_label_path) == False:
        return 0.0

    with open(prev_label_path, "r") as f:
        bx, by = [], []
        for line in f:
            parts = line.strip().split()
            bx.append(float(parts[1]))
            by.append(float(parts[2]))
    
    prev_labels = np.asarray([bx, by]).T    
    n_prev = len(prev_labels)
    n_curr = len(boxes_xyxy)

    if n_prev != target_n: #Probably the previous labels are not correct, don't apply distance penalty
        return 0.0

    if n_prev != n_curr:
        return (n_prev - n_curr)**2
        
    if n_prev == n_curr:

        x_center = (boxes_xyxy[:,0] + boxes_xyxy[:,2]) / 2.0
        y_center = (boxes_xyxy[:,1] + boxes_xyxy[:,3]) / 2.0
        curr_labels = np.asarray([x_center, y_center]).T
        
        prev_labels, curr_labels = linear_assignment_labels(prev_labels, curr_labels)

        dist = curr_labels - prev_labels
        dist = np.sqrt( dist[:,0]**2 + dist[:,1]**2 )
        return np.mean(dist)


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



def linear_assignment_labels(x_current, x_target) -> np.ndarray:
    """Perform linear assignment between current and target(previous) labels using the Hungarian algorithm."""
    
    
    cost = distance_cost_matrix(x_current, x_target)

        
    row, col = linear_sum_assignment(cost)   
 
       
    return x_current[row], x_target[col]


def distance_cost_matrix(X_initial, X_final):
    """Compute the cost matrix based on Euclidean distances between initial and final positions."""
    cost = []
       
    for initial_position in X_initial:
        # distance matrix for a given initial_position
        distance_i = []
        
        for final_position in X_final:
            
            #Compute Euclidean distance
            dis =  np.sqrt( (initial_position[0] - final_position[0])**2 + (initial_position[1] - final_position[1])**2 )
            distance_i.append(dis)
            
        cost.append(distance_i)
    
    cost = np.asarray(cost)

    return cost



def normalize_boxes(boxes, img_path):
    """Normalizes from pixel to [0,1]. both xyxy and xywh formats supported."""

    w, h = Image.open(img_path).size
    b = boxes.copy().astype(float)
    b[:, [0,2]] /= w
    b[:, [1,3]] /= h
    return b