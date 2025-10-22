# -*- coding: utf-8 -*-
"""
Functions to display the results of each iteration in the manipulation experiment.

@author: Ganesh Narasimha
"""

import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from expt_utils import path_exists, filepath_to_filename, saved_next_sxmfilenames
from Manipulation_coords import get_prev_img_name
from IPython.display import clear_output


def show_iter_results(expt_dir, iteration, action_params, prev_action_params, start_session, basename, reward, disp):


    """
    Help function to display the results of the current and previous iterations. Compiles the parameters and results into a dictionary.
    I/p:
        expt_dir: experiment directory
        iteration: current iteration number
        action_params: tuple of (bias, setpoint, speed) for current iteration
        prev_action_params: tuple of (bias, setpoint, speed) for previous iteration
        start_session: boolean indicating if a new session is started
        basename: base filename for the experiment
        reward: reward obtained in the previous iteration
        disp: normalized displacement obtained in the previous iteration
    O/p:
        results: dictionary containing previous and current iteration details
    """

    results = {}

    prev_iter_results = {}
    current_iter_results = {}

    if iteration > 1:
        print("Iteration:  ", iteration - 1)
        tb, tsp, ts = prev_action_params
        print(f"Action params: Bias = {tb:.3f} V,\tSetpoint = {tsp:.2f} nA,\tspeed = {ts:.2f} nm/s")
        print(f"Reward: {reward}, \tnorm_displacement: {disp}\n")
        #show_image_results(expt_dir, basename)

        prev_iter_results['iteration'] = iteration - 1
        prev_iter_results['action_params'] = prev_action_params
        prev_iter_results['reward'] = reward
        prev_iter_results['disp'] = disp

    
    print("\nCurrent iteration:  ", iteration)
    print("start new session: ", start_session)
    tb, tsp, ts = action_params
    print(f"Action params: Bias = {tb:.3f} V,\tSetpoint = {tsp:.2f} nA,\tspeed = {ts:.2f} nm/s \n")

    current_iter_results['iteration'] = iteration
    current_iter_results['action_params'] = action_params
    current_iter_results['start_session'] = start_session
    current_iter_results['reward'] = reward
    current_iter_results['disp'] = disp

    results['previous'] = prev_iter_results
    results['current'] = current_iter_results

    return results





def show_iter_image_results(expt_dir, iteration, action_params, prev_action_params, start_session, basename):

    """
    Help function to display the image results of the current and previous iterations.
    """
    
    # Show previous iteration details
    if iteration > 1:
    
        img_dir = os.path.join(expt_dir, 'images')
        
        while True:
            sxm_filename, _, _ =  saved_next_sxmfilenames(expt_dir, basename)
            print(sxm_filename)
            primary_name = sxm_filename.split('.')[0]
            image_name = primary_name+'.jpg'
            
            im3 = os.path.join(expt_dir, primary_name+'_assigned.jpg')
            #print(im3)
            if path_exists(im3) == True:
                break
            else:
                print("waiting for file...")
                time.sleep(1)
        #print(image_name, im3)
        
        while True:    
            precede = 1
            prev_img_path, logged = get_prev_img_name(image_name, img_dir, precede = precede)
            _, prev_img_name =  filepath_to_filename(prev_img_path)
            #print(prev_img_name)
            
            if logged == True:
                break
            else:
                precede += 1
        
        
        
        img3 = mpimg.imread(im3)
        img2 = mpimg.imread(os.path.join(expt_dir, prev_img_name + '_assigned.jpg'))
        img1 = mpimg.imread(os.path.join(expt_dir, prev_img_name + '_detectCO.jpg'))
        
        
        clear_output(wait = True)
        
        print("Iteration:  ", iteration - 1)
        tb, tsp, ts = prev_action_params
        print(f"Action params: Bias = {tb:.3f} V,\tSetpoint = {tsp:.2f} nA,\tspeed = {ts:.2f} nm/s \n")
        
        
        print("Current filename:", primary_name )
        
        fig, (ax1,ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 4));
        ax1.imshow(img1);
        ax1.set_title("Molecules detected");
        ax1.axis("off");
        ax2.set_title("Before manipulaton");
        ax2.imshow(img2);
        ax2.axis("off");
        ax3.imshow(img3);
        ax3.set_title("After manipulation");
        ax3.axis("off");
        
        plt.show()    
        
        
    # Show current iteration details    
    print("\nCurrent iteration:  ", iteration)
    print("start new session: ", start_session)
    tb, tsp, ts = action_params
    print(f"Action params: Bias = {tb:.3f} V,\tSetpoint = {tsp:.2f} nA,\tspeed = {ts:.2f} nm/s \n")
        
        
    
    return True
