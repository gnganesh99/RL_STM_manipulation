
import gym


from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import time
from IPython.display import clear_output
import os
import numpy as np

from Manipulation_coords import get_manipulation_coords, point_within_margin
from expt_utils import transform_coord_array, translation_angle, write_log, get_latest_file, get_next_file_name, copy_files, get_sxm_filenames, arr_to_linestring, euclidean_distance
from detect_molecules import get_latest_image
from stm_manipulation_experiments import manipulation_and_scan_tcp, rescan_tcp, do_manipulation_LV, rescan_LV
from display_results import show_iter_image_results, show_iter_results
from drift_estimation import drift_correction_far_labels
from get_reward import get_displacement_reward, get_state_reward, get_state_reward_vector
from get_target import random_target, compute_coordinates

from env_functions import *


class Manipulation_multi_object(Env):

    def __init__(self, action_range, default_action_params, expt_name, sxm_basename, expt_dir, X_target, n_transitions = 10, continue_to_complete = True, drift_comp = False, reward_tolerance = 0.03, manipulation_offset = False, anchor = (False, 0), label_margin = 0):

        
        #Observation space is no_of_points/2
        self.observation_space = Box(low= 0 ,high = 1,shape=(5,))

        #action parameters = Bias, Sepoint, speed  
        self.action_space =  Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]))

        self.info = {}
        
        #State variables

        self.basename = sxm_basename
        
        self.X_target = np.asarray(X_target) #self.target is the sorted targets based on the object detection and linear assignment. X_target is the intial set of targets

        self.n_objects = self.X_target.shape[0]

        self.obj_idx = 0

        self.expt_dir = expt_dir


        #Other variables
        self.action_range =  action_range
    
        self.n_transitions = n_transitions
             
        self.default_action_params = default_action_params
    
        self.iterations = n_transitions

        self.expt_name = expt_name
        
        self.start_session = True
        
        self.iteration_count = 0
        
        self.continue_to_complete = continue_to_complete
        
        self.prev_action_params = []
        
        
        self.anchor_drift, self.anchor_idx = anchor
        
        self.drift_comp = drift_comp
        
        self.reward_vec =  np.zeros(shape = (self.n_objects,))
        
        self.reward = np.sum(self.reward_vec)
        
        self.reward_tolerance = reward_tolerance
        
        self.disp = 0

        self.manipulation_offset = manipulation_offset
        
        self.avg_label_width = 0
        
        self.label_margin = label_margin

        #corrected_states depend only on the 
        self.X_current, self.X_target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width = get_state_MO(self.basename, self.expt_dir, self.X_target, self.iteration_count, self.obj_idx, label_margin=self.label_margin)
        
        #state and target indexed and set as 2D array
        self.state  = np.asarray([self.X_current[self.obj_idx]])
        
        self.target = np.asarray([self.X_target[self.obj_idx]])

        _, self.d0_vec = euclidean_distance(self.X_current[:, 0:2], self.X_target)

        self.d_initial = self.d0_vec[self.obj_idx]
    
        self.target_reached = False
        if self.d0_vec[0] < self.reward_tolerance:
            self.target_reached =  True
            
    

    def step(self, action):
        
        
       # Store prev state
        prev_state, prev_target = self.state, self.target
        
        self.iteration_count += 1

        #Rescale parameters
        old_action_range = np.array([[-1, 1],[-1, 1],[-1, 1]])
        action_params = rescale_action_params(action, old_action_range, self.action_range)  #make sure to implement clipping here.



        if self.manipulation_offset:
            initial_coords, info =  rescale_state_coords(self.state, self.state_params, self.expt_dir, self.expt_name, label_width = self.avg_label_width, manipulation_offset = "start_offset", current_state = None)
            final_coords, info =  rescale_state_coords(self.target, self.state_params, self.expt_dir, self.expt_name, label_width = self.avg_label_width, manipulation_offset = "end_offset", current_state = self.state)
        else:
            initial_coords, info =  rescale_state_coords(self.state, self.state_params,self.expt_dir, self.expt_name, manipulation_offset = None)
            final_coords, info =  rescale_state_coords(self.target, self.state_params, self.expt_dir, self.expt_name, manipulation_offset = None)
        

        




        # Display previous and current results
        clear_output(wait = True)
                
        
        try:
            show_iter_results(self.expt_dir, self.iteration_count, action_params, self.prev_action_params, self.start_session, self.basename, self.reward, self.disp)
        
        finally:
            pass
        print("Molecule index:", self.obj_idx)
        print("Action:: ", action)
        print("Rescaled_action: ", action_params)
        
        if info is not None:
            print("Info:", info)
        
        if self.iteration_count > 1:
            
            print("d_space_prev", self.d_space)
            
            # Check for prev_targets_reached. If not, correct them
            correct_prev_targets = False
            
            if correct_prev_targets and self.obj_idx > 0:
                
                for mol_idx in range(self.obj_idx):
                    
                    if not self.complete_vec[mol_idx]:
                        self.obj_idx = mol_idx
                        break
                    
            go_to_manipulation_index = False
             
            if go_to_manipulation_index and self.iteration_count == 3:
                time.sleep(4)
                for mol_idx in range(self.n_objects):
                    
                    if not self.complete_vec[mol_idx]:
                        self.obj_idx = mol_idx
                        break
           
            
            
            
                
            
        
        
        
        
         # Redefine variables for new episode/trajectory
        if self.start_session == True:
            
            self.corrected_states = self.state[:, 0:2]
            self.start_session = False
            self.d_initial =  self.d0_vec[self.obj_idx]
            
            if self.iteration_count > 1:
                self.target_reached = self.complete_vec[self. obj_idx]
                
            #Introduce anchor drift correction
            if self.anchor_drift == True and self.complete_vec[self.anchor_idx] == True:
                self.drift_comp = True
                
                
                
                
            
        
            
            
        
        
        # log state action parameters      
        log_parameters(self.expt_name, self.basename, self.expt_dir, self.state, self.corrected_states, action_params, self.norm_drift, X_current = self.X_current)
        
        print(f"Initial_coords:{initial_coords}\tFinal_coords: {final_coords},\t drift: {self.norm_drift},\t Angle: {(1-self.state[0][-1])*360}\ndefault_params: {self.default_action_params}")
    


        expt_done = False
        # Do a manipulation_experiment
        expt_done = manipulation_and_scan_tcp(initial_coords, final_coords, action_params, self.default_action_params, self.expt_dir, self.state_params, self.norm_drift, self.expt_name, self.drift_comp, self.target_reached)

        #expt_done = do_manipulation_LV(initial_coords, final_coords, action_params, self.default_action_params, self.expt_dir)
        
        while not expt_done:
            time.sleep(1)
        
        #print("expt-done", expt_done)


        #Get new state
        if expt_done == True:

            self.X_current, X_target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width = get_state_MO(self.basename, self.expt_dir, self.X_target, self.iteration_count, self.obj_idx, label_margin=self.label_margin)
            
            self.state  = np.asarray([self.X_current[self.obj_idx]])
        
            self.target = np.asarray([X_target[self.obj_idx]])
            
                  
        #get reward vectors
        self.reward_vec, self.disp, self.complete_vec, self.d_space = get_state_reward_vector(self.corrected_states, prev_state, prev_target, self.d0_vec, self.X_current, X_target, self.obj_idx, tolerance = self.reward_tolerance)
        
        reward_wdrift_vec, disp_wdrift, _, _ = get_state_reward_vector(self.state[:,0:2], prev_state, prev_target, self.d0_vec, self.X_current, X_target, self.obj_idx)
       
        self.reward = np.sum(self.reward_vec)
        
        reward_wdrift = np.sum(reward_wdrift_vec)

        self.target_reached = self.complete_vec[self.obj_idx]
        
       

        #log rewards       
        log_reward(self.expt_name, self.basename, self.expt_dir, self.d_initial, self.reward, reward_wdrift, self.disp, disp_wdrift, reward_vec = self.reward_vec)


       
        # Update end of iteration variables       
        self.iterations -= 1
       
        # Set prev_action_params for display
        self.prev_action_params =  action_params
        
        
        done = False
        if self.iterations <= 0 or self.target_reached:
            done = True
        
        truncated = done
        
        

        if done == True:
            
            if self.continue_to_complete == False:
                self.start_session = True
                
            else:
                if self.target_reached == True:
                    self.start_session = True
                    
                    
        


        return self.state, self.reward, done, truncated, self.info
    
    def render(self):
        pass

    def reset(self):

        # Reset experiments
        if self.start_session == True:
            
            #Update to the next obj_idx
            if self.iteration_count > 0:
                self.obj_idx = int((self.obj_idx+1) % self.n_objects)

            X_current, X_target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width = get_state_MO(self.basename, self.expt_dir, self.X_target, self.iteration_count, self.obj_idx, label_margin=self.label_margin)
            
            self.state  = np.asarray([X_current[self.obj_idx]])
        
            self.target = np.asarray([X_target[self.obj_idx]])

        

        self.iterations = self.n_transitions

        #A similar condition for self.state == None

        return self.state, self.info





class Manipulation(Env):

    def __init__(self, action_range, default_action_params, expt_name, sxm_basename, expt_dir, n_transitions = 10, continue_to_complete = True, drift_comp = False, dmin = 0):

        
        #Observation space is no_of_points/2
        self.observation_space = Box(low= 0 ,high = 1,shape=(5,))
        self.target_space = Box(low= 0 ,high = 1,shape=(2,))

        #action parameters = Bias, Sepoint, speed  
        self.action_space =  Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]))

        self.info = {}
        
        #State variables
        self.basename = sxm_basename
        
        self.expt_dir = expt_dir
        self.dmin = dmin
        
           
        # sample object-target pair with manipulation distance > dmin
        d = 0
        within_margin = False
        while d <= self.dmin or (not within_margin):

            self.X_target = self.target_space.sample()

            #self.target is the sorted targets based on the object detection and linear assignment. X_target is the intial set of targets
            self.state, self.target, self.state_params, self.corrected_states, self.norm_drift = get_state(self.basename, self.expt_dir, self.X_target)
            d, _ = euclidean_distance(self.state[:, 0:2], self.target)
            within_margin = point_within_margin(self.target)

        #Other variables
        self.action_range =  action_range
    
        self.n_transitions = n_transitions
             
        self.default_action_params = default_action_params
    
        self.iterations = n_transitions

        self.expt_name = expt_name
        
        self.start_session = True
        
        self.iteration_count = 0
        
        self.continue_to_complete = continue_to_complete
        
        self.prev_action_params = []
        
        
        self.target_reached = False
        
        self.drift_comp = drift_comp
        
        self.reward = 0
        
        self.disp = 0


    def step(self, action):
        
        
        prev_state, prev_target = self.state, self.target
        
        old_action_range = np.array([[-1, 1],[-1, 1],[-1, 1]])
        action_params = rescale_action_params(action, old_action_range, self.action_range)  #make sure to implement clipping here.

        initial_coords, _ =  rescale_state_coords(self.state, self.state_params, self.expt_dir, self.expt_name)

        final_coords, _ =  rescale_state_coords(self.target, self.state_params, self.expt_dir, self.expt_name)
        
        clear_output(wait = True)
        
        self.iteration_count += 1
        
        # Display previous and current results
        try:
            show_iter_results(self.expt_dir, self.iteration_count, action_params, self.prev_action_params, self.start_session, self.basename, self.reward, self.disp)
        
        finally:
            pass
        
        print("Action:: ", action)
        print("Rescaled_action: ", action_params)
        
        if self.start_session == True:
            
            self.d_initial, _ = euclidean_distance(prev_state[:, 0:2], prev_target)
            self.corrected_states = self.state[:, 0:2]
            self.start_session = False
            self.target_reached = False
            
      
        log_parameters(self.expt_name, self.basename, self.expt_dir, self.state, self.corrected_states, action_params, self.norm_drift)

        
        print(f"Initial_coords:{initial_coords}\tFinal_coords: {final_coords},\t drift: {self.norm_drift},\ndefault_params: {self.default_action_params}")
    
        #print(initial_coords, final_coords, self.default_action_params, action_params, self.expt_dir)
        
        # Do a manipulation_experiment
        expt_done = manipulation_and_scan_tcp(initial_coords, final_coords, action_params, self.default_action_params, self.expt_dir, self.state_params, self.norm_drift, self.expt_name, self.drift_comp, self.target_reached)

        if expt_done == True:

            self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.X_target)
            
            
                  

        self.reward, self.disp, complete = get_state_reward(self.corrected_states, self.target, prev_state, prev_target, self.d_initial)
        reward_wdrift, disp_wdrift, _ = get_state_reward(self.state[:,0:2], self.target, prev_state, prev_target, self.d_initial)
        
    
        
        self.target_reached = complete
        
        log_reward(self.expt_name, self.basename, self.expt_dir, self.d_initial, self.reward, reward_wdrift, self.disp, disp_wdrift)

        self.iterations -= 1
        
        # Set prev_action_params for display
        self.prev_action_params =  action_params
        

        done = False
        if self.iterations <= 0 or complete == True:
            done = True
        
        truncated = done


        if done == True:
            
            if self.continue_to_complete == False:
                
                    self.start_session = True
                    
                
            else:
                if complete == True:
                    self.start_session = True
                    
                    
        
        


        return self.state, self.reward, done, truncated, self.info
    
    def render(self):
        pass

    def reset(self):

        # Reset experiments
        if self.start_session == True:

            # sample object-target pair with manipulation distance > dmin
            d = 0
            within_margin = False
            while d <= self.dmin or (not within_margin):

                self.X_target = self.target_space.sample()
                self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.X_target)
                d, _ = euclidean_distance(self.state[:, 0:2], self.target)
                within_margin = point_within_margin(self.target)
        

        self.iterations = self.n_transitions

        #A similar condition for self.state == None

        return self.state, self.info
    



class Manipulation_horizontal(Env):

    def __init__(self, action_range, default_action_params, expt_name, sxm_basename, expt_dir, n_transitions = 10, continue_to_complete = True, drift_comp = False, dmin = 0):

        
        #Observation space is no_of_points/2
        self.observation_space = Box(low= 0 ,high = 1,shape=(5,))
        self.target_space = Box(low= 0 ,high = 1,shape=(2,))

        #action parameters = Bias, Sepoint, speed  
        self.action_space =  Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]))

        self.info = {}
        
        #State variables
        self.basename = sxm_basename
        
        self.expt_dir = expt_dir
        self.dmin = dmin
        
        self.target_distance = 10*0.23/10   # 10a distance/frame_size
           
        # sample object-target pair with manipulation distance > dmin
        d = 0
        within_margin = False
        while (not within_margin):

            epsilon = np.random.rand()
            
            angle = np.radians(0) if epsilon < 0.5 else np.radians(180)
              
            x_offset, y_offset = compute_coordinates(self.target_distance, angle)
            
            #choose a random dummy target, X_target is a 1D array
            self.X_target = self.target_space.sample()
            
            #self.target is the sorted targets based on the object detection and linear assignment. X_target is the intial set of targets
            self.state, self.target, self.state_params, self.corrected_states, self.norm_drift = get_state(self.basename, self.expt_dir, self.X_target)
            x0, y0 = self.state[0][0:2]
            
            
            #Confirm that molecule not stuck in margin
            if point_within_margin([x0, y0], 0.1):
                            # indicate X_target as a 1D array
                self.X_target = np.asarray([x0+x_offset, y0+y_offset])
                
                # Recompute
                self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.X_target)
            
                        
            d, _ = euclidean_distance(self.state[:, 0:2], self.target)
            within_margin = point_within_margin(self.target, 0.1)


        #Other variables
        self.action_range =  action_range
    
        self.n_transitions = n_transitions
             
        self.default_action_params = default_action_params
    
        self.iterations = n_transitions

        self.expt_name = expt_name
        
        self.start_session = True
        
        self.iteration_count = 0
        
        self.continue_to_complete = continue_to_complete
        
        self.prev_action_params = []
        
        
        self.target_reached = False
        
        self.drift_comp = drift_comp
        
        self.reward = 0
        
        self.disp = 0


    def step(self, action):
        
        
        prev_state, prev_target = self.state, self.target
        
        old_action_range = np.array([[-1, 1],[-1, 1],[-1, 1]])
        action_params = rescale_action_params(action, old_action_range, self.action_range)  #make sure to implement clipping here.

        initial_coords, _ =  rescale_state_coords(self.state, self.state_params, self.expt_dir, self.expt_name)

        final_coords, _ =  rescale_state_coords(self.target, self.state_params,  self.expt_dir, self.expt_name)
        
        clear_output(wait = True)
        
        self.iteration_count += 1
        
        # Display previous and current results
        try:
            show_iter_results(self.expt_dir, self.iteration_count, action_params, self.prev_action_params, self.start_session, self.basename, self.reward, self.disp)
        
        finally:
            pass
        
        print("Action:: ", action)
        print("Rescaled_action: ", action_params)
        
        if self.start_session == True:
            
            self.d_initial, _ = euclidean_distance(prev_state[:, 0:2], prev_target)
            self.corrected_states = self.state[:, 0:2]
            self.start_session = False
            self.target_reached = False
            
      
        log_parameters(self.expt_name, self.basename, self.expt_dir, self.state, self.corrected_states, action_params, self.norm_drift)

        
        print(f"Initial_coords:{initial_coords}\tFinal_coords: {final_coords},\t drift: {self.norm_drift},\ndefault_params: {self.default_action_params}")
    
        #print(initial_coords, final_coords, self.default_action_params, action_params, self.expt_dir)
        
        # Do a manipulation_experiment
        expt_done = manipulation_and_scan_tcp(initial_coords, final_coords, action_params, self.default_action_params, self.expt_dir, self.state_params, self.norm_drift, self.expt_name, self.drift_comp, self.target_reached)

        if expt_done == True:

            self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.X_target)
            
            
                  

        self.reward, self.disp, complete = get_state_reward(self.corrected_states, self.target, prev_state, prev_target, self.d_initial, tolerance = 0.0)
        reward_wdrift, disp_wdrift, _ = get_state_reward(self.state[:,0:2], self.target, prev_state, prev_target, self.d_initial)
        
    
        
        self.target_reached = complete
        
        log_reward(self.expt_name, self.basename, self.expt_dir, self.d_initial, self.reward, reward_wdrift, self.disp, disp_wdrift)

        self.iterations -= 1
        
        # Set prev_action_params for display
        self.prev_action_params =  action_params
        

        done = False
        if self.iterations <= 0 or complete == True:
            done = True
        
        truncated = done


        if done == True:
            
            if self.continue_to_complete == False:
                
                    self.start_session = True
                    
                
            else:
                if complete == True:
                    self.start_session = True
                    
                    
        
        


        return self.state, self.reward, done, truncated, self.info
    
    def render(self):
        pass

    def reset(self):

        # Reset experiments
        if self.start_session == True:

            # sample object-target pair with manipulation distance > dmin
            d = 0
            within_margin = False
            within_margin = False
            
            while (not within_margin):

                epsilon = np.random.rand()
                
                angle = np.radians(0) if epsilon < 0.5 else np.radians(180)
                  
                x_offset, y_offset = compute_coordinates(self.target_distance, angle)
                
                #choose a random dummy target
                self.X_target = self.target_space.sample()
                
                #self.target is the sorted targets based on the object detection and linear assignment. X_target is the intial set of targets
                self.state, self.target, self.state_params, self.corrected_states, self.norm_drift = get_state(self.basename, self.expt_dir, self.X_target)
                x0, y0 = self.state[0][0:2]
                
                if point_within_margin([x0, y0], 0.1):
                    # indicate X_target as a 1D array
                    self.X_target = np.asarray([x0+x_offset, y0+y_offset])
                    
                    # Recompute
                    self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.X_target)
                
                
                
                d, _ = euclidean_distance(self.state[:, 0:2], self.target)
                within_margin = point_within_margin(self.target, 0.1)
        

        self.iterations = self.n_transitions

        #A similar condition for self.state == None

        return self.state, self.info
    
    


class Manipulation_diagonal(Env):

    def __init__(self, action_range, default_action_params, expt_name, sxm_basename, expt_dir, n_transitions = 10, continue_to_complete = True, drift_comp = False, dmin = 0):

        
        #Observation space is no_of_points/2
        self.observation_space = Box(low= 0 ,high = 1,shape=(5,))
        self.target_space = Box(low= 0 ,high = 1,shape=(2,))

        #action parameters = Bias, Sepoint, speed  
        self.action_space =  Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]))

        self.info = {}
        
        #State variables
        self.basename = sxm_basename
        
        self.expt_dir = expt_dir
        self.dmin = dmin
        
        self.target_distance = 20*0.23/10   # 10a distance/frame_size
           
        # sample object-target pair with manipulation distance > dmin
        d = 0
        within_margin = False
        while (not within_margin):

            angles = np.radians([60, 120, 240, 300])
            
            angle = random.choice(angles)
              
            x_offset, y_offset = compute_coordinates(self.target_distance, angle)
            
            #choose a random dummy target, X_target is a 1D array
            self.X_target = self.target_space.sample()
            
            #self.target is the sorted targets based on the object detection and linear assignment. X_target is the intial set of targets
            self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.X_target)
            x0, y0 = self.state[0][0:2]
            
            
            #Confirm that molecule not stuck in margin
            if point_within_margin([x0, y0], 0.1):
                            # indicate X_target as a 1D array
                self.X_target = np.asarray([x0+x_offset, y0+y_offset])
                
                # Recompute
                self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.X_target)
            
                        
            d, _ = euclidean_distance(self.state[:, 0:2], self.target)
            within_margin = point_within_margin(self.target, 0.1)


        #Other variables
        self.action_range =  action_range
    
        self.n_transitions = n_transitions
             
        self.default_action_params = default_action_params
    
        self.iterations = n_transitions

        self.expt_name = expt_name
        
        self.start_session = True
        
        self.iteration_count = 0
        
        self.continue_to_complete = continue_to_complete
        
        self.prev_action_params = []
        
        
        self.target_reached = False
        
        self.drift_comp = drift_comp
        
        self.reward = 0
        
        self.disp = 0


    def step(self, action):
        
        
        prev_state, prev_target = self.state, self.target
        
        old_action_range = np.array([[-1, 1],[-1, 1],[-1, 1]])
        action_params = rescale_action_params(action, old_action_range, self.action_range)  #make sure to implement clipping here.

        initial_coords, _ =  rescale_state_coords(self.state, self.state_params, self.expt_dir, self.expt_name)

        final_coords, _ =  rescale_state_coords(self.target, self.state_params,  self.expt_dir, self.expt_name)
        
        clear_output(wait = True)
        
        self.iteration_count += 1
        
        # Display previous and current results
        try:
            show_iter_results(self.expt_dir, self.iteration_count, action_params, self.prev_action_params, self.start_session, self.basename, self.reward, self.disp)
        
        finally:
            pass
        
        print("Action:: ", action)
        print("Rescaled_action: ", action_params)
        
        if self.start_session == True:
            
            self.d_initial, _ = euclidean_distance(prev_state[:, 0:2], prev_target)
            self.corrected_states = self.state[:, 0:2]
            self.start_session = False
            self.target_reached = False
            
      
        log_parameters(self.expt_name, self.basename, self.expt_dir, self.state, self.corrected_states, action_params, self.norm_drift)

        
        print(f"Initial_coords:{initial_coords}\tFinal_coords: {final_coords},\t drift: {self.norm_drift},\ndefault_params: {self.default_action_params}")
    
        #print(initial_coords, final_coords, self.default_action_params, action_params, self.expt_dir)
        
        # Do a manipulation_experiment
        expt_done = manipulation_and_scan_tcp(initial_coords, final_coords, action_params, self.default_action_params, self.expt_dir, self.state_params, self.norm_drift, self.expt_name, self.drift_comp, self.target_reached)

        if expt_done == True:

            self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.X_target)
            
            
                  

        self.reward, self.disp, complete = get_state_reward(self.corrected_states, self.target, prev_state, prev_target, self.d_initial, tolerance = 0.0)
        reward_wdrift, disp_wdrift, _ = get_state_reward(self.state[:,0:2], self.target, prev_state, prev_target, self.d_initial)
        
    
        
        self.target_reached = complete
        
        log_reward(self.expt_name, self.basename, self.expt_dir, self.d_initial, self.reward, reward_wdrift, self.disp, disp_wdrift)

        self.iterations -= 1
        
        # Set prev_action_params for display
        self.prev_action_params =  action_params
        

        done = False
        if self.iterations <= 0 or complete == True:
            done = True
        
        truncated = done


        if done == True:
            
            if self.continue_to_complete == False:
                
                    self.start_session = True
                    
                
            else:
                if complete == True:
                    self.start_session = True
                    
                    
        
        


        return self.state, self.reward, done, truncated, self.info
    
    def render(self):
        pass

    def reset(self):

        # Reset experiments
        if self.start_session == True:

            # sample object-target pair with manipulation distance > dmin
            d = 0
            within_margin = False
            
            while (not within_margin):

                angles = np.radians([60, 120, 240, 300])
                
                angle = random.choice(angles)
                  
                x_offset, y_offset = compute_coordinates(self.target_distance, angle)
                
                #choose a random dummy target
                self.X_target = self.target_space.sample()
                
                #self.target is the sorted targets based on the object detection and linear assignment. X_target is the intial set of targets
                self.state, self.target, self.state_params, self.corrected_states, self.norm_drift = get_state(self.basename, self.expt_dir, self.X_target)
                x0, y0 = self.state[0][0:2]
                
                if point_within_margin([x0, y0], 0.1):
                    # indicate X_target as a 1D array
                    self.X_target = np.asarray([x0+x_offset, y0+y_offset])
                    
                    # Recompute
                    self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.X_target)
                
                
                
                d, _ = euclidean_distance(self.state[:, 0:2], self.target)
                within_margin = point_within_margin(self.target, 0.1)
        

        self.iterations = self.n_transitions

        #A similar condition for self.state == None

        return self.state, self.info



class Manipulation_position_offset(Env):

    def __init__(self, action_range, default_action_params, expt_name, sxm_basename, expt_dir, n_transitions = 10, continue_to_complete = True, drift_comp = False, dmin = 0):

        
        #Observation space is no_of_points/2
        self.observation_space = Box(low= 0 ,high = 1,shape=(5,))
        self.target_space = Box(low= 0 ,high = 1,shape=(2,))
        
        #action parameters = Bias, Sepoint, speed  
        self.action_space =  Box(low=np.array([-1, -1, -1, -1, -1]), high=np.array([1, 1, 1, 1, 1]))

        self.info = {}
        
        #State variables
        self.basename = sxm_basename
        
        self.expt_dir = expt_dir
        self.dmin = dmin
        
           
        # sample object-target pair with manipulation distance > dmin
        d = 0
        within_margin = False
        while d <= self.dmin or (not within_margin):

            self.X_target = self.target_space.sample()

            #self.target is the sorted targets based on the object detection and linear assignment. X_target is the intial set of targets
            self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.X_target)
            d, _ = euclidean_distance(self.state[:, 0:2], self.target)
            within_margin = point_within_margin(self.target, 0.15)

        #Other variables
        self.action_range =  action_range
    
        self.n_transitions = n_transitions
             
        self.default_action_params = default_action_params
    
        self.iterations = n_transitions

        self.expt_name = expt_name
        
        self.start_session = True
        
        self.iteration_count = 0
        
        self.continue_to_complete = continue_to_complete
        
        self.prev_action_params = []        
        
        self.target_reached = False
        
        self.drift_comp = drift_comp
        
        self.reward = 0
        
        self.disp = 0


    def step(self, action):
        
        tip_action = action[0:3]
        state_action = action[3:5]

        prev_state, prev_target = self.state, self.target
        
        old_action_range = np.array([[-1, 1],[-1, 1],[-1, 1]])
        action_params = rescale_action_params(tip_action, old_action_range, self.action_range)  #make sure to implement clipping here.


        #Randomly sample offsets in range [0, 1] and incorporate into the coordinates.
        # The offsets correspond to [distance, angle]
        state_offset = np.clip([-1, 0], -1, 1)
        initial_coords, info_i =  rescale_initial_coords(self.state, self.state_params, self.expt_dir, self.expt_name, self.basename, offset_params = state_offset, label_width = 0.5*self.avg_label_width, manipulation_offset= None, current_state=None)

        target_offset = np.clip(state_action, -1, 1)
        final_coords, info_t =  rescale_target_coords(self.target, self.state_params, self.expt_dir, self.expt_name, self.basename, offset_params = target_offset, label_width = 2*self.avg_label_width, manipulation_offset="end_offset",  current_state =  self.state)

        
        clear_output(wait = True)
        
        self.iteration_count += 1
        
        # Display previous and current results
        try:
            show_iter_results(self.expt_dir, self.iteration_count, action_params, self.prev_action_params, self.start_session, self.basename, self.reward, self.disp)
        
        finally:
            pass
        
        print("Action:: ", action)
        print("Rescaled_action: ", action_params)
        
        if info_i is not None:
            print(info_i)     
    
        if info_t is not None:
           print(info_t) 
        
        
        if self.start_session == True:
            
            self.d_initial, _ = euclidean_distance(prev_state[:, 0:2], prev_target)
            self.corrected_states = self.state[:, 0:2]
            self.start_session = False
            self.target_reached = False
            
      
        log_parameters(self.expt_name, self.basename, self.expt_dir, self.state, self.corrected_states, action_params, self.norm_drift)

        
        print(f"Initial_coords:{initial_coords}\tFinal_coords: {final_coords},\t drift: {self.norm_drift},\ndefault_params: {self.default_action_params}")
    
        #print(initial_coords, final_coords, self.default_action_params, action_params, self.expt_dir)
        
        # Do a manipulation_experiment
        expt_done = manipulation_and_scan_tcp(initial_coords, final_coords, action_params, self.default_action_params, self.expt_dir, self.state_params, self.norm_drift, self.expt_name, self.drift_comp, self.target_reached)

        if expt_done == True:

            self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.X_target)
            
            
                  

        self.reward, self.disp, complete = get_state_reward(self.corrected_states, self.target, prev_state, prev_target, self.d_initial, tolerance = 0.015)
        reward_wdrift, disp_wdrift, _ = get_state_reward(self.state[:,0:2], self.target, prev_state, prev_target, self.d_initial, tolerance=0.015)
        
    
        
        self.target_reached = complete
        
        log_reward(self.expt_name, self.basename, self.expt_dir, self.d_initial, self.reward, reward_wdrift, self.disp, disp_wdrift)

        self.iterations -= 1
        
        # Set prev_action_params for display
        self.prev_action_params =  action_params
        

        done = False
        if self.iterations <= 0 or complete == True:
            done = True
        
        truncated = done


        if done == True:
            
            if self.continue_to_complete == False:
                
                    self.start_session = True
                    
                
            else:
                if complete == True:
                    self.start_session = True
                    
                    
        
        


        return self.state, self.reward, done, truncated, self.info
    
    def render(self):
        pass

    def reset(self):

        # Reset experiments
        if self.start_session == True:

            # sample object-target pair with manipulation distance > dmin
            d = 0
            within_margin = False
            while d <= self.dmin or (not within_margin):

                self.X_target = self.target_space.sample()
                self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.X_target)
                d, _ = euclidean_distance(self.state[:, 0:2], self.target)
                within_margin = point_within_margin(self.target, 0.15)
        

        self.iterations = self.n_transitions

        #A similar condition for self.state == None

        return self.state, self.info





class Manipulation_maxlen(Env):

    def __init__(self, action_range, default_action_params, expt_name, sxm_basename, expt_dir, n_transitions = 10, objects_per_system = 1, continue_to_complete = True, drift_comp = False):

        
        #Observation space is no_of_points/2
        self.observation_space = Box(low= 0 ,high = 1,shape=(5*objects_per_system,))
        self.target_space = Box(low= 0 ,high = 1,shape=(2*objects_per_system,))

        #action parameters = Bias, Sepoint, speed  
        self.action_space =  Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]))

        self.info = {}
        
        #State variables
        self.basename = sxm_basename
        
        self.n_objects =  objects_per_system
        self.X_target = random_target(size = self.n_objects)

        self.expt_dir = expt_dir
        
            #self.target is the sorted targets based on the object detection and linear assignment. X_target is the intial set of targets

        self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.X_target, max_len = True)
        

        #Other variables
        self.action_range =  action_range
    
        self.n_transitions = n_transitions
             
        self.default_action_params = default_action_params
    
        self.iterations = n_transitions

        self.expt_name = expt_name
        
        self.start_session = True
        
        self.iteration_count = 0
        
        self.continue_to_complete = continue_to_complete
        
        self.prev_action_params = []
        
        self.target_reached = False
        
        self.drift_comp = drift_comp
        
        self.reward = 0
        
        self.disp = 0


    def step(self, action):
        
        
        prev_state, prev_target = self.state, self.target
        
        action_params = rescale_action_params(action, self.action_range)  #make sure to implement clipping here.

        initial_coords =  rescale_state_coords(self.state, self.state_params)

        final_coords =  rescale_state_coords(self.target, self.state_params)
        
        clear_output(wait = True)
        
        self.iteration_count += 1
        
        # Display previous and current results
        try:
            show_iter_results(self.expt_dir, self.iteration_count, action_params, self.prev_action_params, self.start_session, self.basename, self.reward, self.disp)
        
        finally:
            pass
        
        print("Action:: ", action)
        print("Rescaled_action: ", action_params)
        
        if self.start_session == True:
            
            self.d_initial, _ = euclidean_distance(prev_state[:, 0:2], prev_target)
            self.corrected_states = self.state[:, 0:2]
            self.start_session = False
            self.target_reached = False
            
      
        log_parameters(self.expt_name, self.basename, self.expt_dir, self.state, self.corrected_states, action_params, self.norm_drift)

        
        print(f"Initial_coords:{initial_coords}\tFinal_coords: {final_coords},\t drift: {self.norm_drift},\ndefault_params: {self.default_action_params}")
    
        #print(initial_coords, final_coords, self.default_action_params, action_params, self.expt_dir)
        
        # Do a manipulation_experiment
        expt_done = manipulation_and_scan_tcp(initial_coords, final_coords, action_params, self.default_action_params, self.expt_dir, self.state_params, self.norm_drift, self.expt_name, self.drift_comp, self.target_reached)

        if expt_done == True:

            self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.target)
            
            
                  

        self.reward, self.disp, complete = get_state_reward(self.corrected_states, self.target, prev_state, prev_target, self.d_initial)
        reward_wdrift, disp_wdrift, _ = get_state_reward(self.state[:,0:2], self.target, prev_state, prev_target, self.d_initial)
        
    
        
        self.target_reached = complete
        
        log_reward(self.expt_name, self.basename, self.expt_dir, self.d_initial, self.reward, reward_wdrift, self.disp, disp_wdrift)

        self.iterations -= 1
        
        # Set prev_action_params for display
        self.prev_action_params =  action_params
        

        if self.iterations <= 0:
            
            if self.continue_to_complete == False:
                self.start_session = True
                
            else:
                if complete == True:
                    self.start_session = True
                    
                    
        
        done = False
        if self.iterations <= 0:
            done = True
        
        truncated = done


        return self.state, self.reward, done, truncated, self.info
    
    def render(self):
        pass

    def reset(self):

        # Reset experiments
        if self.start_session == True:

            self.X_target = random_target(size = self.n_objects)
            
            self.state, self.target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width  = get_state(self.basename, self.expt_dir, self.X_target, max_len = True)
        

        self.iterations = self.n_transitions

        #A similar condition for self.state == None

        return self.state, self.info
    




def custom_action(manipulation_params, default_params, user_target, sxm_basename, expt_dir, expt_name):
    
    
    state, target, scan_params, corrected_states, norm_drift = get_state(sxm_basename, expt_dir, user_target)
    
    
    initial_coords = rescale_state_coords(state, scan_params)
    
    
    final_coords = rescale_state_coords(target, scan_params)
    
   
    log_parameters(expt_name, sxm_basename, expt_dir, state, corrected_states, manipulation_params)
    print("before manipulation:")
    print(initial_coords, final_coords, default_params, manipulation_params, expt_dir, scan_params)
    
    #Do experiment    
    expt_done = manipulation_and_scan_tcp(initial_coords, final_coords, manipulation_params, default_params, expt_dir)

    
    if expt_done == True:
        
        prev_state, prev_target = state, target
        
        state, target, state_params, corrected_states, norm_drift = get_state(sxm_basename, expt_dir, user_target)
        
        
        d_initial, _ = euclidean_distance(prev_state[:, 0:2], prev_target)
        corrected_states = state[:, 0:2]

              

        reward, disp, complete = get_state_reward(corrected_states, target, prev_state, prev_target, d_initial)

    log_reward(expt_name, sxm_basename, expt_dir, d_initial, reward)

