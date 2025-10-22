
#@author: Ganesh Narasimha

import gym

from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import time
from IPython.display import clear_output
import os
import numpy as np


from Manipulation_coords import get_manipulation_coords
from expt_utils import transform_coord_array, translation_angle, write_log, get_latest_file, get_next_file_name, copy_files, get_sxm_filenames, arr_to_linestring, euclidean_distance
from detect_molecules import get_latest_image
from stm_manipulation_experiments import manipulation_and_scan_tcp, rescan_tcp, do_manipulation_LV, rescan_LV, do_manipulation_LV_dummy, rescan_LV_dummy, path_manipulation_scan_tcp
from display_results import show_iter_image_results, show_iter_results
from drift_estimation import drift_correction_far_labels
from get_reward import get_displacement_reward, get_state_reward, get_state_reward_vector
from get_target import Random_Target
from path_fns import modify_state_coords, transform_to_real_coords, rescale_state_coords, rescale_initial_coords, rescale_target_coords, get_path, target_avoid_collision, object_stuck
from env_functions import *



class Manipulation_multi_object(Env):

    def __init__(self, action_range, default_action_params, expt_name, sxm_basename, expt_dir, n_transitions = 10, continue_to_complete = True, drift_comp = False, 
                 reward_tolerance = 0.03, anchor = (False, 0, False), target_control = None, st_offset = None, end_offset = None, margin = 0.0, label_margin = 0.0,
                 detect_dict = {}):

        
        #Observation space is no_of_points/2
        self.observation_space = Box(low= 0 ,high = 1,shape=(5,))

        #action parameters = Bias, Sepoint, speed  
                #action parameters = Bias, Sepoint, speed  

        self.action_space = {
            "param": Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1])),
            "offset": Box(low=np.array([-1, -1]), high=np.array([1, 1]))
            }


        self.info = {}
        
        #State variables

        self.basename = sxm_basename #  basename of the sxm files
         
        self.target_control = target_control  # dictionary specifying target selection method and parameters

        

        

        self.obj_idx = 0                        # index of the object being manipulated currently

        self.expt_dir = expt_dir               # Directory where the experiments files (.sxm) are stored.
        self.st_offset_condition = st_offset   # manipulation offset condition at start of manipulation
        self.end_offset_condition = end_offset  # manipulation offset condition at end of manipulation
        self.margin = margin                    # margin to be maintained from the edges of the scan area while selecting targets
        self.label_margin = label_margin


           
        self.n_transitions = n_transitions                   # number of manipulation steps in an episode
             
        self.default_action_params = default_action_params   # Default action parameters for the tip when not manipulating.
    
        self.iterations = n_transitions                      # iterations left in the current episode

        self.expt_name = expt_name                           # Name of the experiment (used for logging)
        
        self.start_session = True                            # Flag to indicate start of a new session i.e, a change of the manipulated object.
        
        self.iteration_count = 0                            # counts the number of iterations done in the current episode.
        
        self.continue_to_complete = continue_to_complete    # Flag to indicate whether to continue until the current object is complete.
        
        self.prev_action_params = []                        # stores the previous action parameters for display purposes.
        
        
        self.use_anchor, self.anchor_idx, self.anchor_placed = anchor # anchor_placed = True:anchor had been placed atleast once.
        
        self.drift_comp = False                                             # Flag to indicate whether to perform drift compensation.
        self.drift_comp_condition = self.drift_comp                # Condition for drift compensation. this can be a bool or str
        
        self.avg_label_width = 0                             # average width of the labels in the current experiment.

        self._finished = False                          # Flag to indicate all objects are complete.
        self.complete_vec = None                        # Vector array indicating completion status of all objects.
        self.stuck, self.stuck_count = False, 0     # stuck status and count of consecutive stuck iterations.
        
        
        detect_dict = detect_dict.copy()            
        self.detect_dict = detect_dict               # dictionary containing parameters for molecule detection.
        
    
        count_0 = 0
        while True or (not self._finished):

            self.X_target = self._get_target()
            self.n_objects = self.X_target.shape[0]


            detect_dict["use_prev"] = False #reset use_prev for first iteration
    
            #corrected_states incorporates drift correction.
            self.X_current, self.X_target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width, self.labels, self.file_name = get_state_MO(self.basename, self.expt_dir, self.X_target, self.iteration_count, self.obj_idx, label_margin=self.label_margin, anchor_tuple=(self.anchor_placed, self.anchor_idx), detect_dict=detect_dict)
            
            if self.detect_dict.get("target_n") is None or self.detect_dict.get("target_n") < 1:
                detect_dict["target_n"] = self.labels[0].shape[0]
                self.detect_dict["target_n"] = self.labels[0].shape[0]
            
            
            #state and target indexed and set as 2D array
            self.state  = np.asarray([self.X_current[self.obj_idx]])
            
            self.target = np.asarray([self.X_target[self.obj_idx]])


            #Compute reward and target_reached
            _, self.d0_vec = euclidean_distance(self.X_current[:, 0:2], self.X_target)

            self.d_initial = self.d0_vec[self.obj_idx]
            
            prev_obj_idx = self.obj_idx

            self.reward_tolerance =  reward_tolerance*1E-9/self.state_params[0]  # in real units, assuming state_params[0] is the frame length in meters.

            self.reward, self.target_reached, _ = self._compute_reward()   #self.obj_idx updates to the first incomplete object
            count_0 += 1
            if prev_obj_idx == self.obj_idx and count_0 > 1:  #make sure the obj_idx has not changed. this also helps enable drift_comp
                break


    

        
        self.action_range =  {
            "param": action_range,
            "offset": np.asarray([[-1E-9/self.state_params[0], 1E-9/self.state_params[0]], [-1E-9/self.state_params[0], 1E-9/self.state_params[0]]]) # offset range in +/- 1 nm
            }

        
            
    

    def step(self, action_dict):

        param_action =  action_dict.get("param", None)
        offset_action =  action_dict.get("offset", np.array([0, 0]))

        #Rescale parameters
        old_param_range = np.array([[-1, 1],[-1, 1],[-1, 1]])
        old_offset_range = np.array([[-1, 1],[-1, 1]])


        print("Inspecting action dims",param_action, old_param_range, self.action_range["param"])

        action_params = rescale_action_params(param_action, old_param_range, self.action_range["param"])  #make sure to implement clipping here.
        
        # normalize offset to set within the range of -1 nm to 1 nm.
        tar_offset_params = rescale_action_params(offset_action, old_offset_range, self.action_range["offset"])  #make sure to implement clipping here.


        
       # Store prev state
        self.prev_state, self.prev_target = self.state, self.target
        self.iteration_count += 1
        
        

        # Get the entire path in real coordinates.

        
        # Modify current and target positions based on manipulation_offset
        self.st_offset = {"type": "start_offset", "delta_offset":0} if self.st_offset_condition is None else self.st_offset_condition

        if self.end_offset_condition is None:
            self.end_offset = {"type": "custom_target_offset", "offset": np.ravel(tar_offset_params)}
        else:
            self.end_offset = self.end_offset_condition
        
        state_w_offset, _ = modify_state_coords(self.state, self.expt_dir, self.expt_name, label_width = self.avg_label_width, current_state = self.state, manipulation_offset = self.st_offset) # this is 2D
    
        target_w_offset0, self.info["target_offset"] = modify_state_coords(self.target, self.expt_dir, self.expt_name, label_width = self.avg_label_width, current_state = self.state, manipulation_offset = self.end_offset) # this is 2D

        # Ensure target is within bounds and not collide with labels.
        target_w_offset, _ = target_avoid_collision(self.target, target_w_offset0, self.labels, self.margin, padding = 0.00, x_initial = self.state[:, 0:2], eps = self.reward_tolerance)  # this is 2D
        
        



        path =  get_path(state_w_offset, target_w_offset, self.labels, self.expt_dir, self.expt_name, self.filename, orig_target=self.target, scan_params = self.state_params)

        path_real_coords = transform_to_real_coords(path, self.state_params)  




        # Display previous and current results
        clear_output(wait = True)
                
        
        try:
            display_results = show_iter_results(self.expt_dir, self.iteration_count, action_params, self.prev_action_params, self.start_session, self.basename, self.reward, self.disp)
            self.info["results"] = display_results
        finally:
            pass
        print("Molecule index:", self.obj_idx)
        print("Action:: ", action_dict)
        print("Rescaled_action: ", action_params)
        print(f"Drift compensation: {self.drift_comp}, Anchor placed:{self.anchor_placed}")
        print(f"offset_action: {offset_action}, target_params: {tar_offset_params}, target_diff0:{(target_w_offset0[0] - self.target[0])}, target_difference: ({(target_w_offset[0] - self.target[0])}")
        print(f"Stuck: {self.stuck}, stuck_count:{self.stuck_count}")


        # Iteration updates      

        if self.iteration_count > 1:
            
            print("d_space_prev", self.d_space)
                                           
        if self.start_session:
            self._start_session()   # checkpoints before starting a new session.          
            
        
        # log state action parameters      
        log_parameters(self.expt_name, self.basename, self.expt_dir, self.state, self.corrected_states, action_params, self.norm_drift, X_current = self.X_current)
        
        print(f"Initial_coords:{path_real_coords[0]}\tFinal_coords: {path_real_coords[-1]},\t drift: {self.norm_drift},\t Angle: {(1-self.state[0][-1])*360}\ndefault_params: {self.default_action_params}")
    


        expt_done = False
        # Do a manipulation_experiment
        #expt_done = manipulation_and_scan_tcp(initial_coords, final_coords, action_params, self.default_action_params, self.expt_dir, self.state_params, self.norm_drift, self.expt_name, self.drift_comp, self.target_reached)
        expt_done, tiprec_data = path_manipulation_scan_tcp(path_real_coords, action_params, self.default_action_params, self.expt_dir, self.state_params, self.norm_drift, self.expt_name, self.drift_comp, self.target_reached)
        #expt_done = do_manipulation_LV_dummy(path_real_coords[0], path_real_coords[-1], action_params, self.default_action_params, self.expt_dir)
        time.sleep(2)

        #expt_done = do_manipulation_LV(initial_coords, final_coords, action_params, self.default_action_params, self.expt_dir)
        

        
        #print("expt-done", expt_done)


        #Get new state
        if expt_done == True:

            self.X_current, X_target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width, self.labels, self.filename = get_state_MO(self.basename, self.expt_dir, self.X_target, self.iteration_count, self.obj_idx, label_margin=self.label_margin, anchor_tuple=(self.anchor_placed, self.anchor_idx), detect_dict=self.detect_dict)
            
            self.state  = np.asarray([self.X_current[self.obj_idx]])
        
            self.target = np.asarray([X_target[self.obj_idx]])
            
                  
        
        self.reward, self.target_reached, reward_dict = self._compute_reward()       
        
      

        #log rewards and tiprec_buffer    
        log_reward(self.expt_name, self.basename, self.expt_dir, self.d_initial, self.reward, reward_dict.get("reward_wdrift"), self.disp, reward_dict.get("disp_wdrift"), reward_vec = self.reward_vec)
        if tiprec_data is not None:
            log_tiprec_buffer(tiprec_data, self.expt_name, self.basename, self.expt_dir)

       
        # Update end of iteration variables       
        self.iterations -= 1
       
        # Set prev_action_params for display
        self.prev_action_params =  action_params
        
        
        done = self.target_reached
        truncated = True if self.iterations <= 0 else False

        if self.continue_to_complete:
            self.start_session = True if done else False
        else:        
            self.start_session = True if (done or truncated) else False

        # we use done and truncated interchangeably here.
        done = done or truncated
        truncated = done
        
                                       

        return self._wrap_state(self.state, self.labels, self.state_params[0]), self.reward, done, truncated, self.info
    

    def render(self):
        pass


    def reset(self):

        # Reset experiments
        if self.start_session == True:
            
            
            self.obj_idx = self._update_obj_idx() if self.iteration_count >0 else self._update_obj_idx(lowest_idx = True)            

            self.X_target = self._get_target()

            X_current, X_target, self.state_params, self.corrected_states, self.norm_drift, self.avg_label_width, self.labels, self.filename = get_state_MO(self.basename, self.expt_dir, self.X_target, self.iteration_count, self.obj_idx, label_margin=self.label_margin, anchor_tuple=(self.anchor_placed, self.anchor_idx), detect_dict=self.detect_dict)
            
            self.state  = np.asarray([X_current[self.obj_idx]])
        
            self.target = np.asarray([X_target[self.obj_idx]])

      
        self.iterations = self.n_transitions
        


        #A similar condition for self.state == None

        return self._wrap_state(self.state, self.labels, self.state_params[0]), self.info




    def _get_target(self):

        """ Get target positions based on the target selection method specified in self.target_control."""
        
        
        
        if self.target_control is None:
            print("No target control specified")
        
            return None
        
        target_type = self.target_control.get("target_type", None)


        if target_type == "user_input":
            # get the 2D array of targets from the user
            target_coords = self.target_control.get("target_coords", None)

            # Extract anchor information if provided
            anchor_dict = self.target_control.get("anchor", None)

            if anchor_dict is not None:
                self.use_anchor = anchor_dict.get("condition", False)
                self.anchor_idx = anchor_dict.get("anchor_idx", 0)
            
            # Drift compensation_information:
            self.drift_comp_condition = self.target_control.get("drift_comp", False)
            self._drift_comp()
                
            return np.asarray(target_coords)


        if target_type == "random":

            dmin = self.target_control.get("dmin", 0)

            return Random_Target(margin = self.margin).min_distance(x_ref = self.state[:, 0:2], dmin = dmin)


        if target_type == "random_with_anchor":  # This has two variants: (1) with anchor(2) target at constant distance from anchor.



            # Get anchor information
            anchor_dict = self.target_control.get("anchor", None)

            if anchor_dict is not None:
                anchor_point = anchor_dict.get("anchor_point", [0.15, 0.15])
                self.use_anchor = anchor_dict.get("condition", False)
                self.anchor_idx = anchor_dict.get("anchor_idx", 0)
            else:
                print("No anchor information provided for random_with_anchor target type.")

            # Choose target
            # condition 1 - minimum distance from anchor point
            dmin_anchor_target = self.target_control.get("dmin_anchor_target", 0.1)
            if dmin_anchor_target is not None:
                target = Random_Target(margin= self.margin).min_distance(x_ref = anchor_point, dmin = dmin_anchor_target)

            # condition 2 - constant distance from anchor point. This overrides condition 1 if both are provided.
            d_constant_anchor = self.target_control.get("d_constant_anchor", None)
            if d_constant_anchor is not None:
                target = Random_Target(margin= self.margin).constant_distance(x_ref = anchor_point, d = d_constant_anchor)

            # Insert anchor point at the specified index
            anchor_point = np.asarray(anchor_point).reshape(1, -1)
            if self.anchor_idx == 0:
                target = np.vstack([anchor_point, target])
            else:
                target = np.vstack([target, anchor_point])

            # Drift compensation_information:
            self.drift_comp_condition = self.target_control.get("drift_comp", False)
            self._drift_comp()
            
            
            return target


    def _drift_comp(self):
        """ Update drift compensation based on the specified condition in self.drift_comp_condition."""

        if isinstance(self.drift_comp_condition, bool):
            self.drift_comp = self.drift_comp_condition
            
        if self.drift_comp_condition == 'trigger_on_anchor' and self.use_anchor:
            if self.complete_vec is not None:
                if self.complete_vec[self.anchor_idx]:
                    self.drift_comp = True
                    self.anchor_placed = True
                    
                
    
    def _stuck_status(self):
        """ Compute the stuck status based on the previous states."""

        expt_log_dir = os.path.join(self.expt_dir, 'expt_log', self.expt_name)
        self.stuck, _ = object_stuck(self.state[:, 0:2], expt_log_dir, prev_i = 1, threshold = self.reward_tolerance)
        
        if self.iterations == self.n_transitions: # on every reset condition
            self.stuck_count = 0
        
        if self.stuck:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            
        #print(f"Stuck: {self.stuck}, stuck_count:{self.stuck_count}")
        return (self.stuck_count/self.n_transitions)

        
    def _compute_reward(self, gcrl_reward = False, **kwargs):
        """ 
        Compute the reward based on the current and previous states and targets.
        Use gcrl_reward = True to compute reward for goal-conditioned RL, or to get reward for specific achieved and desired goals.
        
        """

        if self.iteration_count == 0:
        
            self.reward_vec =  np.zeros(shape = (self.n_objects,))
            
            self.reward = np.sum(self.reward_vec)
            
            self.disp = 0

            self.complete_vec = self.d0_vec < self.reward_tolerance

            
            self.obj_idx = self._update_obj_idx(lowest_idx = True)
            self.target_reached = self.complete_vec[self.obj_idx]
                

            
            reward_dict = {
                "reward_vec": self.reward_vec,
                "complete_vec": self.complete_vec,
                "disp": self.disp,  # displacement of object, incomparison to previous position.            
                "d_space": self.d0_vec[self.obj_idx],  # distance between the current and target positions
                "reward_wdrift": self.reward,
                "disp_wdrift": self.disp   
            }

            return self.reward, self.target_reached, reward_dict
        

        if gcrl_reward:

            achieved_goal = kwargs.get("achieved_goal", None)
            goal = kwargs.get("goal", None)

            if achieved_goal is None or goal is None:
                raise ValueError("For gcrl_reward, 'acheived_goal' and 'goal' must be provided as keyword arguments.")
            
            reward, _, success = get_state_reward(achieved_goal, goal, self.d_initial, tolerance = self.reward_tolerance)

            return reward, success


        #get reward vectors
        self.reward_vec, self.disp, self.complete_vec, self.d_space = get_state_reward_vector(self.corrected_states, self.prev_state, self.prev_target, self.d0_vec, self.X_current, self.X_target, self.obj_idx, tolerance = self.reward_tolerance)
        
        reward_wdrift_vec, disp_wdrift, _, _ = get_state_reward_vector(self.state[:,0:2], self.prev_state, self.prev_target, self.d0_vec, self.X_current, self.X_target, self.obj_idx)
       
        self.reward = np.sum(self.reward_vec)
        
        reward_wdrift = np.sum(reward_wdrift_vec)

        self.target_reached = self.complete_vec[self.obj_idx]

        reward_dict = {
            "reward_vec": self.reward_vec,
            "complete_vec": self.complete_vec,
            "disp": self.disp,  # displacement of object, incomparison to previous position.            
            "d_space": self.d_space,  # distance between the current and target positions
            "reward_wdrift": reward_wdrift,
            "disp_wdrift": disp_wdrift
        }

        return self.reward, self.target_reached, reward_dict
    

    
    def _update_obj_idx(self, lowest_idx = False):

        """
        Update the object index to the next incomplete object.
        """
        if all(self.complete_vec) and self.target_control.get("target_type") == "user_input":
            print("All objects are complete.")
            self._finished = True
            return self.obj_idx

        if lowest_idx:
            for idx, complete in enumerate(self.complete_vec):
                if self.anchor_placed and idx == self.anchor_idx:
                    continue
                if not complete:
                    return idx
            return self.obj_idx  # Fallback, should reach here if all are complete.

        count = 0
        while count < self.n_objects:
            self.obj_idx = (self.obj_idx + 1) % self.n_objects

            if self.anchor_placed and self.obj_idx == self.anchor_idx: # skip anchor if already placed. do not manipulate it again.
                continue
            
            self.target_reached = self.complete_vec[self.obj_idx]
            if not self.target_reached:
                break
            count += 1

        return self.obj_idx

    def _start_session(self):
        """ Update variables at the start of a new session (i.e, new object manipulation)."""          
         
        if self.start_session:
            
            self.corrected_states = self.state[:, 0:2]
            self.d_initial =  self.d0_vec[self.obj_idx]
            self.start_session = False

            # update drift compensation condition.
            self._drift_comp()



    
    def _wrap_state(self, state, labels, frame_length):

        """ Wrap the state to include additional information"""
    
        near_relative_coords = get_nearest_coords(state, labels[0], n_near = 3, frame_length = frame_length, cutoff_distance = 2E-9) #shape (1, 6)
        stuck_variable = np.asarray([self._stuck_status()])
        state_with_near_coords = np.concatenate((state.reshape(-1), stuck_variable.reshape(-1), near_relative_coords.reshape(-1)), axis=-1).reshape(1, -1)  # shape (1, 12)

        state_dict = {
            "observation": state,
            "achieved_goal": state[:, 0:2],
            "desired_goal": state[:, 2:4],
            "state_with_near_coords": state_with_near_coords  # this is an array of shape (1, 6) [[dx1, dy1, dx2, dy2, dx3, dy3]]
        }
        return state_dict            
   
















    

    









