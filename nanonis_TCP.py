
import os

import numpy as np
import time
import h5py
#import sidpy
#import pyNSID
import matplotlib.pyplot as plt
from IPython.display import clear_output

from nanonisTCP.nanonisTCP.Bias import Bias as Bias
from nanonisTCP.nanonisTCP.ZController import ZController as ZCtrl
from nanonisTCP.nanonisTCP.Scan import Scan as Scan
from nanonisTCP.nanonisTCP.FolMe import FolMe as FolMe
from nanonisTCP.nanonisTCP import nanonisTCP as nanonisTCP



class Nanonis_TCP():

    def __init__(self):

        self.host_ip = '127.0.0.1'
        self.port = 6501

    def NTCP_open(self, call = False):

        if not call:            
            self.NTCP = nanonisTCP(self.host_ip, self.port)
        else:
            pass

        return
    
    def NTCP_close(self, call = False):
        if not call:
            self.NTCP.close_connection()
        else:
            pass

        return

    def set_bias(self, set_bias, call = False):
        
        self.NTCP_open(call)

        try:
            bias = Bias(self.NTCP)
            bias.Set(set_bias)

        finally:
            self.NTCP_close(call)

        return
        

    def get_bias(self, call = False):

        self.NTCP_open(call)

        try:
            bias = Bias(self.NTCP)
            result = bias.Get()
        
        finally:
            self.NTCP_close(call)

        return result


    def set_setpoint(self, setpoint, limit = 100E-9, call = False):

        if setpoint > limit:

            return "Sepoint is higher than limit"

        self.NTCP_open(call)

        try:
            zctrl = ZCtrl(self.NTCP)
            zctrl.SetpntSet(setpoint)

        finally:
            self.NTCP_close(call)

        return
    

    def get_setpoint(self, call = False):

        self.NTCP_open(call = False)

        try:
            zctrl = ZCtrl(self.NTCP)
            result = zctrl.SetpntGet()

        finally:
            self.NTCP_close(call)

        return result

    def scan(self, scan_direction = "up", call = False):
        

        self.NTCP_open(call)

        try:
            scan = Scan(self.NTCP)
            scan.Action(scan_action = "start", scan_direction = scan_direction)
            
            #Waits till the end of scan
            scan_status, _,_,  = scan.WaitEndOfScan()
            
        finally:
            self.NTCP_close(call)

        # return true if scan complete (scan_status = False)
        return not scan_status

    def scan_Frameset(self, x_center, y_center, frame_x, frame_y, angle, call = False):
        

        self.NTCP_open(call)


        try:
            scan = Scan(self.NTCP)
            scan.FrameSet(x_center, y_center, frame_x, frame_y, angle)
            
            
        finally:
            time.sleep(1)
            self.NTCP_close(call)

        
        return
    
    def scan_Frameget(self, call = False):
        
        self.NTCP_open(call)

        try:
            scan = Scan(self.NTCP)
            x, y, w, h, angle = scan.FrameGet()

        finally:
            self.NTCP_close(call)

        return x,y, w, h, angle

    def folme(self, position, speed = 1E-9, call = False):

        self.NTCP_open(call)

        try:

            folme = FolMe(self.NTCP)
            folme.SpeedSet(speed = speed, custom_speed=True)
            time.sleep(0.5)

            folme.XYPosSet(position[0], position[1], Wait_end_of_move=True)
    

        finally:
            self.NTCP_close(call)

        return True


    def slew_set_setpoint(self, setpoint, rate = 1E-9, limit = 100E-9, call = False):
        
        if setpoint > limit:

            return "Setpoint is higher than limit"



        self.NTCP_open(call)

        ztrl = ZCtrl(self.NTCP)

        try:

            diff = setpoint  - ztrl.SetpntGet()

            if diff > 0:
        
                while True:

                    current_setP = ztrl.SetpntGet()+rate/2

                    if current_setP < setpoint:
                        ztrl.SetpntSet(current_setP)

                    else:
                        ztrl.SetpntSet(setpoint)
                        break
                    time.sleep(0.5)


            else:

                while True:

                    current_setP = ztrl.SetpntGet() - rate/2

                    if current_setP > setpoint:
                        ztrl.SetpntSet(current_setP)

                    else:
                        ztrl.SetpntSet(setpoint)
                        break
                    time.sleep(0.5)
        
        finally:
            self.NTCP_close(call)

    
        return
    

    def slew_set_bias(self, set_bias, rate = 0.1, limit = 0.001, call = False):
        
        if np.abs(set_bias) < limit:

            return "Bias is lower than limit"



        self.NTCP_open(call)

        bias = Bias(self.NTCP)

        if bias.Get()*set_bias < 0:

            self.NTCP_close()
            return "Bias ramps thru 0 V. Check sign !!!"
        

        try:

            diff = set_bias  - bias.Get()

            if diff > 0:
        
                while True:
                    
                    #increment every half second
                    current_bias = bias.Get()+rate/2 

                    if current_bias < set_bias:
                        bias.Set(current_bias)

                    else:
                        bias.Set(set_bias)
                        break
                    time.sleep(0.5)


            else:
        
                while True:

                    current_bias = bias.Get() - rate/2

                    if current_bias > set_bias:
                        bias.Set(current_bias)

                    else:
                        bias.Set(set_bias)
                        break
                    time.sleep(0.5)

        finally:
            self.NTCP_close(call)

    
        return
        

    def manipulation(self, initial_coords, final_coords, manipulation_params, default_params, 
                     bias_slew_rate = 0.03, setpoint_slew_rate = 10E-9,
                     bias_lower_limit = 0.001, setpoint_limit = 100E-9, call = False):
        
        done = False

        if np.abs(manipulation_params[0]) < bias_lower_limit or np.abs(default_params[0]) < bias_lower_limit:
            return "tip bias parameter is lower than the limit !!!"

        if np.abs(manipulation_params[1]) > setpoint_limit or np.abs(default_params[1]) > setpoint_limit:
            
            return "setpoint parameter is higher than the limit !!!"
        
        try:

        
            self.NTCP_open(call)
            folme = FolMe(self.NTCP)

            print("\tMoving to initial position")
            # Go to intial_position:
            folme.XYPosSet(initial_coords[0], initial_coords[1], Wait_end_of_move = True)

            #Updata manipulation parameters
            set_bias, set_setP, set_speed = manipulation_params

            print("\tUpdating manipulation parameters")
            try:
                self.slew_set_bias(set_bias, rate = bias_slew_rate, call = True)
            except:
                pass
        
            try:
                self.slew_set_setpoint(set_setP, rate = setpoint_slew_rate, call = True)
            except:
                pass
            
            
            try:
                
                folme.SpeedSet(set_speed, custom_speed=True)
                time.sleep(1)
                
                print("\tMoving to target position")
                # Move to final position:
                folme.XYPosSet(final_coords[0], final_coords[1], Wait_end_of_move=True)
                time.sleep(0.2)
            except:
                pass


            print("\tBack to default parameters")
            #Updata default parameters
            set_bias, set_setP, set_speed = default_params

            try:
                self.slew_set_bias(set_bias, rate = bias_slew_rate, call=True)
            except:
                pass
            try:
                self.slew_set_setpoint(set_setP, rate = setpoint_slew_rate, call=True)
            except:
                pass

            
            folme.SpeedSet(set_speed, custom_speed=True)

        finally:
            self.NTCP_close(call)
            print("\tManipulation complete")

        done = True  

        return done
