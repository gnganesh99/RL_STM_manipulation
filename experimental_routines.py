import os

import numpy as np
import time
import h5py
#import sidpy
#import pyNSID
import matplotlib.pyplot as plt
from IPython.display import clear_output
import win32com.client

class LV_STM():

    def __init__(self, client = 'LabVIEW.Application', VI_dir = r"E:\labView\Ganesh\LV_programs_2018\v5_infinity\ActiveX_client") -> None:
        
        self.labview = win32com.client.Dispatch(client)
        self.dir = VI_dir

    def set_bias(self, bias):
        """
        Sets tip bias
        
        Parameters: 
            bias: Tip bias (V)
        """

        vi_file = 'voltage_set.vi'
        self.VI = self.labview.getvireference(os.path.join(self.dir, vi_file))
        self.VI._FlagAsMethod("Call")  
        self.VI.setcontrolvalue('set_Bias(V)', bias)
        self.VI.Call()
        
        del self.VI
        
        return

    def set_setpoint(self, setpoint, limit = 1E-9):
        
        """
        Sets the setpoint-parameter of the Z-controller

        Parameters:
            setpoint: Z-controller setpoint (A)

        Kwargs:
            limit: safety upper limit of the setpoint (default = 1E-9)

        Raises:
            Returns warning if setpoint > limit
        """

        if setpoint > limit:
            return "setpoint is higher than limit !!!"
            
        else:
            vi_file = 'setpoint_set.vi'
            self.VI = self.labview.getvireference(os.path.join(self.dir, vi_file))
            self.VI._FlagAsMethod("Call")  
            self.VI.setcontrolvalue('set_setpoint(A)', np.abs(setpoint))
            self.VI.Call()
    
            del self.VI
            return

    def get_bias(self):
        
        """
        Gets the current tip-bias value

        Returns:
            Bias (V)
        """

        vi_file = 'voltage_get.vi'
        self.VI = self.labview.getvireference(os.path.join(self.dir, vi_file))
        self.VI._FlagAsMethod("Run")  
        #self.VI.setcontrolvalue('get_Bias(V)', bias)
        self.VI.Run()
        get_bias_result =  self.VI.getcontrolvalue('get_Bias(V)')

        del self.VI
        
        return get_bias_result 

    
    def get_setpoint(self):

        """
        Gets the current tip-bias value

        Returns:
            Z-controller Setpoint(A)
        """
  
        vi_file = 'setpoint_get.vi'
        self.VI = self.labview.getvireference(os.path.join(self.dir, vi_file))
        self.VI._FlagAsMethod("Run")  
        self.VI.Run()
        self.get_setpoint =  self.VI.getcontrolvalue('get_setpoint(A)')
        
        del self.VI
        
        return self.get_setpoint

    def scan(self, **kw):

        """
        Performs a raster scan
        Uses previously set parameters if no changes 

        Parameters (Kwargs):
            position: Center-coordinates [x, y] of the scan frame in 'nm' units
            scan_frame_nm: Size of the scan frame in 'nm'. 
            scan_speed_nmps: line scan speed in nm/s
            angle: scan angle in degrees
            scan_direction: 0 = downward, 1 = upward
        """

        vi_file = 'raster_scan.vi'
        self.VI = self.labview.getvireference(os.path.join(self.dir, vi_file))
        
        if kw.get('position(nm)') != None:
            self.VI.setcontrolvalue("get_Center_coordinates (nm)", kw.get('position(nm)')) 
        else:
            pass
        
        
        if kw.get('scan_frame_nm') != None:
            
            frame_size = kw.get('scan_frame_nm')
            
            if type(frame_size) == float or type(frame_size) == int:
                frame_size = [frame_size, frame_size]
                
            self.VI.setcontrolvalue("get_frame_size(nm)", frame_size) 
        else:
            pass
        
        
        if kw.get('scan_speed_nmps') != None:
            
            scan_speed = kw.get('scan_speed_nmps')
            
            if type(scan_speed) == float or type(scan_speed) == int:
                scan_speed = [scan_speed, scan_speed]
                
            self.VI.setcontrolvalue("scan_speed (nmps)", scan_speed) 
        else:
            pass
        
        
        if kw.get('angle') != None:
            self.VI.setcontrolvalue("get_Angle (deg)", float(kw.get('angle')))
        else:
            pass
        
        
        if kw.get('scan_direction') != None:
            self.VI.setcontrolvalue("get_Direction", bool(kw.get('scan_direction')))
        else:
            pass
                
              
        self.VI._FlagAsMethod("Run") 
    
        self.VI.Run()
        
        return

    
    def stop_scan(self):
        
        vi_file = 'stop_scan.vi'
        self.VI = self.labview.getvireference(os.path.join(self.dir, vi_file))
        self.VI._FlagAsMethod("Call") 
        self.VI.Call()
        
        #close the VI reference
        del self.VI
        return
    
    def get_filename(self, file_dir, file_basename, index_length = 4):

        """
        Returns the saved sxm filename and the next sxm filename

        Args:
            file_dir: directory where the files are saved
            file_basename: basename of the file (w/o the numerical suffix)
            index_length: length of the numerical string (default = 4)
        """
        
        vi_file = 'get_file_name.vi'
        self.VI = self.labview.getvireference(os.path.join(self.dir, vi_file))
        
        self.VI.setcontrolvalue("directory", file_dir)
        self.VI.setcontrolvalue("file_basename", file_basename)
        self.VI.setcontrolvalue("index_length", index_length)
        
        self.VI._FlagAsMethod("Run")
        self.VI.Run()
        
        saved_filename = self.VI.getcontrolvalue("saved_filename")
        next_file_index = self.VI.getcontrolvalue("next_file_index")
        next_filepath = self.VI.getcontrolvalue("next_filepath")
        next_filename = self.VI.getcontrolvalue("next_filename")

        del self.VI
        return saved_filename, next_filename

    
    def manipulate_and_scan(self, initial_coords, final_coords, manipulation_params, default_params, 
                        bias_slew_rate = 0.09, setpoint_slew_rate = 30000, 
                       bias_lower_limit = 0.001, setpoint_limit = 100000):

        """
        Performs a single manipulation procedure followed by a scan
        setpoint values in pA !

        Parameters:
            initial_coords: intial manipulation coordinates [x, y] in (m)
            final_coords: final manipulaton coordinates [x, y] in (m)
            manipulation_params: Manipulation parameters [Bias (V), Setpoint (pA), Speed (nm/s)]
            default_params: Normal scan parameters [Bias (V), Setpoint (pA), Speed (nm/s)]
            bias_slew_rate: (default = 0.1 V/s) Rate of change of tip-bias parameter
            setpoint_slew_rate: (default = 10000 pA/s or 10 nA/s) Rate of change of setpoint
            bias_lower_limit: (default = 0.001 V) lower limit of the bias
            setpoint_limit: (default = 100000 pA or 100 nA) higher limit of the setpoint

        Raises:
            raises warning if bias < bias_lower limit
            raises warning if setpoint > setpoint_limit
        """

        vi_file = 'manipulation_and_scan.vi'
        self.VI = self.labview.getvireference(os.path.join(self.dir, vi_file))

        if np.abs(manipulation_params[0]) < bias_lower_limit or np.abs(default_params[0]) < bias_lower_limit:
            return "tip bias parameter is lower than the limit !!!"

        elif np.abs(manipulation_params[1]) > setpoint_limit or np.abs(default_params[1]) > setpoint_limit:
            
            return "setpoint parameter is higher than the limit !!!"

        else:
            
            self.VI.setcontrolvalue("Manipulation_parameters", manipulation_params)
            self.VI.setcontrolvalue("Default_parameters", default_params)
            self.VI.setcontrolvalue("initial_point", initial_coords)
            self.VI.setcontrolvalue("final_point", final_coords)
            self.VI.setcontrolvalue("bias_slew_rate (V/s)", bias_slew_rate)
            self.VI.setcontrolvalue("setpoint_slew_rate (pA/s)", setpoint_slew_rate)

            self.VI._FlagAsMethod("Run")
            print("\n##### Manipulation in progress.... #####\n")
            
            self.VI.Run()

            done = self.VI.getcontrolvalue("Done")

            if done == True:
                return "Manipulation procedure complete"
            else:
                time.sleep(5)

        del self.VI



    
