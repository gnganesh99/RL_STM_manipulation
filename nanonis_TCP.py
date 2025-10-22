# @author: Ganesh Narasimha

""" Wrapper functions using Nanonis TCP/IP commands """

import os

import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
from IPython.display import clear_output

from nanonisTCP.nanonisTCP.Bias import Bias as Bias
from nanonisTCP.nanonisTCP.ZController import ZController as ZCtrl
from nanonisTCP.nanonisTCP.Scan import Scan as Scan
from nanonisTCP.nanonisTCP.FolMe import FolMe as FolMe
from nanonisTCP.nanonisTCP import nanonisTCP as nanonisTCP
from nanonisTCP.nanonisTCP.BiasSpectr import BiasSpectr as BiasSpectr
from nanonisTCP.nanonisTCP.TipRec import TipRec as TipRec


class Nanonis_TCP():

    def __init__(self):

        self.host_ip = '127.0.0.1'
        self.port = 6501
        self.port_open = False
        self.freeze_port_status = False

    def NTCP_open(self):
        """ Opens the Nanonis TCP connection if not already open. """

        if not self.port_open and not self.freeze_port_status:            
            self.NTCP = nanonisTCP(self.host_ip, self.port)
            self.port_open = True
        else:
            pass

        return
    
    def NTCP_close(self):
        """ Closes the Nanonis TCP connection if open and not frozen. """

        if self.port_open and not self.freeze_port_status:
            self.NTCP.close_connection()
            self.port_open = False
        else:
            pass

        return

    def set_bias(self, set_bias):

        """ 
        Sets the tip bias voltage.
        
        Args:
            set_bias: bias voltage to be set (in Volts)

        """
     
        self.NTCP_open()

        try:
            bias = Bias(self.NTCP)
            bias.Set(set_bias)

        finally:
            self.NTCP_close()

        return
        

    def get_bias(self):
        """ 
        Gets the tip bias voltage.

        Returns:
            bias voltage (in Volts)
        """

        self.NTCP_open()

        try:
            bias = Bias(self.NTCP)
            result = bias.Get()
        
        finally:
            self.NTCP_close()

        return result


    def set_setpoint(self, setpoint, limit = 100E-9):
        """ 
        Sets the Z controller setpoint.
        Args:
            setpoint: setpoint to be set (in Amps)
            limit: upper limit for the setpoint (in Amps)

        Returns:
            Error message if setpoint is higher than limit.

        """

        if setpoint > limit:

            return "Sepoint is higher than limit"

        self.NTCP_open()

        try:
            zctrl = ZCtrl(self.NTCP)
            zctrl.SetpntSet(setpoint)

        finally:
            self.NTCP_close()

        return
    

    def get_setpoint(self):

        """
        Gets the Z controller setpoint.
        Returns:
            setpoint (in Amps)

        """


        self.NTCP_open()

        try:
            zctrl = ZCtrl(self.NTCP)
            result = zctrl.SetpntGet()

        finally:
            self.NTCP_close()

        return result

    def scan(self, scan_direction = "up"):

        """
        Starts a scan in the specified direction.
        Args:
            scan_direction: "up" or "down" (default is "up")
        """
        

        self.NTCP_open()

        try:
            scan = Scan(self.NTCP)
            scan.Action(scan_action = "start", scan_direction = scan_direction)
            
            #Waits till the end of scan
            scan_status, _,_,  = scan.WaitEndOfScan()
            
        finally:
            self.NTCP_close()

        # return true if scan complete (scan_status = False)
        return not scan_status

    def scan_Frameset(self, x_center, y_center, frame_x, frame_y, angle):

        """
        Sets the scan frame parameters.
        Args:
            x_center: X center of the scan frame (in meters)
            y_center: Y center of the scan frame (in meters)
            frame_x: Frame size in X direction (in meters)
            frame_y: Frame size in Y direction (in meters)
            angle: Angle of the scan frame (in degrees)

        
        """
        
        self.NTCP_open()


        try:
            scan = Scan(self.NTCP)
            scan.FrameSet(x_center, y_center, frame_x, frame_y, angle)
            
            
        finally:
            self.NTCP_close()

        
        return
    
    def scan_Frameget(self):

        """
        Gets the scan frame parameters.
        Returns:
            x_center: X center of the scan frame (in meters)
            y_center: Y center of the scan frame (in meters)
            frame_x: Frame size in X direction (in meters)
            frame_y: Frame size in Y direction (in meters)
            angle: Angle of the scan frame (in degrees)

        """
        
        self.NTCP_open()

        try:
            scan = Scan(self.NTCP)
            x, y, w, h, angle = scan.FrameGet()

        finally:
            self.NTCP_close()

        return x,y, w, h, angle

    def folme(self, position, speed = 1E-9, custom_speed=True, wait_end_of_move=True):

        """
        Moves the tip to the specified (x,y) position using the folme module.
        Args:
            position: (x, y) coordinates to move to (in meters)
            speed: Speed of the movement (in meters/second)
            custom_speed: If True, uses the specified speed. If False, uses the current speed setting in FolMe.
            wait_end_of_move: If True, waits for the movement to complete before returning.
        
        """



        self.NTCP_open()

        try:

            folme = FolMe(self.NTCP)
            folme.SpeedSet(speed = speed, custom_speed=custom_speed)
            time.sleep(0.5)

            folme.XYPosSet(position[0], position[1], Wait_end_of_move=wait_end_of_move)
    

        finally:
            self.NTCP_close()

        return True


    def folme_oversample_set(self, oversample):
        """
        Sets the oversampling factor in the FolMe module.
        Args:
            oversample: Oversampling factor (integer)
        
        """


        self.NTCP_open()

        try:
            folme = FolMe(self.NTCP)
            folme.OversamplSet(oversample)
        finally:
            self.NTCP_close()

        return
    
    def folme_oversample_get(self):
        """
        Gets the oversampling factor and sampling rate in the FolMe module.
        Returns:
            oversample: Oversampling factor (integer)
            sampling_rate: Sampling rate (in Hz)
    
        """


        self.NTCP_open()

        try:
            folme = FolMe(self.NTCP)
            oversample, sampling_rate = folme.OversamplGet()
        finally:
            self.NTCP_close()

        return oversample, sampling_rate


    def BufferSizeSet(self, buffer_size):
        """
        Sets the size of the TipRec buffer.
        Args:
            buffer_size: Size of the buffer (integer)
        """
        self.NTCP_open()
        try:

            tiprec = TipRec(self.NTCP)
            tiprec.BufferSizeSet(buffer_size)

        finally:
            self.NTCP_close()

        return
    
    
    def BufferClear(self):
        """ Clears the TipRec buffer."""

        self.NTCP_open()

        try:

            tiprec = TipRec(self.NTCP)
            tiprec.BufferClear()

        finally:
            self.NTCP_close()

    def Tiprec_DataGet(self):
        """ 
        Gets the recorded data from the TipRec buffer.

        data_dict: A dictionary containing the following keys:
            'channel_indices': List of channel indices.
            'n_rows': Number of rows in the data. This is the number of recorded channels
            'n_cols': Number of columns in the data. This is the size of the buffer
            'data': 2D list containing the recorded data.
        
        """
        self.NTCP_open()
        try:

            tiprec = TipRec(self.NTCP)
            data_dict = tiprec.DataGet()

            return data_dict

        finally:
            self.NTCP_close()
            

    def Tiprec_DataSave(self, basename, clear_buffer = True):
        """
        Saves the recorded data from the TipRec buffer to a file.
        Args:
            basename: Base name for the saved file (string)
            clear_buffer: If True, clears the buffer after saving. Default is True.
        """
        
        self.NTCP_open()
        try:
            tiprec = TipRec(self.NTCP)
            tiprec.DataSave(basename, clear_buffer = clear_buffer)

        finally:
            self.NTCP_close()



    def slew_set_setpoint(self, setpoint, rate = 1E-9, limit = 100E-9):
        """
        Gradually changes the Z controller setpoint at a specified rate.
        
        Args:
            setpoint: Target setpoint to be set (in Amps)
            rate: Rate of change for the setpoint (in Amps per second). Default is 1nA/s
            limit: Upper limit for the setpoint (in Amps). Default is 100nA
            
        Returns:
            Error message if setpoint is higher than limit.
        """
        
        if setpoint > limit:

            return "Setpoint is higher than limit"



        self.NTCP_open()

        ztrl = ZCtrl(self.NTCP)

        try:

            diff = setpoint  - ztrl.SetpntGet()

            if diff > 0:
        
                while True:

                    current_setP = ztrl.SetpntGet()+rate

                    if current_setP < setpoint:
                        ztrl.SetpntSet(current_setP)

                    else:
                        ztrl.SetpntSet(setpoint)
                        break
                    time.sleep(1)


            else:

                while True:

                    current_setP = ztrl.SetpntGet() - rate

                    if current_setP > setpoint:
                        ztrl.SetpntSet(current_setP)

                    else:
                        ztrl.SetpntSet(setpoint)
                        break
                    time.sleep(1)
        
        finally:
            self.NTCP_close()

    
        return
    

    def slew_set_bias(self, set_bias, rate = 0.1, limit = 0.001):
        """
        Gradually changes the tip bias voltage at a specified rate.
        
        Args:
            set_bias: Target bias voltage to be set (in Volts)
            rate: Rate of change for the bias voltage (in Volts per second). Default is 0.1 V/s
            limit: Lower limit for the absolute bias voltage (in Volts). Default is 1mV
            
        Returns:
            Error message if bias is lower than limit or if bias ramps through 0V.
        """
        
        if np.abs(set_bias) < limit:

            return "Bias is lower than limit"


        self.NTCP_open()

        bias = Bias(self.NTCP)

        if bias.Get()*set_bias < 0:

            self.NTCP_close()
            return "Bias ramps thru 0 V. Check sign !!!"
        

        try:

            diff = set_bias  - bias.Get()

            if diff > 0:
        
                while True:

                    current_bias = bias.Get()+rate

                    if current_bias < set_bias:
                        bias.Set(current_bias)

                    else:
                        bias.Set(set_bias)
                        break
                    time.sleep(1)


            else:
        
                while True:

                    current_bias = bias.Get() - rate

                    if current_bias > set_bias:
                        bias.Set(current_bias)

                    else:
                        bias.Set(set_bias)
                        break
                    time.sleep(1)

        finally:
            self.NTCP_close()

    
        return
        



    def manipulation(self, initial_coords, final_coords, manipulation_params, default_params, 
                     bias_slew_rate = 0.03, setpoint_slew_rate = 10E-9,
                     bias_lower_limit = 0.001, setpoint_limit = 100E-9, get_buffer = False):
        """
        Performs tip manipulation by moving from initial to final coordinates with specified parameters.
        
        Args:
            initial_coords: (x, y) starting position coordinates (in meters)
            final_coords: (x, y) target position coordinates (in meters)
            manipulation_params: [bias, setpoint, speed] parameters for manipulation
            default_params: [bias, setpoint, speed] parameters to restore after manipulation
            bias_slew_rate: Rate of bias change during parameter updates (in V/s). Default is 0.03 V/s
            setpoint_slew_rate: Rate of setpoint change during parameter updates (in A/s). Default is 10nA/s
            bias_lower_limit: Lower limit for bias voltage (in Volts). Default is 1mV
            setpoint_limit: Upper limit for setpoint current (in Amps). Default is 100nA
            get_buffer: If True, records TipRec buffer data during manipulation. Default is False
            
        Returns:
            done: Boolean indicating if manipulation completed successfully
            buffer_data: TipRec buffer data (only if get_buffer=True)
        """
        
        done = False

        if np.abs(manipulation_params[0]) < bias_lower_limit or np.abs(default_params[0]) < bias_lower_limit:
            return "tip bias parameter is lower than the limit !!!"

        if np.abs(manipulation_params[1]) > setpoint_limit or np.abs(default_params[1]) > setpoint_limit:
            
            return "setpoint parameter is higher than the limit !!!"
        
        self.freeze_port_status = False
        self. NTCP_open()
        self.freeze_port_status = True
   

        
        try:
      
            
            folme = FolMe(self.NTCP)

            print("\tMoving to initial position")
            # Go to intial_position:
            folme.XYPosSet(initial_coords[0], initial_coords[1], Wait_end_of_move = True)

            #Updata manipulation parameters
            set_bias, set_setP, set_speed = manipulation_params

            print("\tUpdating manipulation parameters")
            try:
                self.slew_set_bias(set_bias, rate = bias_slew_rate)
            except:
                pass
        
            try:
                self.slew_set_setpoint(set_setP, rate = setpoint_slew_rate)
            except:
                pass
            

            if get_buffer:
                self.BufferClear()
                self.BufferSizeSet(1000_000)
                time.sleep(1)
        


            
            try:

                folme.SpeedSet(set_speed, custom_speed=True)

                print("\tMoving to target position")
                # Move to final position:
                folme.XYPosSet(final_coords[0], final_coords[1], Wait_end_of_move=True)
                time.sleep(0.2)
            except:
                pass

                
            #Get Tiprec buffer data
            buffer_data = None
            if get_buffer:
                buffer_data = self.Tiprec_DataGet()
                print(f"\t Tip rec buffer contains {buffer_data.get('n_cols')} points")


            print("\tBack to default parameters")
            #Update default parameters
            set_bias, set_setP, set_speed = default_params

            try:
                self.slew_set_bias(set_bias, rate = bias_slew_rate)
            except:
                pass
            try:
                self.slew_set_setpoint(set_setP, rate = setpoint_slew_rate)
            except:
                pass


            folme.SpeedSet(set_speed, custom_speed=True)        


        finally:

            self.freeze_port_status = False
            self.NTCP_close()
            print("\tManipulation complete")

        done = True  



        if get_buffer:
            return done, buffer_data
        
        return done
    

    def path_manipulation(self, path_coords, manipulation_params, default_params, 
                     bias_slew_rate = 0.03, setpoint_slew_rate = 10E-9,
                     bias_lower_limit = 0.001, setpoint_limit = 100E-9, get_buffer = False):
        """
        Performs tip manipulation along a path of multiple coordinates.
        
        Args:
            path_coords: List of (x, y) coordinate tuples defining the manipulation path (in meters)
            manipulation_params: [bias, setpoint, speed] parameters for manipulation
            default_params: [bias, setpoint, speed] parameters to restore after manipulation
            bias_slew_rate: Rate of bias change during parameter updates (in V/s). Default is 0.03 V/s
            setpoint_slew_rate: Rate of setpoint change during parameter updates (in A/s). Default is 10nA/s
            bias_lower_limit: Lower limit for bias voltage (in Volts). Default is 1mV
            setpoint_limit: Upper limit for setpoint current (in Amps). Default is 100nA
            get_buffer: If True, records TipRec buffer data during manipulation. Default is False
            
        Returns:            
            done: Boolean indicating if path manipulation completed successfully
            buffer_data: TipRec buffer data (only if get_buffer=True)
        """
        
        done = False

        if np.abs(manipulation_params[0]) < bias_lower_limit or np.abs(default_params[0]) < bias_lower_limit:
            return "tip bias parameter is lower than the limit !!!"

        if np.abs(manipulation_params[1]) > setpoint_limit or np.abs(default_params[1]) > setpoint_limit:
            
            return "setpoint parameter is higher than the limit !!!"
        
        self.freeze_port_status = False
        self. NTCP_open()
        self.freeze_port_status = True

        try:
       
            
            folme = FolMe(self.NTCP)

            print("\tMoving to initial position")
            # Go to intial_position:
            folme.XYPosSet(path_coords[0][0], path_coords[0][1], Wait_end_of_move = True)
            time.sleep(0.2)

            #Updata manipulation parameters
            set_bias, set_setP, set_speed = manipulation_params

            print("\tUpdating manipulation parameters")
            try:
                self.slew_set_bias(set_bias, rate = bias_slew_rate)
            finally:
                pass
        
            try:
                self.slew_set_setpoint(set_setP, rate = setpoint_slew_rate)
            finally:
                pass
            
            try:
                folme.SpeedSet(set_speed, custom_speed=True)
            except:
                pass

            if get_buffer:
                self.BufferClear()
                self.BufferSizeSet(1000_000)
                time.sleep(1)
        



            for point in range(1, len(path_coords)):

                if point != (len(path_coords)-1):                
                    print(f"\tMoving to position {point} of {len(path_coords)-1}")
                else:
                    print(f"\tMoving to target position {point} of {len(path_coords)-1}")

                # Move to final position:
                folme.XYPosSet(path_coords[point][0], path_coords[point][1], Wait_end_of_move=True)
                time.sleep(0.1)
                 
            #Get Tiprec buffer data
            buffer_data = None
            if get_buffer:
                buffer_data = self.Tiprec_DataGet()
                print(f"\t Tip rec buffer contains {buffer_data.get('n_cols')} points")



            print("\tBack to default parameters")
            #Updata default parameters
            set_bias, set_setP, set_speed = default_params

            try:
                self.slew_set_bias(set_bias, rate = bias_slew_rate)
            except:
                pass
            try:
                self.slew_set_setpoint(set_setP, rate = setpoint_slew_rate)
            except:
                pass

            
            folme.SpeedSet(set_speed, custom_speed=True)

        finally:

            self.freeze_port_status = False
            self.NTCP_close()
            print("\tManipulation complete")

        done = True  

        if get_buffer:
            return done, buffer_data

        return done


    def do_Bias_Spectr(self, V_start, V_end, num_points = 256, back_sweep = 0, num_sweeps = 0, z_offset = 0, save_all = 0, autosave = 0, basename = 'BiasSpec_'):

        """
        Performs a bias spectroscopy measurement.

        Parameters
        ----------
        V_start : float
            Starting voltage for the bias spectroscopy.
        V_end : float
            Ending voltage for the bias spectroscopy.
        num_points : int, default = 256
            Number of points to measure. Number of sweeps to measure and average. 0 means no change
        back_sweep : int, default = 0
            0 : No change
            1 : Don't acquire a backward sweep.
            2 : Acquire a backward sweep (in addition to the forward)
        num_sweeps : int, default = 0
            Number of sweeps to perform.0 means no change
        z_offset : float, default = 0
            Z-offset for the measurement.
        save_all : int, default = 0
            0 : no change
            1 : data from individual sweeps is saved, along with averaged data.
            2 : Individual sweeps are not saved, only the average of all of them is saved. This parameter only makes sense for multiple sweeps.
        autosave : int, default = 0
            0 : No change
            1 : Autosave data after acquisition
            2 : Don't autosave after acquisition
        basename : str, default = 'BiasSpec_'
            Base name for the saved files.

        """
        

        try:
            self.NTCP_open()
            bias_spectr = BiasSpectr(self.NTCP)
            bias_spectr.Open()
       
            bias_spectr.LimitsSet(V_start, V_end)
            bias_spectr.PropsSet(num_sweeps=num_sweeps, back_sweep=back_sweep,num_points=num_points,z_offset=z_offset,save_all=save_all,autosave=autosave,save_dialog=0)

            try:
                spec_data = bias_spectr.Start(get_data=True, save_base_name = basename)
            finally:
                time.sleep(0.5)

            spec_running = True

            while spec_running:

                spec_running  = bool(bias_spectr.StatusGet())
                time.sleep(0.5)

        finally:
            self.NTCP_close()

        
        return spec_data

        