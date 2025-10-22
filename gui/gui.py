
"""
GUI for manipulation experiment- main page and hyperparameter page.
@author: Ganesh Narasimha
"""

import time
import threading
import sys, time, os, threading
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
import random
import string

from expt_utils import get_latest_file
from gym import Env
import io
import os
import numpy as np
import matplotlib.pyplot as plt



class MainPage(QtWidgets.QMainWindow):
    """
    Main GUI window for the manipulation experiment- Displays images, plots, and messages.
    """

    
    message = QtCore.pyqtSignal(int) # this is some signals window on the GUI:
    img1_sig = QtCore.pyqtSignal(str)
    img2_sig = QtCore.pyqtSignal(str)
    img3_sig = QtCore.pyqtSignal(str)
    plot_sig = QtCore.pyqtSignal(object)

    transmit_attrs = QtCore.pyqtSignal(dict)              # signal to send attributes to other pages.

    def __init__(self, train_model, env, agent, buffer, n_episodes, n_transitions, batch_size, train_start, updates_per_step, epsilon, func_attrs):
        super().__init__()
        self.setWindowTitle("Manipulation Experiment")
        self.setGeometry(1100, 100, 700, 900)

        # --- Internal state (not UI) ---
        self._iter = 0 # intenal tracking counter
        self._iteration = 0
        self._running = False
        self._stop_event = None
        self._thread = None
        self.result_container = {}
        self._agent = agent
        self._train_model = train_model
        self._env = env
        self._expt_dir = env.expt_dir if env is not None else None
        self._expt_log_dir = os.path.join(self._expt_dir, 'expt_log', env.expt_name)
        self._buffer = buffer
        self._n_episodes = n_episodes
        self._n_transitions = n_transitions
        self._batch_size = batch_size
        self._train_start = train_start
        self._updates_per_step = updates_per_step
        self.result_msg = None
        self.connection_lock = threading.Lock()
        self._func_attrs = func_attrs if func_attrs is not None else {}
        self._epsilon = epsilon
        self._recency =  float(self._func_attrs.get("recency_factor", 0.0))

        # --- Central widget + layouts ---
        central = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        # Display label
        self.label = QtWidgets.QLabel(self._fmt(), parent=central)
        self.label.setAlignment(Qt.AlignCenter)
        f = self.label.font(); f.setFamily("Roboto"); f.setPointSize(21); f.setBold(True); self.label.setFont(f)
        vbox.addWidget(self.label, stretch=1)

        # Display messages  
        self.msg_label = QtWidgets.QLabel(self._fmt_msg(self.result_msg), parent=central)
        self.msg_label.setAlignment(Qt.AlignLeft)
        self.msg_label.setContentsMargins(10, 5, 10, 5)  
        f = self.msg_label.font(); f.setFamily("Times New Roman"); f.setPointSize(16); self.msg_label.setFont(f)        
        vbox.addWidget(self.msg_label, stretch=1)

        # Image labels-hbox1
        hbox1 = QtWidgets.QHBoxLayout()
        self.img_label1 = QtWidgets.QLabel(parent=central)
        self.img_label1.setAlignment(Qt.AlignCenter)
        self.img_label1.setText("No image")
        self.img_label1.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.img_label1.setFixedSize(300, 300)

        self.img_label2 = QtWidgets.QLabel(parent=central)
        self.img_label2.setAlignment(Qt.AlignCenter)
        self.img_label2.setText("No image")
        self.img_label2.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.img_label2.setFixedSize(300, 300)
        
        hbox1.addWidget(self.img_label1)
        hbox1.addWidget(self.img_label2)
        vbox.addLayout(hbox1)

        # Image labels-hbox2
        hbox2 = QtWidgets.QHBoxLayout()
        self.img_label3 = QtWidgets.QLabel(parent=central)
        self.img_label3.setAlignment(Qt.AlignCenter)
        self.img_label3.setText("No image")
        self.img_label3.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.img_label3.setFixedSize(300, 300)

        self.plot_label = QtWidgets.QLabel(parent=central)
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setText("No plot")
        self.plot_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.plot_label.setFixedSize(400, 300)

        hbox2.addWidget(self.img_label3)
        hbox2.addWidget(self.plot_label)
        vbox.addLayout(hbox2)

        # Buttons row
        row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start", parent=central)
        self.btn_stop  = QtWidgets.QPushButton("Stop",  parent=central)
        self.quit_btn = QtWidgets.QPushButton("Quit", parent=central)
        self.quit_btn.Alignment = Qt.AlignRight

        self.btn_start.setFixedSize(80, 40)
        self.btn_stop.setFixedSize(80, 40)
        self.quit_btn.setFixedSize(40, 30)

        # Style the buttons
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: green;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton:disabled {
                background-color: #b3e6b3; /* very light green when disabled */
                color: #ffffff;
            }
        """)

        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: red;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton:disabled {
                background-color: #f5b5b5; /* very light red when disabled */
                color: #ffffff;
            }
        """)

        self.quit_btn.setStyleSheet("""
            QPushButton {
                color: red;
                font-weight: bold;  /* optional */
            }
        """)


        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        row.addWidget(self.quit_btn)

        vbox.addLayout(row)

        self.setCentralWidget(central)

        # --- Wire signals ---
        self.btn_start.clicked.connect(self.start_counter)
        self.btn_stop .clicked.connect(self.stop_counter)
        self.quit_btn.clicked.connect(self.close)
        
        self.message.connect(self._info_callback)  # GUI-safe slot
        self.img1_sig.connect(self._on_image_from_worker)
        self.img2_sig.connect(self._on_image_from_worker)
        self.img3_sig.connect(self._on_image_from_worker)
        self.plot_sig.connect(self._on_plot_from_worker)

    # ---------- Slots ----------
    def start_counter(self):
        """Launch the background worker thread if not already running."""
        if self._running:
            return
        self._stop_event = threading.Event()

        # Pass a callback that only emits the Qt signal (safe from worker thread)
        self._thread = threading.Thread(target=self._train_model, 
                           args=(self._env, self._agent, self._buffer, self._n_episodes, self._n_transitions, self._batch_size, 
                                 self._train_start, self._updates_per_step, self._epsilon, self._recency, self._stop_event, self.result_container),
                           kwargs={'info_callback': self._info_callback, 'expt_log_dir': self._expt_log_dir, "get_func_attrs": self._send_func_attrs}
        )

        self._image_thread = threading.Thread(target=self.update_images_and_plot)

        self._thread.start()
        self._image_thread.start()


        self._running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_counter(self):
        """Signal the worker to stop and (optionally) join the thread."""
        if not self._running:
            return
        self._stop_event.set()

        # Joining is optional for responsiveness; here it's quick/safe.
        self._thread.join(timeout=1.5) # wait for atmost 1.5s for the thread to finish.
        self._image_thread.join(timeout=1.5)

        self._thread = None # this is to terminate the thread for cleanliness.
        self._image_thread = None
        self._running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _info_callback(self, info, func_attrs=None):

        self.result_msg = info.get('results', None)

        self.msg_label.setText(self._fmt_msg(self.result_msg)) # update results
        self.label.setText(self._fmt()) # update internal counter and display
        self._read_func_attrs(func_attrs) # read any updated hyperparams from the training function.

    def _read_func_attrs(self, func_attrs): # read the function attributes and transmit to other pages.
        if func_attrs is not None:
            self.transmit_attrs.emit(func_attrs)
            
    def _set_func_attrs(self, func_attrs): # set the function attributes from hyperparam page.
        if func_attrs is not None:
            with self.connection_lock:
                self._func_attrs = func_attrs

    def _send_func_attrs(self): # send the function attributes to the training function.
            with self.connection_lock:
                return self._func_attrs


    # --- Image handling (no resizing) ---
    def _display_image_if_exists(self, path: str, image_label: QtWidgets.QLabel):
        path = path.strip()
        time.sleep(1)  # slight delay to ensure file is ready
    
        if not path or not os.path.exists(path):
            image_label.setText(f"No image loaded at path: \n {path}")
            image_label.setPixmap(QtGui.QPixmap())  # clear
            image_label.adjustSize()
            return
        pix = QtGui.QPixmap(path)
        if pix.isNull():
            image_label.setText("Failed to load image")
            image_label.setPixmap(QtGui.QPixmap())
            image_label.adjustSize()
            return
        
        #ensure centering and scaling behavior
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setScaledContents(False) #keep aspect ratio via manual scaling
        
        #scale to current label size
        target = image_label.contentsRect().size()
        scaled = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label.setPixmap(scaled)


    def update_images_and_plot(self):
        while not self._stop_event.is_set():
            self._on_image_from_worker()
            self._on_plot_from_worker()
        


    def _on_image_from_worker(self):
        """Auto-apply path from worker; will trigger display if file exists."""
        # Setting text triggers _on_path_changed -> attempts display

        p1, p2, p3 = self._get_image_paths(self._env.expt_dir)
        
        self._display_image_if_exists(p1, self.img_label1)
        self._display_image_if_exists(p2, self.img_label2)
        self._display_image_if_exists(p3, self.img_label3)

  
    def _on_plot_from_worker(self):
        """Auto-apply plot from worker; will trigger display if file exists."""
        rewards = self.result_container.get('iter_rewards', None)
       
        
        if rewards is None:
            self.plot_label.setText(f"No plot data. Rewards: {rewards}")
            self.plot_label.setPixmap(QtGui.QPixmap())
            return       

        fig = self._plot_fig(rewards)

        # Setting text triggers _on_path_changed -> attempts display
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)  # save figure to memory buffer
        buf.seek(0)

        qimg = QtGui.QImage.fromData(buf.getvalue())
        pix = QtGui.QPixmap.fromImage(qimg)

        # Scale pixmap to fit label while keeping aspect ratio
        pix = pix.scaled(
            self.plot_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )

        self.plot_label.setPixmap(pix)   # show on label
        self.plot_label.setText("")
        self.plot_label.adjustSize()     # label adopts pixmap size


    # ---------- Helper ----------
    def _fmt(self) -> str:
        self._iter = self._iteration
        iter_str = f"Iteration: {self._iter}"
        
        message = iter_str
       
        return message
    
    
    def _fmt_msg(self, result) -> str:

        if result is None:
            return "No results yet."


        prev_result = result.get('previous', None)
        current_iter_results = result.get('current', None)

        msg1 = ""
        msg2 = ""

        if prev_result is not None:
            prev_iteration = prev_result.get('iteration', None)
            prev_action_params = prev_result.get('action_params', None)
            prev_reward = prev_result.get('reward', None)
            prev_disp = prev_result.get('disp', None)

            msg1 = f"Previous Iteration: {prev_iteration}\n"
            if prev_action_params is not None:
                tb, tsp, ts = prev_action_params
                msg1 += f"Action params: Bias = {tb:.3f} V,\tSetpoint = {tsp:.2f} nA,\tspeed = {ts:.2f} nm/s\n"

            if prev_reward is not None and prev_disp is not None:
                msg1 += f"Reward: {prev_reward:.2f}, \tnorm_displacement: {prev_disp:.3f}\n"

        
        if current_iter_results is not None:
            current_iteration = current_iter_results.get('iteration', None)

            self._iteration = current_iteration
            
            current_action_params = current_iter_results.get('action_params', None)
            start_session = current_iter_results.get('start_session', None)

            msg2 = f"Current iteration: {current_iteration}\n"
            if current_action_params is not None:
                tb, tsp, ts = current_action_params
                msg2 += f"Action params: Bias = {tb:.3f} V,\tSetpoint = {tsp:.2f} nA,\tspeed = {ts:.2f} nm/s\n"
            msg2 += f"start new session: {start_session}\n"

        message = msg2 +"\n" + msg1
        return message



    @staticmethod
    def _get_image_paths(expt_dir):
        latest_name = get_latest_file(expt_dir) if expt_dir else None
        start = time.time()

        if expt_dir is None or latest_name is None:
            print("Experiment directory or latest file not found.")
            return None, None, None
        
        while True:

            img_path_1 = os.path.join(expt_dir,  latest_name.replace('.sxm', '_detectCO.jpg'))
            img_path_2 = os.path.join(expt_dir,  latest_name.replace('.sxm', '_assigned.jpg'))
            img_path_3 = os.path.join(expt_dir,  latest_name.replace('.sxm', '_path.jpg'))

            if os.path.exists(img_path_1) and os.path.exists(img_path_2) and os.path.exists(img_path_3):
                break

            # Wait for a timeout
            if time.time() - start > 5:  # timeout after 5 seconds
                break
    
        return img_path_1, img_path_2, img_path_3
    

    @staticmethod
    def _plot_fig(rewards):

        y = np.asarray(rewards)
        x = np.arange(1, len(y) + 1)

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, y, 'o-')  # corrected x,y order
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Reward')
        ax.tick_params(axis='both', labelsize=14)

        fig.tight_layout()
        plt.close(fig)  # Close the figure to free memory
        
        return fig
    






class HyperparamPage(QtWidgets.QWidget):

    """
    Page to set hyperparameters of the environment and training functions.

    """
    
    transmit_hp_attrs = QtCore.pyqtSignal(dict)

    def __init__(self, _hp_attrs=None):
        super().__init__()
        self.connection_lock = threading.Lock()
        self._hp_attrs = _hp_attrs if _hp_attrs is not None else {}
        self._updates_per_step = self._hp_attrs.get('updates_per_step', 1) if self._hp_attrs else 1
        self._epsilon = float(self._hp_attrs.get('epsilon', 0.01))   
        self._recency = float(self._hp_attrs.get('recency_factor', 1))
        self._batch_size = int(self._hp_attrs.get('batch_size', 32))

        detect_dict = self._hp_attrs.get('detect_dict', {})
        _target_n = detect_dict.get('target_n')
        self._target_n = int(_target_n) if _target_n is not None else 0
        self._use_prev = bool(detect_dict.get('use_prev', False))

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ===== Controls =====
        controls_box = QtWidgets.QGroupBox("Controls")
        controls_box.setAlignment(Qt.AlignLeft)
        controls = QtWidgets.QFormLayout(controls_box)
        controls.setLabelAlignment(Qt.AlignRight)
        controls.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        controls.setHorizontalSpacing(12)
        controls.setVerticalSpacing(6)

        self.spin1 = QtWidgets.QSpinBox()
        self.spin1.setRange(1, 10)
        self.spin1.setValue(self._updates_per_step)
        self.spin1.setFixedWidth(80)
        self.spin1.setAlignment(Qt.AlignRight)
        controls.addRow("Updates per step:", self.spin1)

        self.spin_eps = QtWidgets.QDoubleSpinBox()
        self.spin_eps.setRange(0.0, 1.0)
        self.spin_eps.setSingleStep(0.01)
        self.spin_eps.setDecimals(3)
        self.spin_eps.setValue(self._epsilon)
        self.spin_eps.setFixedWidth(80)
        self.spin_eps.setAlignment(Qt.AlignRight)
        controls.addRow("Epsilon (0–1):", self.spin_eps)

        self.spin_recency = QtWidgets.QDoubleSpinBox()
        self.spin_recency.setRange(0.0, 10.0)
        self.spin_recency.setSingleStep(0.1)
        self.spin_recency.setDecimals(2)
        self.spin_recency.setValue(self._recency)
        self.spin_recency.setFixedWidth(80)
        self.spin_recency.setAlignment(Qt.AlignRight)
        controls.addRow("Recency factor (0–10):", self.spin_recency)

        self.spin_batch = QtWidgets.QSpinBox()
        self.spin_batch.setRange(1, 1000000)
        self.spin_batch.setSingleStep(1)
        self.spin_batch.setValue(self._batch_size)
        self.spin_batch.setFixedWidth(80)
        self.spin_batch.setAlignment(Qt.AlignRight)
        controls.addRow("Batch size:", self.spin_batch)

        # New: target_n (0–100)
        self.spin_target = QtWidgets.QSpinBox()
        self.spin_target.setRange(0, 100)
        self.spin_target.setSingleStep(1)
        self.spin_target.setValue(self._target_n)
        self.spin_target.setFixedWidth(80)
        self.spin_target.setAlignment(Qt.AlignRight)
        controls.addRow("N_Molecules (0–100):", self.spin_target)

        # New: use_prev toggle
        self.chk_use_prev = QtWidgets.QCheckBox("Use previous detections")
        self.chk_use_prev.setChecked(self._use_prev)
        controls.addRow("Use prev label:", self.chk_use_prev)

        root.addWidget(controls_box)

        # ===== Displays =====
        displays_box = QtWidgets.QGroupBox("Displays")
        displays_box.setAlignment(Qt.AlignLeft)
        displays = QtWidgets.QGridLayout(displays_box)
        displays.setHorizontalSpacing(12)
        displays.setVerticalSpacing(6)

        def make_tx_label(text, bold=True, pts=14, h=28):
            lbl = QtWidgets.QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            f = lbl.font(); f.setPointSize(pts); f.setBold(bold)
            lbl.setFont(f); lbl.setFixedHeight(h)
            return lbl

        def make_rx_label(text="msg waiting", pts=18, h=24):
            lbl = QtWidgets.QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            f = lbl.font(); f.setPointSize(pts); f.setBold(False)
            lbl.setFont(f); lbl.setFixedHeight(h)
            return lbl

        hdr_style = "font-weight:600;"

        # updates_per_step
        name_inc = QtWidgets.QLabel("Transmitted updates_per_step:"); name_inc.setStyleSheet(hdr_style)
        name_msg = QtWidgets.QLabel("Received updates_per_step:");    name_msg.setStyleSheet(hdr_style)
        self.label = make_tx_label(str(self._updates_per_step))
        self.label2 = make_rx_label("Msg waiting")

        # epsilon
        name_eps_tx = QtWidgets.QLabel("Transmitted epsilon:"); name_eps_tx.setStyleSheet(hdr_style)
        name_eps_rx = QtWidgets.QLabel("Received epsilon:");    name_eps_rx.setStyleSheet(hdr_style)
        self.label_eps = make_tx_label(f"{self._epsilon:.3f}")
        self.label2_eps = make_rx_label("eps msg waiting")

        # recency
        name_rec_tx = QtWidgets.QLabel("Transmitted recency:"); name_rec_tx.setStyleSheet(hdr_style)
        name_rec_rx = QtWidgets.QLabel("Received recency:");    name_rec_rx.setStyleSheet(hdr_style)
        self.label_recency = make_tx_label(f"{self._recency:.2f}")
        self.label2_recency = make_rx_label("recency msg waiting")

        # batch size
        name_bs_tx = QtWidgets.QLabel("Transmitted batch size:"); name_bs_tx.setStyleSheet(hdr_style)
        name_bs_rx = QtWidgets.QLabel("Received batch size:");   name_bs_rx.setStyleSheet(hdr_style)
        self.label_batch = make_tx_label(str(self._batch_size))
        self.label2_batch = make_rx_label("batch msg waiting")

        # target_n
        name_tn_tx = QtWidgets.QLabel("Transmitted n_molecules:"); name_tn_tx.setStyleSheet(hdr_style)
        name_tn_rx = QtWidgets.QLabel("Received n_molecules:");   name_tn_rx.setStyleSheet(hdr_style)
        self.label_target = make_tx_label(str(self._target_n))
        self.label2_target = make_rx_label("target_n msg waiting")

        # use_prev
        name_up_tx = QtWidgets.QLabel("Transmitted use_prev:"); name_up_tx.setStyleSheet(hdr_style)
        name_up_rx = QtWidgets.QLabel("Received use_prev:");   name_up_rx.setStyleSheet(hdr_style)
        self.label_use_prev = make_tx_label(str(self._use_prev))
        self.label2_use_prev = make_rx_label("use_prev msg waiting")

        # Grid placement
        displays.addWidget(name_inc, 0, 0, alignment=Qt.AlignRight)
        displays.addWidget(self.label, 0, 1)
        displays.addWidget(name_msg, 1, 0, alignment=Qt.AlignRight)
        displays.addWidget(self.label2, 1, 1)

        displays.addWidget(name_eps_tx, 2, 0, alignment=Qt.AlignRight)
        displays.addWidget(self.label_eps, 2, 1)
        displays.addWidget(name_eps_rx, 3, 0, alignment=Qt.AlignRight)
        displays.addWidget(self.label2_eps, 3, 1)

        displays.addWidget(name_rec_tx, 4, 0, alignment=Qt.AlignRight)
        displays.addWidget(self.label_recency, 4, 1)
        displays.addWidget(name_rec_rx, 5, 0, alignment=Qt.AlignRight)
        displays.addWidget(self.label2_recency, 5, 1)

        displays.addWidget(name_bs_tx, 6, 0, alignment=Qt.AlignRight)
        displays.addWidget(self.label_batch, 6, 1)
        displays.addWidget(name_bs_rx, 7, 0, alignment=Qt.AlignRight)
        displays.addWidget(self.label2_batch, 7, 1)

        displays.addWidget(name_tn_tx, 8, 0, alignment=Qt.AlignRight)
        displays.addWidget(self.label_target, 8, 1)
        displays.addWidget(name_tn_rx, 9, 0, alignment=Qt.AlignRight)
        displays.addWidget(self.label2_target, 9, 1)

        displays.addWidget(name_up_tx, 10, 0, alignment=Qt.AlignRight)
        displays.addWidget(self.label_use_prev, 10, 1)
        displays.addWidget(name_up_rx, 11, 0, alignment=Qt.AlignRight)
        displays.addWidget(self.label2_use_prev, 11, 1)

        root.addWidget(displays_box)
        root.addStretch(1)

        # Wiring
        self.spin1.valueChanged.connect(self._on_updates_changed)
        self.spin_eps.valueChanged.connect(self._on_epsilon_changed)
        self.spin_recency.valueChanged.connect(self._on_recency_changed)
        self.spin_batch.valueChanged.connect(self._on_batch_changed)
        self.spin_target.valueChanged.connect(self._on_target_changed)
        self.chk_use_prev.stateChanged.connect(self._on_use_prev_changed)

        # Ensure nested dict exists on init
        self._ensure_detect_dict()

    # ---- slots ----
    @QtCore.pyqtSlot(int)
    def _on_updates_changed(self, val: int):
        self._updates_per_step = int(val)
        self.label.setText(str(self._updates_per_step))
        self._hp_attrs['updates_per_step'] = self._updates_per_step
        self.transmit_attrs()

    @QtCore.pyqtSlot(float)
    def _on_epsilon_changed(self, val: float):
        self._epsilon = float(val)
        self.label_eps.setText(f"{self._epsilon:.3f}")
        self._hp_attrs['epsilon'] = self._epsilon
        self.transmit_attrs()

    @QtCore.pyqtSlot(float)
    def _on_recency_changed(self, val: float):
        self._recency = float(val)
        self.label_recency.setText(f"{self._recency:.2f}")
        self._hp_attrs['recency_factor'] = self._recency
        self.transmit_attrs()

    @QtCore.pyqtSlot(int)
    def _on_batch_changed(self, val: int):
        self._batch_size = int(val)
        self.label_batch.setText(str(self._batch_size))
        self._hp_attrs['batch_size'] = self._batch_size
        self.transmit_attrs()

    @QtCore.pyqtSlot(int)
    def _on_target_changed(self, val: int):
        self._target_n = int(val)
        self.label_target.setText(str(self._target_n))
        self._ensure_detect_dict()
        self._hp_attrs['detect_dict']['target_n'] = self._target_n
        self.transmit_attrs()

    @QtCore.pyqtSlot(int)
    def _on_use_prev_changed(self, state: int):
        self._use_prev = (state == Qt.Checked)
        self.label_use_prev.setText(str(self._use_prev))
        self._ensure_detect_dict()
        self._hp_attrs['detect_dict']['use_prev'] = self._use_prev
        self.transmit_attrs()

    # ---- helpers ----
    def _ensure_detect_dict(self):
        if 'detect_dict' not in self._hp_attrs or not isinstance(self._hp_attrs['detect_dict'], dict):
            self._hp_attrs['detect_dict'] = {}
        self._hp_attrs['detect_dict']['target_n'] = self._target_n
        self._hp_attrs['detect_dict']['use_prev'] = self._use_prev

    def transmit_attrs(self):
        self.transmit_hp_attrs.emit(self._hp_attrs)
    
    def set_attrs(self, attrs: dict):
        with self.connection_lock:
            self._f_attrs = attrs

        r_updates_per_step = self._f_attrs.get('updates_per_step', 'no_updates') if self._f_attrs is not None else 'no_msg'
        self.label2.setText(str(r_updates_per_step))

        r_epsilon = self._f_attrs.get('epsilon', 'no_epsilon') if self._f_attrs is not None else 'no_msg'
        self.label2_eps.setText(str(r_epsilon))

        r_recency = self._f_attrs.get('recency_factor', 'no_recency') if self._f_attrs is not None else 'no_msg'
        self.label2_recency.setText(str(r_recency))

        r_batch = self._f_attrs.get('batch_size', 'no_batch') if self._f_attrs is not None else 'no_msg'
        self.label2_batch.setText(str(r_batch))

        d = self._f_attrs.get('detect_dict', {}) if self._f_attrs is not None else {}
        r_target = d.get('target_n', 'no_target_n')
        r_use_prev = d.get('use_prev', 'no_use_prev')

        self.label2_target.setText(str(r_target))
        self.label2_use_prev.setText(str(r_use_prev))





# ---------------- Main window with multiple PAGES ----------------
class Manipulation_GUI(QtWidgets.QMainWindow):
    def __init__(self, train_model, env, agent, buffer, n_episodes, n_transitions, batch_size, train_start, updates_per_step, epsilon, func_attrs):
        super().__init__()
        self.setWindowTitle("STM Manipulation using RL")
        self.resize(600, 400)

        
        # ========= Custom header area =========
        header_widget = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 6, 10, 6)
        header_layout.setSpacing(10)


        # --- Logo on the left ---
        logo_label = QtWidgets.QLabel()
        logo_pixmap = QtGui.QPixmap("gui/ornl_logo.jpg")  # <-- replace with your image path
        logo_pixmap = logo_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setFixedSize(45, 45)
        header_layout.addStretch(1)
        header_layout.addWidget(logo_label, alignment=Qt.AlignLeft)

        # --- Bold title text ---
        title_label = QtWidgets.QLabel("STM Manipulation using RL")
        title_font = QtGui.QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #222;")  # optional color
        header_layout.addWidget(title_label, alignment=Qt.AlignCenter)

        header_layout.addStretch(1)

        # QTabWidget to host multiple pages
        tabs = QtWidgets.QTabWidget()
        self.main_page = MainPage(train_model, env, agent, buffer, n_episodes, n_transitions, batch_size, train_start, updates_per_step, epsilon, func_attrs)
        self.hyperparam_page = HyperparamPage(func_attrs)

        self.hyperparam_page.transmit_hp_attrs.connect(self.main_page._set_func_attrs)  # Connect signal to slot
        #self.counter_page.set_increment(self.increment_page.spin_inc.value()) 
        self.main_page.transmit_attrs.connect(self.hyperparam_page.set_attrs)  # Connect counter attributes to increment page
        

        tabs.addTab(self.main_page, "Main Page")
        tabs.addTab(self.hyperparam_page, "Hyperparameters")

         # ========= Stack header + tabs =========
        central_widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central_widget)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(header_widget)
        vbox.addWidget(tabs)

        self.setCentralWidget(central_widget)
