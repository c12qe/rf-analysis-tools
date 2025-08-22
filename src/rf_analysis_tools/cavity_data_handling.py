import qcodes as qc
from qcodes.dataset import(load_by_run_spec, initialise_or_create_database_at,experiments)
from qcodes.dataset.experiment_container import Experiment
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import scipy.signal as signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, welch
from typing import Tuple, List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit

# import probst tool
class Sij_data:

    """
    Class to handle S-parameter data from a VNA (Vector Network Analyzer).
    It loads the data from a database and extracts the S21 amplitude and phase.
    """

    # Extract amplitude and phase from S21 data
    def __init__(self, db_path, run_id: list = None):
        self.db_path = db_path
        self.all_runs = run_id
        self.pna_frequency = None
        self.linear_magnitude = None
        self.phase = None
        self.unwrapped_phase = None
        self.dB_magnitude = None
        self.I = None
        self.Q = None


    def load_single_data(self):
        '''
        Loads single datafile or multiple datafiles from the database and concatenated.
        :return: None
        '''
        assert self.run_id >= 0, "Run ID must be provided"
        # Load the database
        initialise_or_create_database_at(self.db_path)
        # Load the dataset

        freq = np.array([])
        linear_magnitude = np.array([])
        db_magnitude = np.array([])
        phase = np.array([])
        unwrapped_phase = np.array([])
        i = np.array([])
        q = np.array([])

        for indx_ii, run_id in enumerate(self.all_runs):
            data = load_by_run_spec(captured_run_id=run_id)
            ds = data.to_xarray_dataset()

            if indx_ii == 0:
                freq = ds['pna_frequency_axis'].values
                linear_magnitude = ds.data_vars['pna_tr1_linear_magnitude'].data
                phase = ds.data_vars['pna_tr1_phase'].data
                unwrapped_phase = ds.data_vars['pna_tr1_unwrapped_phase'].data
                db_magnitude = ds.data_vars['pna_tr1_magnitude'].data
                total_transmission = linear_magnitude * np.exp(1j * phase * np.pi / 180)
                i = np.real(total_transmission)
                q = np.imag(total_transmission)
            else:
                indv_magnitude = ds.data_vars['pna_tr1_linear_magnitude'].data
                indv_phase = ds.data_vars['pna_tr1_phase'].data

                freq = np.concatenate((freq, ds['pna_frequency_axis'].values))
                linear_magnitude = np.concatenate((linear_magnitude, indv_magnitude))
                phase = np.concatenate((phase, indv_phase))
                unwrapped_phase = np.concatenate((unwrapped_phase, ds.data_vars['pna_tr1_unwrapped_phase'].data))
                db_magnitude = np.concatenate((db_magnitude, ds.data_vars['pna_tr1_magnitude'].data))
                iq = indv_magnitude * np.exp(1j * indv_phase * np.pi / 180)

                i = np.concatenate((i, np.real(iq)))
                q = np.concatenate((q, np.imag(iq)))

        self.pna_frequency = freq
        self.linear_magnitude = linear_magnitude
        self.phase = phase
        self.unwrapped_phase = unwrapped_phase
        self.dB_magnitude = db_magnitude
        self.I = i
        self.Q = q

    def load_metadeta(self):
        '''Loads metadata from the database.Relevant parameters:
         temperature, vna_power, parse comment for extracting attenuation, '''
        pass

    def final_tabulated_data_output(self):
        '''Combine relevant metadata and S-parameter data into a single DataFrame for analysis.'''
        pass


    def prepare_runfile_for_qumin(self, setup_file_path):
        # Prepare the setup file for QCoDeS
        with open(setup_file_path, 'w') as f:
            f.write("# Setup file for QCoDeS\n")
            f.write(f"experiment_name = '{self.exp_name}'\n")
            f.write(f"database_path = '{self.db_path}'\n")


Sij_data_vna = Sij_data()
Sij_data_OPX = Sij_data()
