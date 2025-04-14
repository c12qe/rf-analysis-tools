import qcodes as qc
from qcodes.dataset import(load_by_run_spec, initialise_or_create_database_at)
from qcodes.dataset.experiment_container import Experiment
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import scipy.signal as signal


# Data loading and processing
class S21_Analysis():
    def __init__(self, db_path, exp_name):
        self.db_path = db_path
        self.exp_name = exp_name
        self.experiment = None
        self.dataset = None
        self.s21_data = None

    def load_data(self):
        # Load the database
        qc.config.database.connect()
        qc.config.database.initialise_or_create_database_at(self.db_path)

        # Load the experiment
        self.experiment = Experiment(self.exp_name)

        # Load the dataset
        self.dataset = load_by_run_spec(self.experiment.name, 1)

        # Extract S21 data
        self.s21_data = self.dataset.get_data('S21')

    def process_data(self):
        # Process the S21 data (e.g., apply a filter)
        b, a = signal.butter(4, 0.2)
        filtered_s21 = signal.filtfilt(b, a, self.s21_data)