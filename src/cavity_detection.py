import qcodes as qc
from qcodes.dataset import(load_by_run_spec, initialise_or_create_database_at)
from qcodes.dataset.experiment_container import Experiment
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import scipy.signal as signal

class S21_data():
        # Extract amplitude and phase from S21 data
        def __init__(self, db_path, run_id: int, GUID = None):

            assert run_id >= 0, "Run ID must be provided"
            self.db_path = db_path
            self.run_id = run_id
            self.GUID = GUID
            self.dataset = None
            self.s21_data = None

            databse_loc = initialise_or_create_database_at(db_path)
        # Load the dataset
            if GUID is not None:
                self.dataset = load_by_run_spec(GUID=GUID)
                self.dataset = load_by_run_spec(captured_run_id=22)
            ds = dataset.to_xarray_dataset()
            ds
            self.phase = np.angle(self.s21_data)
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
        self. S21_data_ampliftude =

    def cavity_identification(self, threshold=0.5,height=0.5):
        # Identify cavities based on S21 data
        peaks, properties = signal.find_peaks(self.s21_data, height=threshold)
        # TODO: Add more sophisticated cavity identification logic. For Fano resonance
        self.cavities = peaks
        return peaks

    def prepare_runfile_for_qumin(self, setup_file_path):
        # Prepare the setup file for QCoDeS
        with open(setup_file_path, 'w') as f:
            f.write("# Setup file for QCoDeS\n")
            f.write(f"experiment_name = '{self.exp_name}'\n")
            f.write(f"database_path = '{self.db_path}'\n")

    def harmonic_analysis(self, cavity_index):
        # Perform harmonic analysis on the identified cavity.
        if self.cavities is None:
            raise ValueError("No cavities identified. Run cavity_identification first.")

        cavity_data = self.s21_data[self.cavities[cavity_index]]
        frequency = np.linspace(0, 10, len(cavity_data))
        # Perform FFT
        fft_data = np.fft.fft(cavity_data)

    def process_data(self):
        # Process the S21 data (e.g., apply a filter)
        b, a = signal.butter(4, 0.2)
        filtered_s21 = signal.filtfilt(b, a, self.s21_data)