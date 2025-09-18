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
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit

# import probst tool
class resonator_data_visualization:
    """
    Class to handle resonator data visualization.
    It loads the data from a database and extracts the S21 amplitude and phase.
    """

    def __init__(self, db_path, file_name=None, file_type='pandas'):
        self.analyzed_data_path = db_path
        self.analyzed_data_file = file_name
        self.file_type = file_type

    def load_data(self):
        if self.file_type == 'pandas':
            self.data = pd.read_(os.path.join(self.analyzed_data_path, self.analyzed_data_file))
