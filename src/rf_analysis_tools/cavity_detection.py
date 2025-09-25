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

# Import scraps
import scraps as scr
from scraps import Resonator

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit

# import probst tool

class CavityAnalysisTool:
    def __init__(self, frequency_data: np.ndarray, amplitude_data: np.ndarray):
        """
        Initialize Cavity Analysis Tool with frequency and amplitude data

        Args:
            frequency_data (np.ndarray): Frequency domain data
            amplitude_data (np.ndarray): Corresponding amplitude data
        """
        self.frequency = frequency_data
        self.amplitude = amplitude_data
        self.cavity_peaks = None
        self.cavity_parameters = {}

    def identify_cavities(self,
                          prominence: float = 0.1,
                          distance: int = None) -> List[Dict[str, Any]]:
        """
        Identify cavity resonances in the frequency spectrum

        Args:
            prominence (float): Minimum prominence of peaks
            distance (int): Minimum distance between peaks

        Returns:
            List of cavity peak information dictionaries
        """
        # Find peaks in the amplitude spectrum
        peaks, properties = find_peaks(
            self.amplitude,
            prominence=prominence,
            distance=distance
        )

        # Store cavity peaks for further analysis
        self.cavity_peaks = {
            'indices': peaks,
            'frequencies': self.frequency[peaks],
            'amplitudes': self.amplitude[peaks],
            'prominences': properties['prominences']
        }

        # Prepare cavity information list
        cavity_info = []
        for i in range(len(peaks)):
            cavity_info.append({
                'peak_index': peaks[i],
                'frequency': self.frequency[peaks[i]],
                'amplitude': self.amplitude[peaks[i]],
                'prominence': properties['prominences'][i]
            })

        return cavity_info

    def harmonic_analysis(self) -> Dict[str, Any]:
        """
        Perform harmonic analysis of cavity resonances

        Returns:
            Dictionary with harmonic analysis results
        """
        if self.cavity_peaks is None:
            raise ValueError("Cavities must be identified first. Run identify_cavities() first.")

        # Calculate harmonic relationships
        fundamental_freq = self.cavity_peaks['frequencies'][0]
        harmonics = []

        for i, freq in enumerate(self.cavity_peaks['frequencies'][1:], 1):
            # Calculate harmonic ratio
            harmonic_ratio = freq / fundamental_freq
            harmonics.append({
                'order': i + 1,
                'frequency': freq,
                'expected_ratio': round(harmonic_ratio),
                'actual_ratio': harmonic_ratio,
                'deviation': abs(round(harmonic_ratio) - harmonic_ratio)
            })

        return {
            'fundamental_frequency': fundamental_freq,
            'harmonics': harmonics
        }



