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

import numpy as np
import matplotlib.pyplot as plt
from scraps import Resonator
import pandas as pd
import os
from scipy.optimize import curve_fit

# import probst tool

class Sij_data():
        # Extract amplitude and phase from S21 data
        def __init__(self, db_path, run_id: int = None):

            self.db_path = db_path
            self.run_id = run_id
            self.pna_frequency = None
            self.data_variable_keys = None
            self.linear_magnitude = None
            self.phase = None
            self.unwrapped_phase = None
            self.dB_magnitude= None
            self.I=None
            self.Q = None

    def Load_Single_Data(self):
        assert self.run_id >= 0, "Run ID must be provided"
        # Load the database
        initialise_or_create_database_at(self.db_path)
        # Load the dataset
        if self.run_id is None:
            print('Experiment list: 'experiments(initialise_or_create_database_at(self.db_path)))
            val = int(input('Input run id you want to load'))
            self.run_id = val

        data = load_by_run_spec(captured_run_id=self.run_id)
        ds = data.to_xarray_dataset()

        if ds.coords in ['pna_frequency_axis']:
            self.pna_frequency = ds['pna_frequency_axis'].values

        data_variables = list(ds.data_vars.keys())
        self.linear_magnitude = ds.data_vars['pna_tr1_linear_magnitude'].data
        self.phase = ds.data_vars['pna_tr1_phase'].data
        self.unwrapped_phase = ds.data_vars['pna_tr1_unwrapped_phase'].data
        self.dB_magnitude = ds.data_vars['pna_tr1_magnitude'].data

        total_transmission = self.linear_magnitude * np.exp(1j * self.phase * np.pi/180)
        self.I = np.real(total_transmission)
        self.Q = np.imag(total_transmission)

    def Load_Multiple_Data(self, run_ids: List[int]):


    def prepare_runfile_for_qumin(self, setup_file_path):
        # Prepare the setup file for QCoDeS
        with open(setup_file_path, 'w') as f:
            f.write("# Setup file for QCoDeS\n")
            f.write(f"experiment_name = '{self.exp_name}'\n")
            f.write(f"database_path = '{self.db_path}'\n")


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





def load_cavity_data(file_path):
    """
    Load S21 measurement data from a file.
    Expected format: frequency (Hz), magnitude (dB), phase (degrees)
    """
    try:
        data = pd.read_csv(file_path, names=['frequency', 'magnitude', 'phase'],
                           header=None if file_path.endswith('.dat') else 0)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        # If CSV fails, try other formats
        try:
            data = np.loadtxt(file_path)
            if data.shape[1] >= 3:
                return pd.DataFrame({
                    'frequency': data[:, 0],
                    'magnitude': data[:, 1],
                    'phase': data[:, 2]
                })
            else:
                print("File format not recognized")
                return None
        except Exception as e:
            print(f"Error loading data as array: {e}")
            return None


def analyze_cavity(data, resonator_type='notch', initial_guess=None):
    """
    Analyze cavity data and extract quality factor.

    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame with columns: frequency, magnitude, phase
    resonator_type : str
        Type of resonator ('notch' or 'reflection')
    initial_guess : dict
        Initial parameter guesses for the fit

    Returns:
    --------
    res : Resonator object
        Fitted resonator object with extracted parameters
    """
    # Convert magnitude from dB to linear if needed
    if np.mean(data['magnitude']) < 0:  # Likely in dB
        print("Converting magnitude from dB to linear")
        data['magnitude_linear'] = 10 ** (data['magnitude'] / 20)
    else:
        data['magnitude_linear'] = data['magnitude']

    # Convert phase from degrees to radians if needed
    if np.max(np.abs(data['phase'])) > 2 * np.pi:
        print("Converting phase from degrees to radians")
        data['phase_rad'] = data['phase'] * np.pi / 180
    else:
        data['phase_rad'] = data['phase']

    # Calculate complex S21
    data['S21_real'] = data['magnitude_linear'] * np.cos(data['phase_rad'])
    data['S21_imag'] = data['magnitude_linear'] * np.sin(data['phase_rad'])

    # Create a Resonator object with the data
    freqs = data['frequency'].values
    S21 = data['S21_real'].values + 1j * data['S21_imag'].values

    # Default initial guesses if none provided
    if initial_guess is None:
        # Estimate resonant frequency from minimum of magnitude
        f0_idx = np.argmin(data['magnitude_linear'].values)
        f0_guess = freqs[f0_idx]

        # Other default parameters
        initial_guess = {
            'f_0': f0_guess,
            'Q': 10000,  # Reasonable starting Q for superconducting cavities
            'Q_e': 20000,
            'phi_0': 0,
            'a': 1.0,
            'tau': 0
        }

    # Create and fit the resonator
    print(
        f"Creating {resonator_type} resonator with initial f_0 = {initial_guess['f_0'] / 1e9:.6f} GHz, Q = {initial_guess['Q']}")

    res = Resonator(freqs, S21, resonator_type=resonator_type)

    try:
        res.fit(**initial_guess)
        return res
    except Exception as e:
        print(f"Fit failed with error: {e}")
        print("Trying with different initial parameters...")

        # If first fit fails, try with more conservative guesses
        f0_idx = np.argmin(data['magnitude_linear'].values)
        f0_guess = freqs[f0_idx]

        conservative_guess = {
            'f_0': f0_guess,
            'Q': 5000,
            'Q_e': 10000,
            'phi_0': 0,
            'a': 0.5,
            'tau': 0
        }

        try:
            res = Resonator(freqs, S21, resonator_type=resonator_type)
            res.fit(**conservative_guess)
            return res
        except Exception as e:
            print(f"Second fit attempt failed: {e}")
            return None


def plot_resonator_results(res, data, output_dir='./results'):
    """
    Plot the resonator fitting results and save to file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot the magnitude and phase fits
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Get the original and fitted data
    freqs = res.freq
    S21_data = res.S21
    S21_fit = res.S21_fit

    # Magnitude plot (in dB)
    ax1.plot(freqs / 1e9, 20 * np.log10(np.abs(S21_data)), 'o', ms=4, alpha=0.6, label='Data')
    ax1.plot(freqs / 1e9, 20 * np.log10(np.abs(S21_fit)), '-', lw=2, label='Fit')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('|S21| (dB)')
    ax1.legend()
    ax1.grid(True)

    # Phase plot (in degrees)
    ax2.plot(freqs / 1e9, np.angle(S21_data) * 180 / np.pi, 'o', ms=4, alpha=0.6, label='Data')
    ax2.plot(freqs / 1e9, np.angle(S21_fit) * 180 / np.pi, '-', lw=2, label='Fit')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.legend()
    ax2.grid(True)

    # Add results to the plot
    params = res.fit_params

    # Create a text box with results
    textstr = '\n'.join((
        f"$f_0$ = {params['f_0'] / 1e9:.6f} GHz",
        f"$Q_i$ = {res.Q_i:.0f}",
        f"$Q_e$ = {params['Q_e']:.0f}",
        f"$Q_t$ = {params['Q']:.0f}",
        f"$\\kappa$ = {res.coupling_quality():.2f}",
        f"$\\phi_0$ = {params['phi_0']:.2f}"
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.05, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resonator_fit.png'), dpi=300)
    plt.close()

    # Plot the complex S21 data on the IQ plane
    plt.figure(figsize=(8, 8))
    plt.plot(S21_data.real, S21_data.imag, 'o', ms=4, alpha=0.6, label='Data')
    plt.plot(S21_fit.real, S21_fit.imag, '-', lw=2, label='Fit')
    plt.xlabel('Re(S21)')
    plt.ylabel('Im(S21)')
    plt.title('Complex S21 (IQ plane)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'resonator_iq.png'), dpi=300)
    plt.close()

    return


def save_results(res, output_dir='./results'):
    """
    Save the resonator parameters to a file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    params = res.fit_params

    # Create a dictionary with all important parameters
    results = {
        'f_0 (Hz)': params['f_0'],
        'f_0 (GHz)': params['f_0'] / 1e9,
        'Q_total': params['Q'],
        'Q_internal': res.Q_i,
        'Q_external': params['Q_e'],
        'phi_0 (rad)': params['phi_0'],
        'a': params['a'],
        'tau (s)': params['tau'],
        'kappa (coupling)': res.coupling_quality()
    }

    # Save as CSV
    df = pd.DataFrame(results, index=[0])
    df.to_csv(os.path.join(output_dir, 'resonator_parameters.csv'), index=False)

    # Also save as a text file for easy reading
    with open(os.path.join(output_dir, 'resonator_parameters.txt'), 'w') as f:
        f.write("Cavity Resonator Analysis Results\n")
        f.write("================================\n\n")

        for key, value in results.items():
            if isinstance(value, float):
                if abs(value) < 0.01 or abs(value) > 1000:
                    f.write(f"{key}: {value:.6e}\n")
                else:
                    f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"Results saved to {output_dir}")
    return results


def main():
    """
    Main function to run the analysis
    """
    print("Cavity Quality Factor Analysis using SCRAPS")
    print("===========================================")

    # Get input file
    file_path = input("Enter path to S21 data file: ")

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Load the data
    print(f"Loading data from {file_path}")
    data = load_cavity_data(file_path)

    if data is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Loaded {len(data)} data points")
    print(f"Frequency range: {data['frequency'].min() / 1e9:.6f} - {data['frequency'].max() / 1e9:.6f} GHz")

    # Ask for resonator type
    resonator_type = input("Enter resonator type (notch/reflection) [default: notch]: ").strip().lower()
    if resonator_type not in ['notch', 'reflection']:
        resonator_type = 'notch'
        print(f"Using default resonator type: {resonator_type}")

    # Get initial parameter guesses (optional)
    print("\nEnter initial parameter guesses (or press Enter to use automatic estimation):")
    initial_guess = {}

    f0_input = input("  Resonant frequency f_0 (GHz) [auto]: ").strip()
    if f0_input:
        initial_guess['f_0'] = float(f0_input) * 1e9  # Convert to Hz

    Q_input = input("  Total quality factor Q [10000]: ").strip()
    if Q_input:
        initial_guess['Q'] = float(Q_input)
    else:
        initial_guess['Q'] = 10000

    Qe_input = input("  External quality factor Q_e [20000]: ").strip()
    if Qe_input:
        initial_guess['Q_e'] = float(Qe_input)
    else:
        initial_guess['Q_e'] = 20000

    # If no initial f_0 was provided, estimate it
    if 'f_0' not in initial_guess:
        initial_guess = None

    # Analyze the data
    print("\nAnalyzing cavity data...")
    resonator = analyze_cavity(data, resonator_type, initial_guess)

    if resonator is None:
        print("Analysis failed. Exiting.")
        return

    # Display the results
    params = resonator.fit_params
    print("\nResults:")
    print(f"  Resonant frequency (f_0): {params['f_0'] / 1e9:.6f} GHz")
    print(f"  Internal quality factor (Q_i): {resonator.Q_i:.0f}")
    print(f"  External quality factor (Q_e): {params['Q_e']:.0f}")
    print(f"  Total quality factor (Q_t): {params['Q']:.0f}")
    print(f"  Coupling coefficient (κ): {resonator.coupling_quality():.2f}")
    print(f"  Phase offset (phi_0): {params['phi_0']:.2f} rad")

    # Plot and save the results
    output_dir = input("\nEnter output directory for results [./results]: ").strip()
    if not output_dir:
        output_dir = './results'

    plot_resonator_results(resonator, data, output_dir)
    save_results(resonator, output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
# Example usage and demonstration
def generate_example_cavity_data(
        num_points: int = 1000,
        num_cavities: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic cavity spectrum data

    Args:
        num_points (int): Number of data points
        num_cavities (int): Number of cavity resonances to simulate

    Returns:
        Tuple of frequency and amplitude arrays
    """
    # Create frequency range
    frequencies = np.linspace(0, 1000, num_points)

    # Initialize amplitude array
    amplitudes = np.zeros_like(frequencies)

    # Add cavity resonances
    for i in range(num_cavities):
        # Fundamental frequency and harmonics
        center_freq = 100 * (i + 1)
        amplitude = 1.0 / (i + 1)
        width = 5.0

        # Create Lorentzian peak
        peak = amplitude * width ** 2 / ((frequencies - center_freq) ** 2 + width ** 2)
        amplitudes += peak

    # Add some noise
    amplitudes += np.random.normal(0, 0.1, num_points)

    return frequencies, amplitudes



