#!/usr/bin/env python3
"""
Cavity Identification and Quality Factor Analysis Script

This script uses the SCRAPS (Superconducting Cavity Resonator Analysis Program Suite)
package to identify cavity resonances from VNA data and analyze quality factors.

Usage:
    python cavity_analysis.py data_file.csv

Requirements:
    - scraps
    - numpy
    - matplotlib
    - scipy
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scraps import Resonator
import scraps as scp


def load_data(filename):
    """
    Load VNA data from various file formats.

    Parameters:
        filename (str): Path to the data file

    Returns:
        tuple: (freq, s21) arrays
    """
    print(f"Loading data from {filename}")

    # Determine file type by extension
    ext = os.path.splitext(filename)[1].lower()

    if ext == '.csv' or ext == '.txt':
        # Assume CSV format with columns: frequency, real(s21), imag(s21)
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        freq = data[:, 0]  # First column is frequency
        # Check if data has complex values directly or separate real/imag columns
        if data.shape[1] == 2:
            s21 = data[:, 1]  # Assume already complex
        else:
            s21_real = data[:, 1]  # Real part
            s21_imag = data[:, 2]  # Imaginary part
            s21 = s21_real + 1j * s21_imag

    elif ext == '.npz':
        # Numpy compressed format
        data = np.load(filename)
        freq = data['freq']
        s21 = data['s21']

    elif ext == '.npy':
        # Assume structured data with freq in first column, s21 real and imag in next columns
        data = np.load(filename)
        freq = data[:, 0]
        s21_real = data[:, 1]
        s21_imag = data[:, 2]
        s21 = s21_real + 1j * s21_imag

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    print(f"Loaded {len(freq)} data points")
    return freq, s21


def identify_cavity_resonances(freq, s21, min_depth_db=3, min_distance_idx=20):
    """
    Identify cavity resonances in VNA data by finding dips in |S21|.

    Parameters:
        freq (array): Frequency data in Hz
        s21 (array): Complex S21 data
        min_depth_db (float): Minimum depth in dB to be considered a resonance
        min_distance_idx (int): Minimum separation between resonances in array indices

    Returns:
        list: Indices of identified resonances
    """
    # Convert S21 to magnitude in dB
    s21_db = 20 * np.log10(np.abs(s21))

    # Find peaks in the negative of s21_db (which are dips in the original)
    peak_indices, peak_props = find_peaks(-s21_db, height=min_depth_db, distance=min_distance_idx)

    # Sort by depth (deepest first)
    peak_depths = peak_props['peak_heights']
    sorted_indices = np.argsort(-peak_depths)
    peak_indices = peak_indices[sorted_indices]

    print(f"Found {len(peak_indices)} resonances")

    return peak_indices


def analyze_resonance(freq, s21, peak_idx, window_size=100):
    """
    Analyze a single resonance using SCRAPS.

    Parameters:
        freq (array): Frequency data in Hz
        s21 (array): Complex S21 data
        peak_idx (int): Index of the resonance in the data arrays
        window_size (int): Number of points to include on each side of the resonance

    Returns:
        Resonator: Fitted SCRAPS Resonator object
    """
    # Extract data in the vicinity of the resonance
    start_idx = max(0, peak_idx - window_size)
    end_idx = min(len(freq), peak_idx + window_size + 1)

    freq_window = freq[start_idx:end_idx]
    s21_window = s21[start_idx:end_idx]

    # Initial guess for resonator parameters
    fr_guess = freq[peak_idx]
    Q_guess = 10000
    Qc_guess = 15000

    # Create resonator object and fit
    resonator = Resonator(freq_window, s21_window, fr=fr_guess, Ql=Q_guess, Qc=Qc_guess)

    # Try multiple fit models if first fit fails
    models = ['DCM', 'CPZM', 'INV']
    success = False

    for model_name in models:
        try:
            print(f"Trying {model_name} model...")
            resonator.model_type = model_name
            resonator.fit()
            success = True
            print(f"Successfully fit with {model_name} model")
            break
        except Exception as e:
            print(f"Fit with {model_name} model failed: {e}")

    if not success:
        print("WARNING: All fit attempts failed")
        return None

    return resonator


def plot_resonance(resonator, freq, s21, peak_idx, window_size=100, output_dir="."):
    """
    Create plots for a resonance and its fit.

    Parameters:
        resonator (Resonator): Fitted SCRAPS Resonator object
        freq (array): Original frequency data in Hz
        s21 (array): Original complex S21 data
        peak_idx (int): Index of the resonance
        window_size (int): Number of points to include on each side
        output_dir (str): Directory to save plots
    """
    # Extract data in the vicinity of the resonance
    start_idx = max(0, peak_idx - window_size)
    end_idx = min(len(freq), peak_idx + window_size + 1)

    freq_window = freq[start_idx:end_idx]
    s21_window = s21[start_idx:end_idx]

    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Resonance at {resonator.fr / 1e9:.6f} GHz", fontsize=16)

    # Plot magnitude
    axs[0, 0].plot(freq_window / 1e9, 20 * np.log10(np.abs(s21_window)), 'o', ms=2, label='Data')
    axs[0, 0].plot(freq_window / 1e9, 20 * np.log10(np.abs(resonator.model)), '-', label='Fit')
    axs[0, 0].set_xlabel('Frequency (GHz)')
    axs[0, 0].set_ylabel('|S21| (dB)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot phase
    axs[0, 1].plot(freq_window / 1e9, np.angle(s21_window), 'o', ms=2, label='Data')
    axs[0, 1].plot(freq_window / 1e9, np.angle(resonator.model), '-', label='Fit')
    axs[0, 1].set_xlabel('Frequency (GHz)')
    axs[0, 1].set_ylabel('Phase (rad)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot real part
    axs[1, 0].plot(freq_window / 1e9, np.real(s21_window), 'o', ms=2, label='Data')
    axs[1, 0].plot(freq_window / 1e9, np.real(resonator.model), '-', label='Fit')
    axs[1, 0].set_xlabel('Frequency (GHz)')
    axs[1, 0].set_ylabel('Re(S21)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot imaginary part
    axs[1, 1].plot(freq_window / 1e9, np.imag(s21_window), 'o', ms=2, label='Data')
    axs[1, 1].plot(freq_window / 1e9, np.imag(resonator.model), '-', label='Fit')
    axs[1, 1].set_xlabel('Frequency (GHz)')
    axs[1, 1].set_ylabel('Im(S21)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"resonance_{peak_idx}_fit.png"))

    # Create complex plane plot (S21 circle)
    plt.figure(figsize=(8, 8))
    plt.plot(np.real(s21_window), np.imag(s21_window), 'o', ms=2, label='Data')
    plt.plot(np.real(resonator.model), np.imag(resonator.model), '-', label='Fit')

    # Find the point closest to resonance frequency
    res_idx = np.argmin(np.abs(freq_window - resonator.fr))
    plt.plot(np.real(s21_window[res_idx]), np.imag(s21_window[res_idx]), 'rx', ms=10, label='Resonance')

    plt.xlabel('Re(S21)')
    plt.ylabel('Im(S21)')
    plt.title(f'Complex Plane (S21 Circle) - Resonance at {resonator.fr / 1e9:.6f} GHz')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"resonance_{peak_idx}_circle.png"))

    plt.close('all')


def plot_overview(freq, s21, resonance_indices, output_dir="."):
    """
    Create an overview plot showing all identified resonances.

    Parameters:
        freq (array): Frequency data in Hz
        s21 (array): Complex S21 data
        resonance_indices (list): Indices of identified resonances
        output_dir (str): Directory to save plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(freq / 1e9, 20 * np.log10(np.abs(s21)), 'b-')

    # Mark each resonance
    for i, idx in enumerate(resonance_indices):
        plt.axvline(x=freq[idx] / 1e9, color='r', linestyle='--', alpha=0.5)
        plt.text(freq[idx] / 1e9, 20 * np.log10(np.abs(s21[idx])) - 2, f"{i + 1}",
                 ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.xlabel('Frequency (GHz)')
    plt.ylabel('|S21| (dB)')
    plt.title('Identified Cavity Resonances')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "resonance_overview.png"))
    plt.close()


def create_summary_file(results, freq, output_dir="."):
    """
    Create a summary file of all resonances and quality factors.

    Parameters:
        results (list): List of (peak_idx, resonator) tuples
        freq (array): Frequency data in Hz
        output_dir (str): Directory to save summary
    """
    with open(os.path.join(output_dir, "resonance_summary.txt"), 'w') as f:
        f.write("Cavity Resonance Analysis Summary\n")
        f.write("================================\n\n")

        for i, (peak_idx, resonator) in enumerate(results):
            if resonator is None:
                f.write(f"Resonance {i + 1} at {freq[peak_idx] / 1e9:.6f} GHz: FIT FAILED\n\n")
                continue

            f.write(f"Resonance {i + 1} at {freq[peak_idx] / 1e9:.6f} GHz:\n")
            f.write(f"  Fitted fr: {resonator.fr / 1e9:.9f} GHz\n")
            f.write(f"  Internal Quality Factor (Qi): {resonator.Qi:.0f}\n")
            f.write(f"  Coupling Quality Factor (Qc): {resonator.Qc:.0f}\n")
            f.write(f"  Loaded Quality Factor (Ql): {resonator.Ql:.0f}\n")
            f.write(f"  Coupling coefficient (κ): {resonator.kappa:.4f}\n")
            f.write(f"  Fit model: {resonator.model_type}\n\n")

    # Also create a CSV file
    with open(os.path.join(output_dir, "resonance_summary.csv"), 'w') as f:
        f.write("Index,Frequency (GHz),fr (GHz),Qi,Qc,Ql,kappa,Model\n")

        for i, (peak_idx, resonator) in enumerate(results):
            if resonator is None:
                f.write(f"{i + 1},{freq[peak_idx] / 1e9:.9f},,,,,\n")
                continue

            f.write(f"{i + 1},{freq[peak_idx] / 1e9:.9f},{resonator.fr / 1e9:.9f},")
            f.write(f"{resonator.Qi:.0f},{resonator.Qc:.0f},{resonator.Ql:.0f},")
            f.write(f"{resonator.kappa:.4f},{resonator.model_type}\n")


def main():
    """Main function to run the script."""
    # Check command line arguments

    data_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        freq, s21 = load_data(data_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Identify resonances
    resonance_indices = identify_cavity_resonances(freq, s21)

    # Create overview plot
    plot_overview(freq, s21, resonance_indices, output_dir)

    # Analyze each resonance
    results = []
    for i, peak_idx in enumerate(resonance_indices):
        print(f"\nAnalyzing resonance {i + 1} at {freq[peak_idx] / 1e9:.6f} GHz...")

        # Analyze the resonance
        resonator = analyze_resonance(freq, s21, peak_idx)
        results.append((peak_idx, resonator))

        # Generate plots if fit was successful
        if resonator is not None:
            # Print results
            print(f"Results:")
            print(f"  fr: {resonator.fr / 1e9:.9f} GHz")
            print(f"  Qi: {resonator.Qi:.0f}")
            print(f"  Qc: {resonator.Qc:.0f}")
            print(f"  Ql: {resonator.Ql:.0f}")
            print(f"  κ: {resonator.kappa:.4f}")

            # Create plots
            plot_resonance(resonator, freq, s21, peak_idx, output_dir=output_dir)

    # Create summary files
    create_summary_file(results, freq, output_dir)

    print("\nAnalysis complete!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()