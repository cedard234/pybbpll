"""
Time-domain jitter utilities.

Convert simulator zero-crossing times into a jitter sequence and a phase-noise PSD,
for comparing transient simulations against the frequency-domain model.

Author: Di Wang
"""


import numpy as np
import scipy.signal as signal

def get_jitter_sequence(crossing_time, hold_off_time: float = 0.0, oscillation_frequency: float = 0):
    """
    calculate the jitter sequence from crossing times.
    Parameters:
    ----------
    crossing_time : list of float
        List of crossing times in seconds.
    hold_off_time : float, optional
        Hold-off time in seconds to start the jitter calculation. Used to avoid locking behavior at the beginning of the sequence.
    oscillation_frequency : float, optional
        Jitterless fundamental frequency in Hz. If not provided, it will be calculated based on the crossing times.
    Returns:
    -------
    jitter_sequence : list of float
        List of jitter values in seconds, calculated as the difference between
        each crossing time and the expected time based on a linear fit.
    """
    jitter_sequence = []
    # crossing_time_holdoff = [x for x in crossing_time if x >= hold_off_time]
    crossing_time_holdoff = []
    for item in crossing_time:
        if item >= hold_off_time:
            crossing_time_holdoff.append(item)
    if oscillation_frequency == 0:
        true_period = (crossing_time_holdoff[-1] - crossing_time_holdoff[0]) / (len(crossing_time_holdoff) - 1)
    else:
        true_period = 1 / oscillation_frequency
    for i in range(0, len(crossing_time_holdoff)):
        jitter = crossing_time_holdoff[i] - (i * true_period)
        jitter_sequence.append(jitter)
    jitter_sequence = jitter_sequence - np.mean(jitter_sequence)  # Remove DC offset
    return jitter_sequence

def calc_phase_noise_psd(jitter_sequence, oscillation_frequency: float = 1.625e9):
    """
    Calculate phase noise from jitter sequence, and return as PN/freq doublet.
    Parameters:
    ----------
    jitter_sequence : list of float
        List of jitter values in seconds.
    oscillation_frequency : float, optional
        Jitterless fundamental frequency in Hz. 
    Returns:
    -------
    frequencies : ndarray
        Frequencies at which the phase noise is calculated.
    phase_noise : ndarray
        Phase noise in dBc/Hz, calculated from the power spectral density of the phase sequence.
    """
    # frequencies, psd = signal.welch(jitter_sequence, fs=fundamental_frequency, nperseg=256)
    # Convert the jitter sequence to phase sequence
    phase_sequence = 2 * np.pi * np.array(jitter_sequence) * oscillation_frequency
    # frequencies, psd = signal.periodogram(phase_sequence, fs=oscillation_frequency)
    frequencies, psd = signal.periodogram(phase_sequence, fs=oscillation_frequency, window='hann')
    phase_noise = 10 * np.log10(psd)  # Convert to dBc/Hz
    return frequencies, phase_noise

