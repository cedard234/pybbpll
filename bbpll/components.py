"""
PLL loop components.

Each component exposes ``tf(f)`` returning its transfer function evaluated at the
offset frequencies ``f`` (Hz), so a loop gain is formed by multiplying them together.

Author: Di Wang
"""

import numpy as np

from . import noise

class PhaseFrequencyDetector:
    """
    A class representing a phase frequency detector in a PLL system.
    This class models a simple phase frequency detector with a gain Kpd.
    """

    def __init__(self, kpd, input_referred_jitter):
        """
        Initializes the PhaseFrequencyDetector with a gain Kpd.
        Parameters:
        ----------
        - kpd: Gain of the phase frequency detector
        """
        self.kpd = kpd
        self.input_reffered_jitter = input_referred_jitter

    def tf(self, f: np.array):
        """
        Returns the transfer function of the phase frequency detector.
        The transfer function is represented as Kpd.
        :return: Transfer function of the phase frequency detector
        """
        return self.kpd

class BangBangPhaseDetector:
    """
    A class representing a bang-bang phase detector in a PLL system.
    This class models a simple bang-bang phase detector with a gain Kpd.
    """

    def __init__(self, rms_jitter_radians):
        """
        Initializes the BangBangPhaseDetector with a gain Kpd based on the RMS jitter.
        Parameters:
        ----------
        - kpd: Gain of the bang-bang phase detector
        """
        self.kpd = self.__calculate_kpd__(rms_jitter_radians)
    
    def __calculate_kpd__(self, rms_jitter_radians):
        """
        Calculates the gain Kpd based on the RMS jitter in radians, assuming jitter is Gausian distributed.
        WARNING: beware that the Kpd is a small-signal approximation around when the phase error is 0.
        Parameters:
        ----------
        - rms_jitter: RMS jitter in radians
        Returns:
        - Gain Kpd
        """
        return 2 / np.sqrt(2 * np.pi * rms_jitter_radians**2)

    def tf(self, f: np.array):
        """
        Returns the transfer function of the bang-bang phase detector.
        The transfer function is represented as Kpd.
        :return: Transfer function of the bang-bang phase detector
        """
        return self.kpd

class ChargePumpFilter:
    """
    A class representing a charge pump filter.
    The charge pump filter, usd as the proportional control path in a hybrid PLL,
    models the charge pump's output current divided by 2π times R, which is used to control the VCO.
    """

    def __init__(self, Icp, R, input_referred_noise, C_ripple: float = 0.0):
        """
        Initializes the ChargePump with a gain kp based on the charge pump current Icp and resistor R,
        also sets the input referred noise of the charge pump.
        Parameters:
        ----------
        - Icp: Charge pump current in Amperes
        - R: Resistance in Ohms
        - C_ripple: Optional ripple capacitance in Farads (default 0.0)
        - input_referred_noise: Input referred noise of the charge pump in Amperes
        """
        self.Icp = Icp
        self.R = R
        self.C_ripple = C_ripple
        self.input_referred_noise = input_referred_noise

    def tf(self, f):
        """
        Returns the transfer function of the charge pump.
        The transfer function is represented as Kcp.
        :return: Transfer function of the charge pump
        """
        if self.C_ripple == 0:
            # If no ripple capacitance is provided, just return a constant gain
            loop_filter_impedance = self.R
        else:
            # otherwise this will be a frequency-dependent impedance
            s = 2 * np.pi * f * 1j  # Laplace variable in continuous-time
            loop_filter_impedance = self.R / (1 + s * self.R * self.C_ripple)

        return self.Icp * loop_filter_impedance
            

class DigitalLoopFilter:
    """
    A class representing a digital loop filter in a PLL system.
    This class models a pure second-order digital loop filter with an integral gain of Ki.
    """

    def __init__(self, ki, sampling_time, latency = 0.0, q_noise = 1/12, kp = 0, approximation_method='direct'):
        """
        Initializes the DigitalLoopFilter with integral gain Ki and sampling time.
        Parameters:
        ----------
        - ki: Integral gain of the digital loop filter
        - sampling_time: Sampling time for the digital system
        - latency: Latency introduced by the digital loop filter in seconds
        - q_noise: Quantization noise, default is 1/12 for uniform quantization
        - kp: Proportional gain of the digital loop filter, default is 0 so the dlf is just an accumulator
        - approximation_method: Method for discretizing the filter ('direct', 'tustin', 'ideal').

        """
        self.kp = kp
        self.ki = ki
        self.sampling_time = sampling_time
        self.latency = latency # Latency in radians, default is 0
        if approximation_method not in ['direct', 'tustin', 'ideal', 'ZoH']:
            raise ValueError("Invalid approximation method. Choose from 'direct', 'tustin', 'ideal', or 'ZoH'.")
        self.approximation_method = approximation_method
        self.q_noise = q_noise  # Quantization noise, default is 1/12 for uniform quantization
    
    def tf(self, f):
        """
        Returns the transfer function of the digital loop filter.
        The transfer function is represented as Ki / (1 - z^-1).
        :return: Transfer function of the digital loop filter
        """
        s = 2 * np.pi * f * 1j
        z = 0
        tf_latency = np.exp(-self.latency * s)
        if self.approximation_method == 'direct':
            z = np.exp(self.sampling_time * s)
        elif self.approximation_method == 'tustin':
            z = (1 + 0.5 * s * self.sampling_time) / (1 - 0.5 * s * self.sampling_time)
        elif self.approximation_method == 'ideal':
            return (self.ki / (s * self.sampling_time) + self.kp) * tf_latency
        elif self.approximation_method == 'ZoH':
            z = np.exp(self.sampling_time * s)
            # tf_ZoH = np.sin(np.pi * f * self.sampling_time) / (np.pi * f * self.sampling_time) * np.exp(-1j * np.pi * f * self.sampling_time)
            tf_ZoH = (1 - np.exp(-s * self.sampling_time)) / (s * self.sampling_time)
            # tf_ZoH = 1
            tf_digital_delay = np.exp(-s * 0.5 * self.sampling_time)
            return (self.ki * tf_digital_delay / (1 - z ** (-1)) + self.kp) * tf_latency * tf_ZoH 
            # return (self.ki * tf_digital_delay / (1 - z ** (-1))) * tf_latency * tf_ZoH + self.kp
            # return (self.ki * z ** (-1) / (1 - z ** (-1)) + self.kp) * tf_latency * tf_ZoH
            # return tf_ZoH
        
        return  (self.ki * z ** (-1) / (1 - z ** (-1)) + self.kp)* tf_latency

class Divider:
    """
    A class representing a frequency divider in a PLL system.
    """

    def __init__(self, divider_ratio, latency):
        """
        Initializes the Divider with a specified divider ratio and latency.
        Parameters:
        ----------
        - divider_ratio: Ratio by which the frequency is divided
        - latency: Latency introduced by the divider in seconds
        """
        self.divider_ratio = divider_ratio
        self.latency = latency  # Latency in radians
    
    def tf(self, f: np.array):
        """
        Returns the transfer function of the divider.
        The transfer function is represented as exp(-latency * s) / (divider_ratio).
        :return: Transfer function of the divider
        """
        s = 2 * np.pi * f * 1j  # Laplace variable in continuous-time
        return np.exp(-self.latency * s) / self.divider_ratio

class DVCO:
    """
    A class representing a digital voltage-controlled oscillator (DVCO) in a PLL system.
    This class models a DVCO with a gain kvco, kdco and a phase noise model.
    """

    def __init__(self, kvco, kdco, phase_noise):
        """
        Initializes the DVCO with a gain kvco and kdco.
        Parameters:
        ----------
        - kvco: Gain of the DVCO
        - kdco: Gain of the DCO
        - phase_noise: Phase noise model of the DVCO
        """
        self.kvco = kvco
        self.kdco = kdco
        self.phase_noise: noise.PhaseNoise = phase_noise

    def tf(self, f: np.array):
        """
        The transfer function of the DVCO is a 2 to 1 transfer function.
        Returns the transfer function as two separate components:
        1. kvco / s: Represents the analog control path of the DVCO.
        2. kdco / s: Represents the digital control path of the DVCO.
        :return: Transfer function of the DVCO
        """
        s = 2 * np.pi * f * 1j
        return self.kvco / s, self.kdco / s

