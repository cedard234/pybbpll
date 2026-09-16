"""
Phase noise profiles and idealized PLL blocks.

``PhaseNoise`` synthesizes a dBc/Hz profile from white, flicker, and floor terms and
converts between dBc/Hz and linear power. The remaining classes are ideal (continuous-time)
counterparts to the sampled components in :mod:`bbpll.components`.

Author: Di Wang
"""

from typing import Optional
import numpy as np

class PhaseDetector:
    """
    A class representing a phase detector in a PLL system.
    This class models a simple phase detector with a gain Kpd.
    """

    def __init__(self, kpd):
        """
        Initializes the PhaseDetector with a gain Kpd.
        Parameters:
        ----------
        - kpd: Gain of the phase detector
        """
        self.kpd = kpd

    def tf(self, f: np.array):
        """
        Returns the transfer function of the phase detector.
        The transfer function is represented as Kpd.
        :return: Transfer function of the phase detector
        """
        return self.kpd


class LoopFilter:
    """
    A class representing a loop filter in a PLL system.
    This class models a simple second-order loop filter with proportional and integral gains Kp and Ki.
    """

    def __init__(self, Icp: float, R: float, C: float, C_ripple: float = 0.0):
        """
        Initializes the LoopFilter with Icp, R, C, and an optional C_Ripple.
        
        Parameters:
        ----------
        - Icp: Charge pump current in Amperes
        - R: Resistance in Ohms
        - C: Capacitance in Farads
        - C_Ripple: Optional ripple capacitance in Farads (default 0.0)
        
        """
        self.Icp = Icp
        self.R = R
        self.C = C
        self.C_ripple = C_ripple

    def tf(self, f: np.array):
        """
        Returns the transfer function of the loop filter.
        H(s) = Icp * (R + 1/(s*C)).
        If C_Ripple is provided, it adds an additional term to the transfer function.
        """
        s = 2 * np.pi * f * 1j  # Laplace variable in continuous-time
        impedance_without_C_ripple = (self.R + 1 / (s * self.C))
        if self.C_ripple == 0.0:
            return self.Icp * impedance_without_C_ripple  # Continuous-time transfer function
        else:
            impedance_C_ripple = 1 / (s * self.C_ripple)  # Impedance of the ripple capacitor
            impedance_with_C_ripple = (impedance_without_C_ripple * impedance_C_ripple) / (impedance_without_C_ripple + impedance_C_ripple)
            return self.Icp * impedance_with_C_ripple  # Continuous-time transfer function with ripple capacitor
        

class PhaseNoise:
    """
    A class representing general phase noise.
    """

    def __init__(self, white_noise_100MHz = -500, flicker_noise_10kHz: Optional[float] = None, noise_floor: Optional[float] = None, flicker_noise_distribution_10kHz: Optional[float] = None):
        """
        Initializes the PhaseNoise with white noise at 100 MHz, flicker noise at 10 kHz,
        and an optional noise floor.
        
        Parameters:
        ----------
        - white_noise_100MHz: White noise level at 100 MHz in dBc/Hz
        - flicker_noise_10kHz: Flicker noise level at 10 kHz (in dBc/Hz, default None)
        - noise_floor: Optional noise floor level in dBc/Hz (default None, set to 0.0 if not provided)
        - flicker_noise_distribution_10kHz: Optional flicker noise due to distribution at 10 kHz (in dBc/Hz, default None)
        """
        self.white_noise_100MHz = white_noise_100MHz
        self.flicker_noise_10kHz = flicker_noise_10kHz if flicker_noise_10kHz is not None else None
        self.noise_floor = noise_floor if noise_floor is not None else None
        self.flicker_noise_distribution_10kHz = flicker_noise_distribution_10kHz if flicker_noise_distribution_10kHz is not None else None


    def to_dbc(self, value):
        """
        Converts a value to dBc/Hz format.
        
        Parameters:
        ----------
        - value: Value to convert
        
        Returns:
        -------
        - Converted value in dBc/Hz
        """
        return 10 * np.log10(value)


    def to_linear(self, value):
        """
        Converts a dBc/Hz value back to its linear scale.
        
        Parameters:
        ----------
        - value: Value in dBc/Hz
        
        Returns:
        -------
        - Converted value in linear scale
        """
        return 10 ** (value / 10.0)


    def get_noise_power(self, frequency):
        """
        Returns the single-sided phase noise power at a given frequency.
        
        Parameters:
        ----------
        - frequency: Frequency in Hz at which to compute the phase noise
        
        Returns:
        -------
        - Phase noise level in dBc/Hz at the specified frequency
        """

        white_noise_linear_100MHz = self.to_linear(self.white_noise_100MHz)

        total_noise = white_noise_linear_100MHz / (frequency / 100e6)**2
        if self.flicker_noise_10kHz is not None:
            flicker_noise_linear_10kHz = self.to_linear(self.flicker_noise_10kHz)
            total_noise += flicker_noise_linear_10kHz / (frequency / 10e3)**3
        if self.noise_floor is not None:
            noise_floor_linear = self.to_linear(self.noise_floor)
            total_noise += noise_floor_linear
        if self.flicker_noise_distribution_10kHz is not None:
            flicker_noise_distribution_linear_10kHz = self.to_linear(self.flicker_noise_distribution_10kHz)
            total_noise += flicker_noise_distribution_linear_10kHz / (frequency / 10e3)

        return self.to_dbc(total_noise)
    
    def sum_phase_noise(self, phase_noise_list):
        """
        Sums the phase noise contributions from multiple sources.

        Parameters:
        ----------
        - phase_noise_list: List of PhaseNoise objects to sum in dBc/Hz

        Returns:
        -------
        - Total phase noise in dBc/Hz
        """

        # Check that all entries have the same length
        lengths = [len(item) for item in phase_noise_list]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All phase noise entries must have the same length.")

        total_noise_linear = np.zeros(lengths[0])
        for item in phase_noise_list:
            total_noise_linear += self.to_linear(item)
        
        for item in total_noise_linear:
            if item <= 0:
                raise ValueError("Phase noise contributions must be positive to be summed.")

        return self.to_dbc(total_noise_linear)

class VCO:
    """
    A class representing a voltage-controlled oscillator (VCO) in a PLL system.
    This class models a simple VCO with a gain Kvco and an optional offset frequency.
    """

    def __init__(self, 
                 kvco, 
                 oscillation_frequency: float = 26e9,
                 white_noise_100MHz: float = -135, 
                 flicker_noise_10kHz: float = -15, 
                 noise_floor: float = -500):
        """
        Constructs a VCO with specified parameters.
        Parameters:
        ----------
        - kvco: Gain of the VCO in Hz/V
        - white_noise_100MHz: White noise level at 100 MHz in dBc/Hz (default -135 dBc/Hz)
        - flicker_noise_10kHz: Flicker noise level at 10 kHz in dBc/Hz (default -15 dBc/Hz)
        - noise_floor: Optional noise floor level in dBc/Hz (default 0 dBc/Hz)
        """
        self.kvco = kvco
        self.oscillation_frequency = oscillation_frequency
        self.phase_noise = PhaseNoise(
            white_noise_100MHz=white_noise_100MHz,
            flicker_noise_10kHz=flicker_noise_10kHz,
            noise_floor=noise_floor
        )

    def tf(self, f: np.array):
        """
        Returns the transfer function of the VCO.
        The transfer function is represented as Kvco * s + offset_frequency.
        
        :return: Transfer function of the VCO
        """
        s = 2 * np.pi * f * 1j  # Laplace variable in continuous-time
        return self.kvco / s

class Divider:
    """
    A class representing a frequency divider in a PLL system.
    This class models a simple frequency divider with a division factor N.
    """

    def __init__(self, n):
        """
        Initializes the Divider with a division factor N.
        Parameters:
        ----------
        - n: Division factor of the frequency divider
        """
        self.n = n

    def tf(self, f: np.array):
        """
        Returns the transfer function of the frequency divider.
        The transfer function is represented as 1/N.
        :return: Transfer function of the frequency divider
        """
        return 1 / self.n

class CDR:
    """
    A class representing a clock data recovery (CDR) system in a PLL.
    This class models a simple CDR with a gain Kcdr.
    """

    def __init__(self, order, bw):
        """
        Initializes the CDR with order, bandwidth.
        """
        if order != 2:
            # FIXME: CDR order only supports 2 for now
            raise ValueError("CDR order only supports 2 for now.")
        self.order = order
        self.bw = bw

    def tf(self, f: np.array):
        """
        Returns the transfer function of the CDR.
        Assume the CDR filter is a butterworth filter. TODO: not necessarily true
        :return: Transfer function of the CDR
        """
        w0 = 2*np.pi * 4e6
        w1 = 2*np.pi * 1.552e6
        dampf1 = 1 / np.sqrt(2)
        dampf2 = 1
        s = 2 * np.pi * f * 1j
        
        # tf_CDR =  s ** 2 / (s ** 2 + np.sqrt(2) * self.bw * s + self.bw ** 2)
        tf_CDR = (s ** 2) / ((s + w0) * (s + w1)) * ((s ** 2 + 2 * dampf2 * w0 * s + w0 ** 2) / (s ** 2 + 2 * dampf1 * w0 * s + w0 ** 2))
        return tf_CDR