"""
BBPLL linear model: frequency-domain modeling of bang-bang phase-locked loops.

Build a loop from components, shape each noise source through its closed-loop
transfer function, and integrate the result into RMS jitter::

    import numpy as np
    from bbpll import BangBangPhaseDetector, DigitalLoopFilter, DVCO, Divider
    from bbpll import PhaseNoise, get_shaped_phase_noise, get_jitter, get_pm_bw

    f = np.logspace(3, 9.6, 100000)
    LG = bbpd.tf(f) * dlf.tf(f) * dvco_digital * div.tf(f)
    shaped = get_shaped_phase_noise(f, vco_pn, 1 / (1 + LG))
    _, jitter_fs = get_jitter(f, shaped, oscillation_frequency=8e9,
                              integration_bandwidth=(1e3, 4e9))

Submodules:
    components  loop building blocks, each exposing ``tf(f)``
    noise       phase noise profiles and idealized blocks
    analysis    noise shaping, jitter integration, stability, plotting
    timedomain  zero-crossing times to jitter sequence and PSD
"""

from . import analysis, components, noise, timedomain

from .components import (
    BangBangPhaseDetector,
    ChargePumpFilter,
    DigitalLoopFilter,
    Divider,
    DVCO,
    PhaseFrequencyDetector,
)
from .noise import CDR, LoopFilter, PhaseDetector, PhaseNoise, VCO
from .analysis import (
    get_jitter,
    get_pm_bw,
    get_shaped_phase_noise,
    jitter_breakdown,
    jitter_breakdown_piechart,
    log_bin_data,
    metric_prefix_formatter,
    plot_jitter_heatmap_2D,
    plot_phase_noise,
    plot_tf,
    process_csv,
)
from .timedomain import calc_phase_noise_psd, get_jitter_sequence

__version__ = "0.1.0"

__all__ = [
    # submodules
    "analysis", "components", "noise", "timedomain",
    # components
    "BangBangPhaseDetector", "ChargePumpFilter", "DigitalLoopFilter",
    "Divider", "DVCO", "PhaseFrequencyDetector",
    # noise and ideal blocks
    "CDR", "LoopFilter", "PhaseDetector", "PhaseNoise", "VCO",
    # analysis
    "get_jitter", "get_pm_bw", "get_shaped_phase_noise", "jitter_breakdown",
    "jitter_breakdown_piechart", "log_bin_data", "metric_prefix_formatter",
    "plot_jitter_heatmap_2D", "plot_phase_noise", "plot_tf", "process_csv",
    # time domain
    "calc_phase_noise_psd", "get_jitter_sequence",
]
