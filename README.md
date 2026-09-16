# BBPLL Linear Model

Frequency-domain modeling of bang-bang phase-locked loops: build a loop from components, shape
each noise source through its closed-loop transfer function, integrate into RMS jitter.

Worked example:
[`examples/BBPLL_100MHz_ref_8GHz_out.ipynb`](examples/BBPLL_100MHz_ref_8GHz_out.ipynb) — a
100 MHz-reference, 8 GHz-output BBPLL. Needs no external data files.

## Install

```bash
pip install -e .
jupyter lab examples/BBPLL_100MHz_ref_8GHz_out.ipynb
```

Installs as `pybbpll`; imports as `bbpll`.

## Layout

| Module | Contents |
|---|---|
| [`bbpll/components.py`](bbpll/components.py) | `BangBangPhaseDetector`, `PhaseFrequencyDetector`, `DigitalLoopFilter`, `ChargePumpFilter`, `Divider`, `DVCO` — each exposes `tf(f)` |
| [`bbpll/noise.py`](bbpll/noise.py) | `PhaseNoise` profile synthesis, plus ideal `VCO`, `LoopFilter`, `Divider`, `CDR` |
| [`bbpll/analysis.py`](bbpll/analysis.py) | `get_shaped_phase_noise`, `get_jitter`, `get_pm_bw`, `jitter_breakdown`, plotting |
| [`bbpll/timedomain.py`](bbpll/timedomain.py) | Zero-crossing times → jitter sequence → phase-noise PSD |
| [`examples/`](examples/) | Worked notebooks |

## Usage

```python
import numpy as np
from bbpll import (BangBangPhaseDetector, DigitalLoopFilter, Divider, DVCO,
                   PhaseNoise, get_shaped_phase_noise, get_jitter, get_pm_bw)

f = np.logspace(3, 9.6, 100000)
kvco, kdco = 100e6 * 2*np.pi, 1.25e6 * 2*np.pi
vco_pn = PhaseNoise(white_noise_100MHz=-130, flicker_noise_10kHz=-15).get_noise_power(f)

# sigma is the jitter at the detector input, and must be consistent with the gains below
sigma = 1013.2e-15 * 2*np.pi*100e6

bbpd = BangBangPhaseDetector(rms_jitter_radians=sigma)
dlf  = DigitalLoopFilter(kp=0.507793, ki=0.0667074, sampling_time=10e-9,
                         latency=0, approximation_method='ZoH')
_, dvco_d = DVCO(kvco, kdco, vco_pn).tf(f)
div = Divider(divider_ratio=80, latency=0)

LG = bbpd.tf(f) * dlf.tf(f) * dvco_d * div.tf(f)

vco_out = get_shaped_phase_noise(f, vco_pn, 1 / (1 + LG))       # VCO sees 1/(1+LG)
_, jitter_fs = get_jitter(f, vco_out, oscillation_frequency=8e9,
                          integration_bandwidth=(1e3, 4e9))
phase_margin, bandwidth = get_pm_bw(f, LG)      # -> 60.0 deg, 10.00 MHz
```

## BBPD gain

A BBPD is a 1-bit quantizer with no intrinsic gain. Linearized for Gaussian input jitter, its
small-signal gain about zero phase error is

$$K_{bbpd} = \frac{2}{\sqrt{2\pi\sigma^2}}$$

valid **only near zero phase error**. Since $\sigma$ is the jitter at the detector input — which
the loop itself shapes — $K_{bbpd}$ and the output jitter are mutually dependent. The notebook
resolves this with a damped fixed-point iteration.

## Two gotchas

**Sampled loops have a phase-margin ceiling.** The zero-order-hold half-sample delay costs phase
in proportion to `BW × T_sample` — at 10 ns sampling that is −18° at 10 MHz, capping phase margin
at 72° with no transport latency. Asking for more is unattainable, not a tuning problem.

**`fsolve` fails silently past that ceiling**, returning gains around `1e8` and nonsense jitter
while `get_pm_bw` reports a plausible-looking phase margin. Check the feasibility bound, the
convergence flag, and the residual, then confirm with `get_pm_bw`.

## Example results

100 MHz reference → 8 GHz output (N = 80), 10 MHz bandwidth, 60° phase margin — **1013 fs** total:
VCO 657 fs (42 % of jitter power), reference 541 fs (29 %), BBPD quantization 492 fs (24 %),
DAC quantization 243 fs (6 %).

## Scope

Linear and frequency-domain: small-signal noise shaping, loop stability, steady-state jitter of a
locked loop. Not acquisition, cycle slipping, or BBPD limit cycles.

## License

MIT
