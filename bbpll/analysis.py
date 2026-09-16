"""
Noise shaping, jitter integration, and plotting.

Shape an open-loop noise profile through a closed-loop transfer function with
``get_shaped_phase_noise``, integrate it to RMS jitter with ``get_jitter``, and read
loop stability off a loop gain with ``get_pm_bw``.

Author: Di Wang
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from .noise import PhaseNoise
import numpy as np
import pandas as pd

def metric_prefix_formatter(x, pos):
    """Convert number to string with metric prefix."""
    if x == 0:
        return '0'
    prefixes = ['', 'k', 'M', 'G', 'T']
    exp = min(max(0, int(np.log10(abs(x)) // 3)), len(prefixes)-1)
    value = x / (1000 ** exp)
    return f'{value:.0f}{prefixes[exp]}'

def plot_tf(f : np.array, 
            tf_values,
            title: str='Plot',
            grid: bool=True):
    """
    Plots the Bode plot of a transfer function.
    Parameters:
    ----------
    - f: Frequency array (in Hz or rad/s)
    - tf_values: Transfer function values (magnitude and phase)
    - xlabel: Label for the x-axis
    - title: Title of the plot
    - grid: Boolean indicating if grid should be shown
    
    Returns:
    -------
    - None, displays the plot
    """

    # Plot both magnitude and phase
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    mag = np.abs(tf_values)
    phase = np.angle(tf_values)
    phase = np.unwrap(phase)  # Unwrap phase to avoid discontinuities
    mag_dB = 20 * np.log10(np.abs(mag))

    phase_deg = phase * 180 / np.pi
    ax1.semilogx(f, mag_dB)
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title(title)
    ax1.grid(grid)
    ax2.semilogx(f, phase_deg, color='orange')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_xlabel('Offset Frequency (Hz)')
    ax2.grid(grid)
    
    # Add metric prefix formatter to x-axis
    formatter = FuncFormatter(metric_prefix_formatter)
    ax2.xaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    plt.show()
    return

def get_shaped_phase_noise(f: np.array,
                          phase_noise_profile: np.array,
                          tf_values):
    """
    Computes the shaped phase noise profile based on a transfer function.
    
    Parameters:
    ----------
    - f: Frequency array (in Hz)
    - phase_noise_profile: Phase noise values (in dBc/Hz)
    - tf_values: transfer function values (magnitude and phase)
    
    Returns:
    -------
    - Shaped phase noise profile in dBc/Hz
    """
    mag = np.abs(tf_values)
    shaped_phase_noise = phase_noise_profile + 20 * np.log10(np.abs(mag))
    
    return shaped_phase_noise

def plot_phase_noise(f: np.array,
                     phase_noise_profile_list: list,
                     title: str='Phase Noise Plot',
                     xlabel: str='Offset Frequency (Hz)',
                     ylabel: str='Phase Noise (dBc/Hz)', 
                     legend_list : list=None,
                     grid: bool=True,
                     ylim: tuple=(-180, -80),
                     xlim: tuple=(1e3, 10e9)):
    """
    Plots the phase noise data.
    
    Parameters:
    ----------
    - f: Frequency array (in Hz)
    - phase_noise_profile_list: List of phase noise profiles (in dBc/Hz)
    - tf: Transfer function (default is 1, can be used to plot phase noise of a system)
    - title: Title of the plot
    - xlabel: Label for the x-axis
    - ylabel: Label for the y-axis
    - legend_list: List of labels for each phase noise profile (default None)
    
    Returns:
    -------
    - None, displays the plot
    """
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(10, 6))
    for i, phase_noise_profile in enumerate(phase_noise_profile_list):
        if legend_list is not None:
            label = legend_list[i]
        else:
            label = f'Profile {i+1}'
        plt.semilogx(f, phase_noise_profile, label=label, color=f'C{i}')
    plt.legend(fontsize=16)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(grid)
    
    # Add metric prefix formatter to x-axis
    formatter = FuncFormatter(metric_prefix_formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.tight_layout()
    plt.show()

def get_pm_bw(f: np.array,
              tf_values):
    """
    Computes the phase margin and unity gain bandwidth of a transfer function at given frequencies.
    Parameters:
    ----------
    - f: Frequency array (in Hz)
    - tf_values: Transfer function values (magnitude and phase)

    Returns:
    -------
    - Phase margin in degrees
    """
    
    omega = 2 * np.pi * f
    mag = np.abs(tf_values)
    phase = np.angle(tf_values)
    phase_deg = phase * 180 / np.pi
    # Find the frequency where magnitude crosses 0 dB
    idx = np.where(np.abs(mag) < 1)[0]
    if len(idx) == 0:
        raise ValueError("No crossover frequency found where magnitude crosses 0 dB.")
    crossover_idx = idx[0]
    # Calculate phase margin
    phase_margin = 180 + phase_deg[crossover_idx]
    return phase_margin, f[crossover_idx]

def get_jitter(f: np.array,
               phase_noise_profile: np.array,
               oscillation_frequency: float,
               integration_bandwidth: tuple=(1e3, 1e6)): 
    """
    Computes the integrated phase noise and converts it to time jitter.
    Parameters:
    ----------
    - f: Frequency array (in Hz)
    - phase_noise_profile: Phase noise values (in dBc/Hz)
    - oscillation_frequency: Oscillation frequency in Hz
    - integration_bandwidth: Tuple specifying the integration bandwidth (lower, upper) in Hz
    Returns:
    -------
    - Absolute jitter in radians RMS
    - Absolute jitter in femtoseconds RMS
    """
    pn_util = PhaseNoise(white_noise_100MHz=-500)
    phase_noise_profile_linear = pn_util.to_linear(phase_noise_profile)
    
    # Integrate the phase noise over the specified bandwidth
    jitter_radians_power = np.trapezoid(phase_noise_profile_linear[(f >= integration_bandwidth[0]) & (f <= integration_bandwidth[1])],
                               f[(f >= integration_bandwidth[0]) & (f <= integration_bandwidth[1])])
    # Convert radians to seconds
    jitter_radians_rms: np.float64 = np.sqrt(jitter_radians_power)
    jitter_seconds = jitter_radians_rms/ (2 * np.pi * oscillation_frequency) * 1e15
    return  jitter_radians_rms, jitter_seconds
    
def process_csv(file_path: str, skip_row: int = 0):
    """
    Reads a CSV file and returns the data as a numpy array.
    Parameters:
    ----------
    - file_path: Path to the CSV file
    - skip_row: Number of rows to skip at the beginning of the file (default is 6) from the original code.

    Returns:
    -------
    - Data as a numpy array
    """
    try:
        data = np.genfromtxt(file_path, delimiter=',', skip_header=skip_row)
        return data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def jitter_breakdown(f_offset: np.array,
                     phase_noise_profile_list: list,
                     legend_list: list,
                     oscillation_frequency: float,
                     integration_bandwidth: tuple=(1e3, 1e6)):
    """
    Computes the jitter breakdown for a list of phase noise profiles.
    Parameters:
    ----------
    - f_offset: Frequency offset array (in Hz)
    - phase_noise_profile_list: List of phase noise profiles (in dBc/Hz). The first element is the total phase noise profile.
    - oscillation_frequency: Oscillation frequency in Hz
    - integration_bandwidth: Tuple specifying the integration bandwidth (lower, upper) in Hz
    Returns:
    -------
    - jitter breakdown table
    """
    jitter_list = [] 
    for phase_noise_profile in phase_noise_profile_list:
        jitter_radians, jitter_seconds = get_jitter(f_offset, phase_noise_profile, oscillation_frequency, integration_bandwidth)
        jitter_power_radians = jitter_radians ** 2
        jitter_list.append((jitter_radians, jitter_seconds, jitter_power_radians))
    
    # Print the jitter breakdown in a table
    arr = np.array(jitter_list)
    df = pd.DataFrame(arr, columns=['RJ[rad]', 'RJ[fs]', 'Jitter Power[rad^2]'], index=legend_list)
    df.style.format(precision=4)
    
    return df

def jitter_breakdown_piechart(df: pd.DataFrame,
                              title: str='Jitter Breakdown',
                              figsize: tuple=(7, 6),
                              save_path: str=None,
                              show_total_center: bool=False,
                              colors: list=None):
    """
    Plots a professional publication-quality pie chart of the jitter breakdown.
    
    Parameters:
    ----------
    - df: DataFrame containing the jitter breakdown
    - title: Title of the pie chart (not displayed, for reference only)
    - figsize: Size of the figure (default: 7x6 for compact size)
    - save_path: If provided, saves the figure to this path (e.g., 'figure.pdf' or 'figure.png')
    - show_total_center: If True, shows total jitter in center instead of side
    - colors: Custom color list for pie slices (default: professional color scheme)
    
    Returns:
    -------
    - total_jitter_fs: Total jitter value in femtoseconds
    """
    # Set Helvetica font
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    
    # Professional color scheme (colorblind-friendly)
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # Make background transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Save total jitter from the first row before discarding it
    total_jitter_fs = df.iloc[0]['RJ[fs]']
    
    # discard the first row since it is the total jitter
    df = df.iloc[1:]
    
    # Create custom autopct function to show both percentage and jitter value inside slices
    jitter_values = df['RJ[fs]'].values
    
    # Use a counter to track which slice we're on
    counter = {'idx': -1}
    
    def make_autopct(values):
        def my_autopct(pct):
            # Increment counter for each slice
            counter['idx'] += 1
            idx = counter['idx'] % len(values)
            
            # Format with 1 decimal place
            jitter_val = values[idx]
            jitter_str = f'{jitter_val:.0f}'
            pct_str = f'{pct:.0f}'
            
            return f'{pct_str}%\n{jitter_str} fs'
        return my_autopct
    
    # Create the pie chart with enhanced styling
    wedges, texts, autotexts = ax.pie(df['Jitter Power[rad^2]'], 
                                        labels=df.index, 
                                        autopct=make_autopct(jitter_values),
                                        startangle=90,  # Start at top
                                        colors=colors[:len(df)],
                                        pctdistance=0.75,
                                        labeldistance=1.15,
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 2},
                                        textprops={'fontsize': 20})
    
    # Style the text labels (component names) - BIGGER FONTS
    for text in texts:
        text.set_fontsize(24)
        text.set_weight('bold')
    
    # Style the percentage/value text inside slices - BIGGER FONTS
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(22)
        autotext.set_weight('bold')
    
    # Add total jitter annotation to the side
    if show_total_center:
        # Show in center of pie
        ax.text(0, 0, f'Total\n{total_jitter_fs:.0f} fs', 
                ha='center', va='center', fontsize=26, weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2))
    else:
        # Show to the side of the pie chart
        ax.text(1.5, 0, f'Total Jitter:\n{total_jitter_fs:.0f} fs', 
                ha='left', va='center', fontsize=26, weight='bold',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2))
    
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                    transparent=True, format=save_path.split('.')[-1])
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return total_jitter_fs

# def plot_jitter_heatmap_2D(jitter: np.array,
#                            bw: np.array,
#                            pm: np.array,
#                            title: str='Jitter Heatmap',):
#     """
#     Plots a 2D heatmap of the given data.
#     X axis is PLL bandwidth, Y axis is PLL phase margin.
#     """
    
#     plt.figure(figsize=(12, 8))
#     im = plt.imshow(jitter, cmap='RdYlGn_r', aspect ='auto')
    
#     cbar = plt.colorbar(im, orientation='vertical')
#     cbar.set_label('Jitter (fs)', fontsize=14)
    
#     for i in range(jitter.shape[0]):
#         for j in range(jitter.shape[1]):
#             plt.text(j, i, f'{jitter[i, j]:.1f}', ha='center', va='center', color='black', fontsize=8)
    
#     # set x and y ticks, and labels
#     plt.yticks(ticks=np.arange(len(bw)), labels=[f'{bw_val/1e6:.1f}' for bw_val in bw], fontsize=10)
#     plt.xticks(ticks=np.arange(len(pm)), labels=[f'{pm_val:.1f}' for pm_val in pm], fontsize=10)
#     plt.ylabel('PLL Bandwidth (MHz)', fontsize=12)
#     plt.xlabel('PLL Phase Margin (°)', fontsize=12)

#     plt.gca().invert_yaxis()  # Invert the y-axis

#     plt.title(title, fontsize=16)
#     plt.tight_layout()
#     plt.show()

def plot_jitter_heatmap_2D(jitter: np.array,
                           bw: np.array,
                           pm: np.array,
                           title: str='Jitter Heatmap',):
    """
    Plots a 2D heatmap of the given data.
    X axis is PLL bandwidth, Y axis is PLL phase margin.
    """
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(jitter, cmap='RdYlGn_r', aspect='auto')
    
    cbar = plt.colorbar(im, orientation='vertical')
    cbar.set_label('Jitter (fs)', fontsize=14)
    
    for i in range(jitter.shape[0]):
        for j in range(jitter.shape[1]):
            # Add text annotation
            plt.text(j, i, f'{jitter[i, j]:.1f}', ha='center', va='center', color='black', fontsize=8)
            
            # Highlight cells with jitter > 45fs
            if jitter[i, j] > 45:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                   fill=False, edgecolor='red', linewidth=3)
                plt.gca().add_patch(rect)
    
    # set x and y ticks, and labels
    plt.yticks(ticks=np.arange(len(bw)), labels=[f'{bw_val/1e6:.1f}' for bw_val in bw], fontsize=10)
    plt.xticks(ticks=np.arange(len(pm)), labels=[f'{pm_val:.1f}' for pm_val in pm], fontsize=10)
    plt.ylabel('PLL Bandwidth (MHz)', fontsize=12)
    plt.xlabel('PLL Phase Margin (°)', fontsize=12)

    plt.gca().invert_yaxis()  # Invert the y-axis

    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def log_bin_data(x, y, n_bins):
    """Logarithmically bin data"""
    log_x_min = np.log10(x.min())
    log_x_max = np.log10(x.max())
    
    # Create log-spaced bin edges
    log_bin_edges = np.linspace(log_x_min, log_x_max, n_bins + 1)
    bin_edges = 10**log_bin_edges
    
    # Digitize data into bins
    bin_indices = np.digitize(x, bin_edges)
    
    # Calculate bin centers and averages
    bin_centers = []
    bin_averages = []
    
    for i in range(1, len(bin_edges)):
        mask = bin_indices == i
        if np.any(mask):
            bin_centers.append(np.sqrt(bin_edges[i-1] * bin_edges[i]))  # Geometric mean
            bin_averages.append(np.mean(y[mask]))
    
    # return np.array(bin_centers), np.array(bin_averages)
    return np.array(bin_centers), np.array(bin_averages)