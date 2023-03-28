from typing import List, Tuple
from collections import namedtuple
import numpy as np
# import nolds
from scipy import interpolate
from scipy import signal
# from astropy.stats import LombScargle
import matplotlib.pyplot as plt
import matplotlib.style as style
import io
import streamlit as st

# Frequency Methods name
WELCH_METHOD = "welch"
LOMB_METHOD = "lomb"

# Named Tuple for different frequency bands
VlfBand = namedtuple("Vlf_band", ["low", "high"])
LfBand = namedtuple("Lf_band", ["low", "high"])
HfBand = namedtuple("Hf_band", ["low", "high"])


def _create_timestamp_list(nn_intervals: List[float]) -> List[float]:
    """
    Creates corresponding time interval for all nn_intervals
    Parameters
    ---------
    nn_intervals : list
        List of Normal to Normal Interval.
    Returns
    ---------
    nni_tmstp : list
        list of time intervals between first NN-interval and final NN-interval.
    """
    # Convert in seconds
    nni_tmstp = np.cumsum(nn_intervals) / 1000

    # Force to start at 0
    return nni_tmstp - nni_tmstp[0]

def _create_interpolated_timestamp_list(nn_intervals: List[float], sampling_frequency: int = 7) -> List[float]:
    """
    Creates the interpolation time used for Fourier transform's method
    Parameters
    ---------
    nn_intervals : list
        List of Normal to Normal Interval.
    sampling_frequency : int
        Frequency at which the signal is sampled.
    Returns
    ---------
    nni_interpolation_tmstp : list
        Timestamp for interpolation.
    """
    time_nni = _create_timestamp_list(nn_intervals)
    # Create timestamp for interpolation
    nni_interpolation_tmstp = np.arange(0, time_nni[-1], 1 / float(sampling_frequency))
    return nni_interpolation_tmstp

def _get_freq_psd_from_nn_intervals(nn_intervals: List[float], method: str = WELCH_METHOD,
                                    sampling_frequency: int = 4,
                                    interpolation_method: str = "linear",
                                    vlf_band: namedtuple = VlfBand(0.003, 0.04),
                                    hf_band: namedtuple = HfBand(0.15, 0.40)) -> Tuple:
    """
    Returns the frequency and power of the signal.
    Parameters
    ---------
    nn_intervals : list
        list of Normal to Normal Interval
    method : str
        Method used to calculate the psd. Choice are Welch's FFT or Lomb method.
    sampling_frequency : int
        Frequency at which the signal is sampled. Common value range from 1 Hz to 10 Hz,
        by default set to 7 Hz. No need to specify if Lomb method is used.
    interpolation_method : str
        Kind of interpolation as a string, by default "linear". No need to specify if Lomb
        method is used.
    vlf_band : tuple
        Very low frequency bands for features extraction from power spectral density.
    hf_band : tuple
        High frequency bands for features extraction from power spectral density.
    Returns
    ---------
    freq : list
        Frequency of the corresponding psd points.
    psd : list
        Power Spectral Density of the signal.
    """

    timestamp_list = _create_timestamp_list(nn_intervals)

    if method == WELCH_METHOD:
        # ---------- Interpolation of signal ---------- #
        funct = interpolate.interp1d(x=timestamp_list, y=nn_intervals, kind=interpolation_method)

        timestamps_interpolation = _create_interpolated_timestamp_list(nn_intervals, sampling_frequency)
        nni_interpolation = funct(timestamps_interpolation)

        # ---------- Remove DC Component ---------- #
        nni_normalized = nni_interpolation - np.mean(nni_interpolation)

        #  --------- Compute Power Spectral Density  --------- #
        freq, psd = signal.welch(x=nni_normalized, fs=sampling_frequency, window='hann', nfft=4096)

    # elif method == LOMB_METHOD:
    #     freq, psd = LombScargle(timestamp_list, nn_intervals,
    #                             normalization='psd').autopower(minimum_frequency=vlf_band[0],
    #                                                            maximum_frequency=hf_band[1])
    else:
        raise ValueError("Not a valid method. Choose between 'lomb' and 'welch'")

    return freq, psd

def plot_psd(nn_intervals: List[float], method: str = "welch", sampling_frequency: int = 7, interpolation_method: str = "linear", vlf_band: namedtuple = VlfBand(0.003, 0.04), lf_band: namedtuple = LfBand(0.04, 0.15), hf_band: namedtuple = HfBand(0.15, 0.40)):
    """
    Function plotting the power spectral density of the NN Intervals.
    Arguments
    ---------
    nn_intervals : list
        list of Normal to Normal Interval.
    method : str
        Method used to calculate the psd. Choice are Welch's FFT (welch) or Lomb method (lomb).
    sampling_frequency : int
        frequence at which the signal is sampled. Common value range from 1 Hz to 10 Hz, by default
        set to 7 Hz. No need to specify if Lomb method is used.
    interpolation_method : str
        kind of interpolation as a string, by default "linear". No need to specify if lomb method is
        used.
    vlf_band : tuple
        Very low frequency bands for features extraction from power spectral density.
    lf_band : tuple
        Low frequency bands for features extraction from power spectral density.
    hf_band : tuple
        High frequency bands for features extraction from power spectral density.
    """

    freq, psd = _get_freq_psd_from_nn_intervals(nn_intervals=nn_intervals, method=method,
                                                sampling_frequency=sampling_frequency,
                                                interpolation_method=interpolation_method)

    # Calcul of indices between desired frequency bands
    vlf_indexes = np.logical_and(freq >= vlf_band[0], freq < vlf_band[1])
    lf_indexes = np.logical_and(freq >= lf_band[0], freq < lf_band[1])
    hf_indexes = np.logical_and(freq >= hf_band[0], freq < hf_band[1])

    frequency_band_index = [vlf_indexes, lf_indexes, hf_indexes]
    label_list = ["VLF component", "LF component", "HF component"]

    style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel("Frequency (Hz)", fontsize=15)
    ax.set_ylabel("PSD (s2/ Hz)", fontsize=15)

    if method == "lomb":
        ax.set_title("Lomb's periodogram", fontsize=20)
        for band_index, label in zip(frequency_band_index, label_list):
            ax.fill_between(freq[band_index], 0, psd[band_index] / (1000 * len(psd[band_index])), label=label)
        ax.legend(prop={"size": 15}, loc="best")

    elif method == "welch":
        ax.set_title("FFT Spectrum", fontsize=20)
        for band_index, label in zip(frequency_band_index, label_list):
            ax.fill_between(freq[band_index], 0, psd[band_index] / (1000 * len(psd[band_index])), label=label)
        ax.legend(prop={"size": 15}, loc="best")
        ax.set_xlim(0, hf_band[1])
    else:
        raise ValueError("Not a valid method. Choose between 'lomb' and 'welch'")

    st.pyplot(fig)

