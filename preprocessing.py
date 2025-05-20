import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, filtfilt, decimate, iirnotch, freqz
# import matplotlib.pyplot as plt
from scipy import signal
import mne

def binary_to_direction(number):
    mapping = {0: 'left', 1: 'right'}
    return mapping[number]


def butter_bandpass(lowcut, highcut, fs, order=14):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    return b, a


def apply_filter(data, b, a):
    return filtfilt(b, a, data, axis=-1)


def downsample_eeg_data(eeg_data, original_fs, target_fs):
    # Calculate the downsampling factor
    downsample_factor = int(original_fs // target_fs)

    # Initialize the downsampled EEG data array
    num_subjects, num_channels, num_samples = eeg_data.shape
    downsampled_data = np.zeros((num_subjects, num_channels, num_samples // downsample_factor))

    # Perform downsampling for each subject and channel
    for i in range(num_subjects):
        for j in range(num_channels):
            # Decimate (downsample) the EEG data for the current channel
            downsampled_data[i, j, :] = decimate(eeg_data[i, j, :], downsample_factor)

    return downsampled_data

# Apply notch filter
# Design notch filter
def notch_filt(f0, Q, fs):
    b, a = iirnotch(f0, Q, fs)
    return b, a

