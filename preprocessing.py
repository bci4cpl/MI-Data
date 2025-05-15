import pandas as pd
import numpy as np
import scipy 
from scipy.signal import butter, filtfilt, decimate, iirnotch, freqz
# import matplotlib.pyplot as plt
from scipy import signal
import mne



# import the data
# path = r'C:\Users\owner\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn Recorder\desert sessions Rave'
# idle_path = r'C:\Users\owner\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn Recorder\4 min rest Rave'
# idle_bdf_raw = mne.io.read_raw_bdf(f'{idle_path}/UnicornRecorder_05_05_2025_14_56_31.bdf', preload=True)
# bdf_file = 'UnicornRecorder_06_05_2025_15_44_52.bdf'
# raw = mne.io.read_raw_bdf(path + '\\' + bdf_file, preload=True)
# fs = int(raw.info['sfreq'])  # Sampling frequency
# trial = 7
#
# # filter parameters
# lowcut = 4.0 # High frequency to be removed from signal (Hz) (BPF)
# highcut = 40.0 # High frequency to be removed from signal (Hz) (BPF)
# f0 = 50    # Frequency to be removed from signal (Hz) (notch filter)
# Q = 30.0   # Quality factor (higher Q = narrower notch)
# # new_fs = 128
# beta = 14  # Shape parameter for the Kaiser window
# window = 'kaiser'  # Specify the Kaiser window
# channels = raw.info['ch_names']
# ch_name = np.array(['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'])
# trigger_list = np.where(raw.get_data()[-1]==1)[0] #take trigger indicies
# raw_data = raw.get_data()[:-1] #data with no trigger
# idle_data = idle_bdf_raw.get_data()[:-1]
# seg_raw_data = np.zeros((trigger_list.shape[0],len(channels)-1 , int(4.5 * fs)))
# desert_labels = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) # left hand MI is 0 and right hand MI is 1
# forest_labels = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) # left hand MI is 0 and right hand MI is 1

# for index, trig in enumerate(trigger_list):
#     seg_raw_data[index] = raw_data[: ,trig : trig + int(4.5 * fs)]

def binary_to_direction(number):
    mapping = {0: 'left', 1: 'right'}
    return mapping[number]

def butter_bandpass(lowcut, highcut, fs, order=14):
    # nyquist = 0.5 * fs
    # low = lowcut / nyquist
    # high = highcut / nyquist
    # b, a = butter(order, [low, high], fs=fs, btype='band')
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    w, h = signal.freqz(b, a, fs=fs, worN=2000)
    # plt.plot(w, 20 * np.log10(abs(h)), label="order = %d" % order)
    # # w, h = scipy.signal.freqs(b, a)
    # # plt.semilogy(w, 20 * np.log10(abs(h)))
    # plt.title('Butterworth filter frequency response')
    # plt.xlabel('Frequency [radians / second]')
    # plt.ylabel('Amplitude [dB]')
    # plt.margins(0, 0.1)
    # plt.grid(which='both', axis='both')
    # plt.axvline(1.0, color='green') # cutoff frequency
    # plt.axvline(40, color='green')  # cutoff frequency
    # plt.axvline(50, color='m')
    # plt.show()
    return b, a

def apply_filter(data, b, a):
    return filtfilt(b, a, data, axis=-1)

    

# def fir_bandpass_filter_2d(data, lowcut, highcut, fs, order=101):
#     """
#     Apply a FIR bandpass filter to a 2D array of data.
#
#     Parameters:
#     - data: 2D numpy array (channels x samples), where each row is a signal (e.g., EEG data).
#     - lowcut: Lower frequency cut-off for the bandpass filter.
#     - highcut: Upper frequency cut-off for the bandpass filter.
#     - fs: Sampling frequency of the data.
#     - order: Order of the FIR filter (default is 101).
#
#     Returns:
#     - filtered_data: 2D numpy array with the FIR bandpass-filtered data.
#     """
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#
#     # Design the FIR filter using firwin
#     # firwin creates a low-pass filter, so we specify 'bandpass' by passing both low and high cutoff frequencies
#     b = signal.firwin(order, [low, high], pass_zero=False)
#     filtered_data = signal.filtfilt(b, 1.0, data, axis=-1)
#     # Apply the FIR filter to each signal (row) in the data using lfilter
#     # filtered_data = np.zeros_like(data)
#     # for i in range(data.shape[0]):
#     #     # filtered_data[i, :] = signal.lfilter(b, 1.0, data[i, :])
#     #     filtered_data[i, :] = signal.filtfilt(b, 1.0, data[i, :], axis=-1)
#
#     return filtered_data

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



# Frequency response notch(optional visualization)
# freq, h = freqz(b, a, fs=fs)
# plt.plot(freq, 20 * np.log10(abs(h)))
# plt.title('Notch Filter Frequency Response')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Amplitude [dB]')
# plt.grid()
# plt.show()

# Apply notch filter
# Design notch filter
def notch_filt(f0, Q, fs):
    b, a = iirnotch(f0, Q, fs)
    return b, a


# b, a = notch_filt(f0, Q, fs)
# notch_seg_data = apply_filter(raw, b, a)
# notch_seg_data = filtfilt(b, a, seg_raw_data, axis=-1)
# notch_idle_data = filtfilt(b, a, idle_data, axis=-1)
# apply band pass filter on the raw data
# b, a = butter_bandpass(lowcut, highcut, fs, order=6)
# filtered_signal = apply_filter(notch_seg_data, b, a)
# filtered_idle_signal = apply_filter(notch_idle_data, b, a)

# filtered_signal = fir_bandpass_filter_2d(seg_raw_data, lowcut, highcut, fs, order=101)
#
# labels_mask = np.where(desert_labels == 0)[0] # zero is left hand MI
# left_hand_data = filtered_signal[labels_mask]
# right_hand_data = filtered_signal[~labels_mask]
# channel_indices = np.where(np.isin(ch_name, ['C3', 'C4']))[0]
# filtered_signal_c3_c4 = filtered_signal[:,  channel_indices, :]
# # c3_indices = np.where(np.isin(ch_name, 'C3'))[0]
# # c4_indices = np.where(np.isin(ch_name, 'C4'))[0]
# left_hand_data_c3_c4 = left_hand_data[:, channel_indices, :]
# right_hand_data_c3_c4 = right_hand_data[:, channel_indices, :]
# mean_right_hand_c3_c4 = np.mean(right_hand_data_c3_c4, axis=0)
# mean_left_hand_c3_c4 = np.mean(left_hand_data_c3_c4, axis=0)
# idle_data_c3_c4 = filtered_idle_signal[channel_indices, :]

# plt.figure()
# plt.plot(idle_data_c3_c4[0][2000:-1000], label='C3')
# plt.plot(idle_data_c3_c4[1][2000:-1000], label='C4')
# plt.legend(loc='best')
# plt.xlabel('Time [ms]')
# plt.ylabel('Amplitude [uV]')
# plt.title(f'C3 and C4 electrodes idle')
# plt.grid()
# plt.show()

# figure of one epoch of C3 and C4 elec of left hand MI
# sampling = np.arange(int(4.5*fs))
# seconds = sampling / fs

# plt.figure()
# plt.subplot(211)
# plt.plot(seconds, left_hand_data_c3_c4[trial, 0], label='C3')
# plt.plot(seconds, left_hand_data_c3_c4[trial, 1], label='C4')
# plt.legend(loc='best')
# plt.xlabel('Time [ms]')
# plt.ylabel('Amplitude [uV]')
# plt.title(f'C3 and C4 electrodes left hand MI trial {binary_to_direction(desert_labels[trial])}')
# plt.grid()
# plt.subplot(212)
# plt.plot(seconds, right_hand_data_c3_c4[trial, 0], label='C3')
# plt.plot(seconds, right_hand_data_c3_c4[trial, 1], label='C4')
# plt.legend(loc='best')
# plt.xlabel('Time [ms]')
# plt.ylabel('Amplitude [uV]')
# plt.title(f'C3 and C4 electrodes right hand MI trial {binary_to_direction(desert_labels[trial])}')
# plt.grid()
# plt.show()

# figure of the mean of C3 and C4 elec of left hand MI
# plt.figure()
# plt.plot(seconds, mean_left_hand_c3_c4[0], label='C3')
# plt.plot(seconds, mean_left_hand_c3_c4[1], label='C4')
# plt.legend(loc='best')
# plt.xlabel('Time [ms]')
# plt.ylabel('Amplitude [uV]')
# plt.title(f'C3 and C4 electrodes left hand MI mean')
# plt.grid()
# plt.show()


# # Compute the PSD using Welch's method trial 0
# C3_frequencies, C3_psd = signal.welch(left_hand_data_c3_c4[trial, 0], fs, nperseg=1024)
# C4_frequencies, C4_psd = signal.welch(left_hand_data_c3_c4[trial, 1], fs, nperseg=1024)
# C3_frequencies_right, C3_psd_right = signal.welch(right_hand_data_c3_c4[trial, 0], fs, nperseg=1024)
# C4_frequencies_right, C4_psd_right = signal.welch(right_hand_data_c3_c4[trial, 1], fs, nperseg=1024)
# # compute the PSD using Welch for resting record
# C3_frequencies_idle, C3_psd_idle = signal.welch(idle_data_c3_c4[0], fs, nperseg=1024)
# C4_frequencies_idle, C4_psd_idle = signal.welch(idle_data_c3_c4[1], fs, nperseg=1024)
# Plot
# plt.figure(figsize=(12, 8))
# plt.subplot(311)
# plt.semilogy(C3_frequencies, C3_psd, label='C3')
# plt.semilogy(C4_frequencies, C4_psd, label='C4')
# plt.title(f'Power Spectral Density (PSD) - left hand Trial {binary_to_direction(desert_labels[trial])}')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (V²/Hz)')
# plt.legend()
# plt.grid(True)
# plt.subplot(312)
# plt.semilogy(C3_frequencies, C3_psd_right, label='C3')
# plt.semilogy(C4_frequencies, C4_psd_right, label='C4')
# plt.title(f'Power Spectral Density (PSD) - right hand Trial {binary_to_direction(desert_labels[trial])}')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (V²/Hz)')
# plt.legend()
# plt.grid(True)
# plt.subplot(313)
# plt.semilogy(C3_frequencies_idle, C3_psd_idle, label='C3')
# plt.semilogy(C4_frequencies_idle, C4_psd_idle, label='C4')
# plt.title('Power Spectral Density (PSD) - idle')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (V²/Hz)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#
# C3_frequencies, C3_psd = signal.welch(filtered_signal_c3_c4[trial, 0], fs, nperseg=1024)
# C4_frequencies, C4_psd = signal.welch(filtered_signal_c3_c4[trial, 1], fs, nperseg=1024)
# # plot diff power spectrum between elec c3 and c4 (c3-c4)
# plt.figure()
# plt.plot(C3_frequencies[:125], C3_psd[:125], alpha=0.7, label='C3')
# plt.plot(C3_frequencies[:125], C4_psd[:125], alpha=0.7, label='C4')
# plt.plot(C3_frequencies[:125], C3_psd[:125] - C4_psd[:125], label='C3-C4')
# # plt.plot(C3_frequencies[:125], C3_psd_right[:125]-C4_psd_right[:125], label='C3-C4 (right)')
# plt.title(f'Power Spectral Density (PSD) - MI Trial {binary_to_direction(desert_labels[trial])} ')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (V²/Hz)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Compute the spectrogram trial 0
# C3_freqs, C3_times, C3_Sxx = signal.spectrogram(left_hand_data_c3_c4[trial, 0], fs, window=(window, beta), nperseg=256, noverlap=128, nfft=512, scaling='density')
# C4_freqs, C4_times, C4_Sxx = signal.spectrogram(left_hand_data_c3_c4[trial, 1], fs, window=(window, beta), nperseg=256, noverlap=128, nfft=512, scaling='density')
# C3_freqs_right, C3_times_right, C3_Sxx_right = signal.spectrogram(right_hand_data_c3_c4[trial, 0], fs, window=(window, beta), nperseg=256, noverlap=128, nfft=512, scaling='density')
# C4_freqs_right, C4_times_right, C4_Sxx_right = signal.spectrogram(right_hand_data_c3_c4[trial, 1], fs, window=(window, beta), nperseg=256, noverlap=128, nfft=512, scaling='density')
# # compute spectrogram for resting state
# C3_freqs_idle, C3_times_idle, C3_Sxx_idle = signal.spectrogram(idle_data_c3_c4[0, 3000:3000+1125], fs, window=(window, beta), nperseg=256, noverlap=128, nfft=512, scaling='density')
# C4_freqs_idle, C4_times_idle, C4_Sxx_idle = signal.spectrogram(idle_data_c3_c4[1, 3000:3000+1125], fs, window=(window, beta), nperseg=256, noverlap=128, nfft=512, scaling='density')


# Plotting
# fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# # C3 Spectrogram
# im1 = axs[0].pcolormesh(C3_times, C3_freqs, 10 * np.log10(C3_Sxx), shading='gouraud')
# axs[0].set_title(f'Spectrogram - C3 (Trial {binary_to_direction(desert_labels[trial])})')
# axs[0].set_ylabel('Frequency [Hz]')
# axs[0].set_xlabel('Time [s]')
# fig.colorbar(im1, ax=axs[0], label='Power [dB]')
#
# # C4 Spectrogram
# im2 = axs[1].pcolormesh(C4_times, C4_freqs, 10 * np.log10(C4_Sxx), shading='gouraud')
# axs[1].set_title(f'Spectrogram - C4 (Trial {binary_to_direction(desert_labels[trial])})')
# axs[1].set_ylabel('Frequency [Hz]')
# axs[1].set_xlabel('Time [s]')
# fig.colorbar(im2, ax=axs[1], label='Power [dB]')
# plt.suptitle('Left hand MI')
# plt.tight_layout()
# plt.show()
#
# fig, axs = plt.subplots(2, 1, figsize=(12, 8))
#
# # C3 Spectrogram
# im1 = axs[0].pcolormesh(C3_times_idle, C3_freqs_idle, 10 * np.log10(C3_Sxx_idle), shading='gouraud')
# axs[0].set_title('Spectrogram - C3 resting')
# axs[0].set_ylabel('Frequency [Hz]')
# axs[0].set_xlabel('Time [s]')
# fig.colorbar(im1, ax=axs[0], label='Power [dB]')
#
# # C4 Spectrogram
# im2 = axs[1].pcolormesh(C4_times_idle, C4_freqs_idle, 10 * np.log10(C4_Sxx_idle), shading='gouraud')
# axs[1].set_title('Spectrogram - C4 resting')
# axs[1].set_ylabel('Frequency [Hz]')
# axs[1].set_xlabel('Time [s]')
# fig.colorbar(im2, ax=axs[1], label='Power [dB]')
# plt.suptitle('Resting state')
# plt.tight_layout()
# plt.show()


# C3_freqs, C3_times, C3_Sxx = signal.spectrogram(filtered_signal_c3_c4[trial, 0, :], fs, window=(window, beta), nperseg=512, noverlap=256, nfft=512, scaling='density')
# C4_freqs, C4_times, C4_Sxx = signal.spectrogram(filtered_signal_c3_c4[trial, 1, :], fs, window=(window, beta), nperseg=512, noverlap=256, nfft=512, scaling='density')
#
# C3_freqs, C3_times, C3_Sxx = signal.spectrogram(filtered_signal_c3_c4[trial, 0, :], fs, window=(window, beta), nperseg=128, noverlap=64, nfft=128, scaling='density')
# C4_freqs, C4_times, C4_Sxx = signal.spectrogram(filtered_signal_c3_c4[trial, 1, :], fs, window=(window, beta), nperseg=128, noverlap=64, nfft=128, scaling='density')
# # C3 - C4 spectrogram
# plt.figure(figsize=(12, 8))
# im1 = plt.pcolormesh(C3_times, C3_freqs[:int(np.where(C3_freqs < 45)[0].shape[0])], (10 * np.log10(C3_Sxx - C4_Sxx + 1e-08))[:int(np.where(C3_freqs < 45)[0].shape[0])], shading='gouraud')
# plt.title(f'Spectrogram - C3 - C4 (Trial {binary_to_direction(desert_labels[trial])} MI)')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [s]')
# plt.colorbar(im1, label='Power [dB]')
# plt.show()
# # C3 Spectrogram
# im2 = plt.pcolormesh(C3_times, C3_freqs[:int(np.where(C3_freqs < 45)[0].shape[0])], 10 * np.log10(C3_Sxx)[:int(np.where(C3_freqs < 45)[0].shape[0])], shading='gouraud')
# plt.title(f'Spectrogram - C3 (Trial {binary_to_direction(desert_labels[trial])} MI)')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [s]')
# plt.colorbar(im2, label='Power [dB]')
# plt.show()
#
# # c4 spectrogram
# im3 = plt.pcolormesh(C4_times, C4_freqs[:int(np.where(C3_freqs < 45)[0].shape[0])], 10 * np.log10(C4_Sxx)[:int(np.where(C3_freqs < 45)[0].shape[0])], shading='gouraud')
# plt.title(f'Spectrogram - C4 (Trial {binary_to_direction(desert_labels[trial])} MI)')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [s]')
# plt.colorbar(im3, label='Power [dB]')
# plt.show()
#
# # Compute the PSD using Welch's method
# C3_mean_frequencies, C3_mean_psd = signal.welch(mean_left_hand_c3_c4[0], fs, nperseg=1024)
# C4_mean_frequencies, C4_mean_psd = signal.welch(mean_left_hand_c3_c4[1], fs, nperseg=1024)
#
# # plot
# plt.figure(figsize=(10, 5))
# plt.semilogy(C3_mean_frequencies, C3_mean_psd, label='C3')
# plt.semilogy(C4_mean_frequencies, C4_mean_psd, label='C4')
# plt.title(f'Power Spectral Density (PSD) - left hand Trial {binary_to_direction(desert_labels[trial])}')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (V²/Hz)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# # Compute the spectrogram
# C3_mean_freqs, C3_mean_times, C3_mean_Sxx = signal.spectrogram(mean_left_hand_c3_c4[0], fs, nperseg=512, noverlap=256, nfft=512, scaling='density')
# C4_mean_freqs, C4_mean_times, C4_mean_Sxx = signal.spectrogram(mean_left_hand_c3_c4[1], fs, nperseg=512, noverlap=256, nfft=512, scaling='density')
#
# # Plotting
# fig, axs = plt.subplots(2, 1, figsize=(12, 8))
#
# # C3 Spectrogram
# im1 = axs[0].pcolormesh(C3_mean_times, C3_mean_freqs, 10 * np.log10(C3_mean_Sxx), shading='gouraud')
# axs[0].set_title(f'Spectrogram - C3 (Trial {binary_to_direction(desert_labels[trial])})')
# axs[0].set_ylabel('Frequency [Hz]')
# axs[0].set_xlabel('Time [s]')
# fig.colorbar(im1, ax=axs[0], label='Power [dB]')
#
# # C4 Spectrogram
# im2 = axs[1].pcolormesh(C4_mean_times, C4_mean_freqs, 10 * np.log10(C4_mean_Sxx), shading='gouraud')
# axs[1].set_title(f'Spectrogram - C4 (Trial {binary_to_direction(desert_labels[trial])})')
# axs[1].set_ylabel('Frequency [Hz]')
# axs[1].set_xlabel('Time [s]')
# fig.colorbar(im2, ax=axs[1], label='Power [dB]')
# plt.suptitle('Mean of left hand MI')
# plt.tight_layout()
# plt.show()



