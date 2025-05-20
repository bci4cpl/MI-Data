import os

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal
from data_plots import *
from preprocessing import *
from dwt_svm import *

path = r'C:\Users\owner\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn Recorder\recordings'
lowcut = 4
highcut = 40
order = 6

def data_extract(path, lowcut, highcut, order):
    dir_list = os.listdir(path)
    arr = []
    labels = []
    win = 4.5
    f0 = 50
    Q = 30.0
    desert_labels = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) # left hand MI is 0 and right hand MI is 1
    forest_labels = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) # left hand MI is 0 and right hand MI is 1

    for file in dir_list:
        raw = mne.io.read_raw_bdf(f'{path}/{file}', preload=True)
        fs = int(raw.info['sfreq'])
        channels = raw.info['ch_names']
        trigger_list = np.where(raw.get_data()[-1] == 1)[0]  # take trigger indices
        raw_data = raw.get_data()[:-1]  # data with no trigger
        b, a = notch_filt(f0, Q, fs)
        notched_data = apply_filter(raw_data, b, a)
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        filtered_data = apply_filter(notched_data, b, a)

        seg_raw_data = np.zeros((trigger_list.shape[0], len(channels)-1 , int(win * fs)))
        for index, trig in enumerate(trigger_list):
            seg_raw_data[index] = filtered_data[:, trig: trig + int(win * fs)]
        temp = file.split('.')[0]
        temp = temp.split('_')[-1]
        if 'for' in temp:
            labels.append(forest_labels)
        else:
            labels.append(desert_labels)
        arr.append(seg_raw_data)

    data_arr = np.concatenate(np.array(arr))
    data_labels = np.concatenate(np.array(labels))
    return data_arr, data_labels, fs


def split_data_by_label(data_arr, data_labels):
    """
    Splits the EEG data into left-hand (label 0) and right-hand (label 1) trials.

    Parameters:
        data_arr (ndarray): EEG data of shape (trials, channels, samples).
        data_labels (ndarray): Corresponding labels (0 for left, 1 for right).

    Returns:
        left_hand_data (ndarray): EEG trials labeled as left-hand MI.
        right_hand_data (ndarray): EEG trials labeled as right-hand MI.
    """
    left_indices = np.where(data_labels == 0)[0]
    right_indices = np.where(data_labels == 1)[0]

    left_hand_data = data_arr[left_indices]
    right_hand_data = data_arr[right_indices]

    return left_hand_data, right_hand_data


data_arr, data_labels, fs = data_extract(path, lowcut, highcut, order)

left_data, right_data = split_data_by_label(data_arr,data_labels)
# c3_freqs, c3_times, c3_Sxx = signal.spectrogram(data_arr[0][1], fs, nperseg=256, noverlap=128, nfft=1024, scaling='density')
# plt.figure()
# plt.pcolormesh(c3_times, c3_freqs, 20 * np.log10(c3_Sxx), shading='gouraud')
# plt.colorbar(label='Power [dB]')
# plt.ylabel('Freq [Hz]')
# plt.xlabel('Time [s]')
# plt.grid()
# # plt.show()

# c3_freqs, c3_times, c3_Sxx = signal.spectrogram(data_arr[0][1], fs, nperseg=128, noverlap=64, nfft=256, scaling='density')
# plt.figure()
# plt.pcolormesh(c3_times[:np.where(c3_freqs > 50)[0][0]], c3_freqs[:np.where(c3_freqs > 50)[0][0]], 20 * np.log10(c3_Sxx[:np.where(c3_freqs > 50)[0][0]]), shading='gouraud')
# plt.colorbar(label='Power [dB]')
# plt.ylabel('Freq [Hz]')
# plt.xlabel('Time [s]')
# plt.grid()
# plt.show()

features = FeatureExtraction(data_labels, mode='offline')
# DWT + CSP features
eeg_features = features.features_concat(data_arr.astype(np.float64), 'dwt+csp')

svm_model = SVModel()
svm_model.split_dataset(eeg_features, data_labels)
svm_model.train_model(calibrate=True)
rbf_val_acc, y_pred_rbf, rbf_test_accuracy, y_pred_rbf_prob = svm_model.test_model(svm_model.X_test, svm_model.y_test)
print(f'test acc: {rbf_test_accuracy}')

# plot_mean_spectrograms(data_arr,data_labels,fs)

detect_high_mu_power_segments(right_data, np.ones(len(right_data)), fs, ch_index=1)  # C3

detect_high_mu_power_segments(left_data, np.zeros(len(left_data)), fs, ch_index=3)  # C4







