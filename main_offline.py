import os
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal
from data_plots import *
from preprocessing import *
from dwt_svm import *

path = r'output_files/all data gathered may 21st/recordings'
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
        if not file.lower().endswith('.bdf'):
            continue  # Skip non-BDF files

        raw = mne.io.read_raw_bdf(f'{path}/{file}', preload=True)
        fs = int(raw.info['sfreq'])
        channels = raw.info['ch_names']
        trigger_list = np.where(raw.get_data()[-1] == 1)[0]  # take trigger indices
        raw_data = raw.get_data()[:-1]  # data with no trigger

        # *** if there are more than 8 EEG channels, slice to first 8
        if raw_data.shape[0] != 8:
            print(f"*** {file!r} has {raw_data.shape[0]} EEG channels: {channels[:-1]}")
            print(f"***   slicing to keep only first 8 EEG channels: {channels[:8]}")
            raw_data = raw_data[:8, :]

        b, a = notch_filt(f0, Q, fs)
        notched_data = apply_filter(raw_data, b, a)
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        filtered_data = apply_filter(notched_data, b, a)

        # *** always allocate 8 channels in the middle dimension
        seg_raw_data = np.zeros((trigger_list.shape[0], 8, int(win * fs)))

        # seg_raw_data = np.zeros((trigger_list.shape[0], len(channels)-1 , int(win * fs)))

        print(f"File {file!r}  →  seg_raw_data.shape = {seg_raw_data.shape}")

        for index, trig in enumerate(trigger_list):
            seg_raw_data[index] = filtered_data[:, trig: trig + int(win * fs)]
        temp = file.split('.')[0]
        temp = temp.split('_')[-1]
        if 'for' in temp:
            labels.append(forest_labels)
        else:
            labels.append(desert_labels)
        arr.append(seg_raw_data)

    # *** concatenate directly, now that all seg_raw_data have shape (*, 8, *)
    data_arr = np.concatenate(arr, axis=0)
    data_labels = np.concatenate(labels, axis=0)

    #take only c3 and c4 electrodes for train
    # data_arr = np.hstack([data_arr[:, np.newaxis, 1, :], data_arr[:, np.newaxis, 3, :]])

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


features = FeatureExtraction(data_labels, mode='offline')
# DWT + CSP features
eeg_features = features.features_concat(data_arr.astype(np.float64), 'dwt+csp')

svm_model = SVModel()
svm_model.split_dataset(eeg_features, data_labels)
svm_model.train_model(calibrate=True)
rbf_val_acc, y_pred_rbf, rbf_test_accuracy, y_pred_rbf_prob = svm_model.test_model(svm_model.X_test, svm_model.y_test)
print(f'test acc: {rbf_test_accuracy}')

# plot_mean_spectrograms(data_arr,data_labels,fs)

detect_high_mu_power_segments(right_data, fs, ch_index=1)  # C3

detect_high_mu_power_segments(left_data, fs, ch_index=3)  # C4
