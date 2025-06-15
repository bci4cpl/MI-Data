import os
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal
from data_plots import *
from preprocessing import *
from dwt_svm import *
from sklearn.metrics import confusion_matrix, accuracy_score


path = r'output_files/all data gathered may 21st/recordings'
lowcut = 4
highcut = 40
order = 6

def data_extract(path, lowcut, highcut, order):
    dir_list = os.listdir(path)
    arr = []
    labels = []
    win = 8
    pre_sec = 1.0      # seconds *before* trigger  

    f0 = 50
    Q = 30.0
    desert_labels = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) # left hand MI is 0 and right hand MI is 1
    forest_labels = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) # left hand MI is 0 and right hand MI is 1

    for file in dir_list:
        if not file.lower().endswith('.bdf'):
            continue  # Skip non-BDF files

        raw = mne.io.read_raw_bdf(f'{path}/{file}', preload=True)
        fs = int(raw.info['sfreq'])

        #parameters to adjust sample range
        pre_trigger_samples = int(pre_sec * fs)
        post_trigger_samples = int(win * fs) # after trigger samples to include
        total_samples = pre_trigger_samples + post_trigger_samples

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
        seg_raw_data = np.zeros((trigger_list.shape[0], 8, total_samples))

        print(f"File {file!r}  →  seg_raw_data.shape = {seg_raw_data.shape}")

        for index, trig in enumerate(trigger_list):
            start = max(trig - pre_trigger_samples, 0)
            end   = trig + post_trigger_samples
            segment = filtered_data[:, start:end]

            # If you hit the start boundary, pad on the left with zeros
            if start == 0 and segment.shape[1] < total_samples:
                pad_width = total_samples - segment.shape[1]
                segment = np.pad(segment, ((0,0),(pad_width,0)), mode='constant')

            seg_raw_data[index] = segment

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

    # data_arr = np.concatenate(np.array(arr))
    # data_labels = np.concatenate(np.array(labels))
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

# detect_high_mu_power_segments(right_data, fs, ch_index=1)  # C3

# detect_high_mu_power_segments(left_data, fs, ch_index=3)  # C4

# plot_band_derivatives(data_arr, fs, ch_indices=(1,3))

# — now get train & val predictions using your fitted rbf_model —
y_train_pred = svm_model.rbf_model.predict(svm_model.X_train)
y_val_pred   = svm_model.rbf_model.predict(svm_model.X_val)
y_test_pred  = y_pred_rbf

# compute accuracies
train_acc = accuracy_score(svm_model.y_train, y_train_pred) * 100
val_acc   = accuracy_score(svm_model.y_val,   y_val_pred)   * 100
test_acc  = accuracy_score(svm_model.y_test,  y_test_pred)  * 100
# helper to plot one confusion matrix on a given Axes
print(f"Train acc: {train_acc:.1f}%")
print(f"Val   acc: {val_acc:.1f}%")
print(f"Test  acc: {test_acc:.1f}%")

# # — plot all three in one row —
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# plot_cm_ax(axes[0], svm_model.y_train, y_train_pred, ['Right','Left'], "Train CM", confusion_matrix, accuracy_score)
# plot_cm_ax(axes[1], svm_model.y_val,   y_val_pred,   ['Right','Left'], "Val   CM", confusion_matrix, accuracy_score)
# plot_cm_ax(axes[2], svm_model.y_test,  y_test_pred,  ['Right','Left'], "Test  CM", confusion_matrix, accuracy_score)
#
# plt.tight_layout()
# plt.show()


plot_overall_mean_spectrogram_with_envelopes(right_data, fs)
plot_overall_mean_spectrogram_with_envelopes(left_data, fs)
