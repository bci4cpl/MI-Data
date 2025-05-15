import os
import mne
import numpy as np

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
    return data_arr, data_labels

data_arr, data_labels = data_extract(path, lowcut, highcut, order)

features = FeatureExtraction(data_labels, mode='offline')
# DWT + CSP features
eeg_features = features.features_concat(data_arr.astype(np.float64), 'dwt+csp')

svm_model = SVModel()
svm_model.split_dataset(eeg_features, data_labels)
svm_model.train_model(calibrate=True)
rbf_val_acc, y_pred_rbf, rbf_test_accuracy, y_pred_rbf_prob = svm_model.test_model(svm_model.X_test, svm_model.y_test)
print(f'test acc: {rbf_test_accuracy}')


