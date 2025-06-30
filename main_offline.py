import os
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal
from data_plots import *
from preprocessing import *
from dwt_svm import *
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt, hilbert
from mne.decoding import CSP
from pyriemann.estimation   import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline       import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score




path = r'output_files/all data gathered may 21st/recordings'
lowcut = 4
highcut = 40
order = 6

#notes about training: using pre = 0 and win = 7 with extra features, gives train 93% and test 70%
#using the 7 sec window is the most logical choice to classify between right and left because
#thats the imagination time
def data_extract(path, lowcut, highcut, order):
    dir_list = os.listdir(path)
    arr = []
    labels = []
    win = 7
    pre_sec = 0     # seconds *before* trigger  

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

def augment_with_envelopes_and_diffs(data_arr, fs,
                                     alpha_band=(8,14),
                                     beta_band =(14,40),
                                     mode='both'):
    """
    data_arr : ndarray, shape (n_trials, n_ch, n_samps)
    fs       : sampling rate
    alpha_band, beta_band :   2-tuples of frequency bounds
    mode     : 'hilbert', 'diff', or 'both' (default).
               Controls which extra features are appended.

    Returns
    -------
    aug : ndarray, shape (n_trials, n_ch + N_extra, n_samps)
        where N_extra = 2 if mode in {'hilbert','diff'}, else 4.
    """
    def bandpass(x, fs, band, order=4):
        b, a = butter(order, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
        return filtfilt(b, a, x)

    n_trials, n_ch, n_samps = data_arr.shape

    # decide how many extra channels
    if mode == 'hilbert' or mode == 'diff':
        n_extra = 2
    elif mode == 'both':
        n_extra = 4
    else:
        raise ValueError("mode must be 'hilbert', 'diff', or 'both'")

    aug = np.zeros((n_trials, n_ch + n_extra, n_samps), dtype=float)

    for i in range(n_trials):
        trial = data_arr[i]           # shape (n_ch, n_samps)
        # copy original EEG
        aug[i, :n_ch] = trial

        # compute the grand-average signal across channels
        sig = trial.mean(axis=0)      # shape (n_samps,)

        # bandpassed + Hilbert envelopes
        α = np.abs(hilbert(bandpass(sig, fs, alpha_band)))
        β = np.abs(hilbert(bandpass(sig, fs, beta_band)))

        # normalize envelopes to [0,1]
        α /= (α.max() + 1e-12)
        β /= (β.max() + 1e-12)

        # first differences
        dα = np.concatenate([[0], np.diff(α)])
        dβ = np.concatenate([[0], np.diff(β)])

        # now stack in the requested mode
        col = n_ch
        if mode in ('hilbert', 'both'):
            aug[i, col]   = α;   col += 1
            aug[i, col]   = β;   col += 1
        if mode in ('diff',    'both'):
            aug[i, col]   = dα;  col += 1
            aug[i, col]   = dβ;  col += 1

    return aug

data_arr, data_labels, fs = data_extract(path, lowcut, highcut, order)
# print("this is data_arr:",data_arr)
print("this is data_arr.shape:",np.shape(data_arr))

# print("this is data_labels:",data_labels)
print("this is data_labels.shape:",np.shape(data_labels))

left_data, right_data = split_data_by_label(data_arr,data_labels)

#---------------------------------Original train

# features = FeatureExtraction(data_labels, mode='offline')
# eeg_features = features.features_concat(data_arr.astype(np.float64), 'dwt+csp')

# svm_model = SVModel()
# svm_model.split_dataset(eeg_features, data_labels)      # trains once here
# rbf_val_acc, y_pred_rbf, rbf_test_accuracy, _ = svm_model.test_model(
#     svm_model.X_test, svm_model.y_test)

# # only compute & print final summary here
# y_train_pred = svm_model.rbf_model.predict(svm_model.X_train)
# y_val_pred   = svm_model.rbf_model.predict(svm_model.X_val)
# y_test_pred  = y_pred_rbf

# train_acc = accuracy_score(svm_model.y_train, y_train_pred)*100
# val_acc   = accuracy_score(svm_model.y_val,   y_val_pred)*100
# test_acc  = accuracy_score(svm_model.y_test,  y_test_pred)*100

# print(f"Train acc: {train_acc:.1f}%")
# print(f"Val   acc: {val_acc:.1f}%")
# print(f"Test  acc: {test_acc:.1f}%")


#--------------------------------applying extra features with CSP only
# the results give similar numbers, train = 79-80% and test = 76%
# chossing components = 4 for the csp seems to be the best

# 1) split your raw data
X_train, X_test, y_train, y_test = train_test_split(
    data_arr, data_labels,
    test_size=0.20, stratify=data_labels,
    random_state=42
)

# 2) augment both train and test with envelopes + diffs
X_train_aug = augment_with_envelopes_and_diffs(X_train, fs, mode='diff')
X_test_aug  = augment_with_envelopes_and_diffs(X_test,  fs, mode= 'diff')

# 3) build the pipeline
csp = CSP(n_components=4, reg=None, log=True)
svm = SVC(kernel='rbf', C=33.0, class_weight={0:1,1:1.2}, probability=True)
pipe = Pipeline([('csp', csp), ('svc', svm)])

# 4) internal 5-fold CV on the *augmented* training set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X_train_aug, y_train,
                            cv=cv, scoring='accuracy')
print(f"Train CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 5) final fit on full train
pipe.fit(X_train_aug, y_train)

# 6) evaluate on train, test
y_train_pred = pipe.predict(X_train_aug)
y_test_pred  = pipe.predict(X_test_aug)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test,  y_test_pred)
print(f"Train accuracy: {train_acc:.3f}")
print(f"Test  accuracy: {test_acc:.3f}")

# 7) confusion matrices
print("Train confusion matrix:")
print(confusion_matrix(y_train, y_train_pred))
print("Test confusion matrix:")
print(confusion_matrix(y_test,  y_test_pred))


#-------------------------------- train+val+test, csp+svm+riemannian
#results train 99%, val= 78%, test = 73%

# # 1) Train/Val/Test split
# X_tmp, X_test, y_tmp, y_test = train_test_split(
#     data_arr, data_labels,
#     test_size=0.20,
#     stratify=data_labels,
#     random_state=42
# )
# X_train, X_val, y_train, y_val = train_test_split(
#     X_tmp, y_tmp,
#     test_size=0.25,        # 0.25 * 0.80 = 0.20
#     stratify=y_tmp,
#     random_state=42
# )

# # 2) Dual‐branch feature union: CSP(log-var)  &  Riemannian Tangent-Space
# feats = FeatureUnion([
#     ("csp", Pipeline([
#         ("CSP", CSP(n_components=4, reg=None, log=True))
#     ])),
#     ("riem", Pipeline([
#         ("cov", Covariances(estimator="lwf")),
#         ("ts",  TangentSpace(metric="riemann"))
#     ])),
# ])

# # 3) Final pipeline: features → RBF-SVM
# pipe = Pipeline([
#     ("features", feats),
#     ("svc", SVC(kernel="rbf",
#                 C=33.0,
#                 class_weight={0:1, 1:1.2},
#                 probability=True))
# ])

# # 4) 5-fold CV on TRAIN set only (for quick internal check)
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
# print(f"➤ Train CV (5-fold) accuracy: {100*cv_scores.mean():.1f}% ± {100*cv_scores.std():.1f}%")

# # 5) Fit on full TRAIN, evaluate on TRAIN / VAL / TEST
# pipe.fit(X_train, y_train)

# for name, X, y in [("Train", X_train, y_train),
#                    ("Validation", X_val, y_val),
#                    ("Test", X_test, y_test)]:
#     y_pred = pipe.predict(X)
#     acc    = accuracy_score(y, y_pred) * 100
#     cm     = confusion_matrix(y, y_pred)
#     print(f"\n➤ {name} accuracy: {acc:.1f}%")
#     print(f"{name} confusion matrix:\n{cm}")


#---------------------------------------train,val, test csp + svm
#results very similar to result with no val...: train = 80%, val = 76%, test = 75.6% !

# # 1) First split off 20% as a final TEST set
# X_temp, X_test, y_temp, y_test = train_test_split(
#     data_arr, data_labels,
#     test_size=0.20,
#     stratify=data_labels,
#     random_state=42
# )

# # 2) Then split the remaining 80% into TRAIN (60%) and VAL (20%)
# X_train, X_val, y_train, y_val = train_test_split(
#     X_temp, y_temp,
#     test_size=0.25,      # 0.25 * 0.80 = 0.20 total
#     stratify=y_temp,
#     random_state=42
# )

# # 3) Augment each fold with your envelopes & diffs
# X_train_aug = augment_with_envelopes_and_diffs(X_train, fs, mode='diff')
# X_val_aug   = augment_with_envelopes_and_diffs(X_val,   fs, mode='diff')
# X_test_aug  = augment_with_envelopes_and_diffs(X_test,  fs, mode='diff')

# # 4) Build the CSP→SVM pipeline
# pipe = Pipeline([
#     ("csp", CSP(n_components=4, reg=None, log=True)),
#     ("svc", SVC(kernel='rbf',
#                 C=33.0,
#                 class_weight={0:1, 1:1.2},
#                 probability=True))
# ])

# # 5) Quick 5-fold CV on *TRAIN* only
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(pipe, X_train_aug, y_train,
#                             cv=cv, scoring='accuracy')
# print(f"Train CV (5-fold) accuracy: {100*cv_scores.mean():.1f}% ± {100*cv_scores.std():.1f}%")

# # 6) Fit on the full TRAIN set
# pipe.fit(X_train_aug, y_train)

# # 7) Evaluate on Train / Val / Test
# for name, X, y in [
#     ("Train",      X_train_aug, y_train),
#     ("Validation", X_val_aug,   y_val),
#     ("Test",       X_test_aug,  y_test),
# ]:
#     y_pred = pipe.predict(X)
#     acc    = accuracy_score(y, y_pred) * 100
#     cm     = confusion_matrix(y, y_pred)
#     print(f"\n{name} accuracy: {acc:.1f}%")
#     print(f"{name} confusion matrix:\n{cm}")

#----------------------------------------Plots

# detect_high_mu_power_segments(right_data, fs, ch_index=1)  # C3

# detect_high_mu_power_segments(left_data, fs, ch_index=3)  # C4

# plot_band_derivatives(data_arr, fs, ch_indices=(1,3))
