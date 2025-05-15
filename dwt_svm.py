
from dwt import DWT
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from mne.decoding import CSP
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import roc_curve, auc
import torch
from sklearn.preprocessing import label_binarize
import pickle
import joblib
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedKFold

    
class FeatureExtraction:
    def __init__(self, y_label, mode):
        super().__init__()
        self.mode = mode
        if mode == 'offline':
            self.y_label = y_label
        else:
            self.y_label = None
        self.csp_features = None
        self.dwt_features = None
        self.cnn_features = None

    def extract_csp_features(self, signal):
        # Extract CSP features
        csp = CSP(n_components=99, reg=None, log=True, norm_trace=False)

        if self.mode == 'offline':
            # Fit CSP only on the training data
            csp.fit(signal.astype(np.float64), self.y_label)
            # Save CSP model using joblib
            joblib.dump(csp, 'csp_weights/csp_model.pkl')
            csp_features = csp.transform(signal)
            self.csp_features = np.expand_dims(csp_features, axis=-1)
        else:
            # Load the saved CSP model
            csp = joblib.load('csp_weights/csp_model.pkl')
            signal = signal[np.newaxis, :, :]
            csp_features = csp.transform(signal)
            self.csp_features = np.expand_dims(csp_features, axis=-1)

        # return cnn_latent_features
    
    def extract_dwt_features(self, signal, wavelet='haar'):
        # Apply DWT on the eeg data 
        dwt = DWT(signal, self.mode, wavelet)
        # Extract DWT features
        # features_by_day, _ = dwt.dwt_eeg_band_features_multiday(days_labels)
        dwt_features, _ = dwt.get_dwt_features()
        # if self.mode == 'online':
        #     self.dwt_features = dwt_features.reshape(dwt_features.shape[0], dwt_features.shape[1].shape[1]*dwt_features.shape[2])
        # else:
        self.dwt_features = dwt_features.reshape(dwt_features.shape[0], dwt_features.shape[1], dwt_features.shape[2]*dwt_features.shape[3])
        # return dwt_features
    
    # def extract_cnn_features(self, signal):
    #     self.cnn_features = self.denoiser.model.latent_acc(torch.tensor(signal), self.y_label).detach().cpu().numpy()
        # return cnn_features
    
    # def select_features(self):
    #     # concat the features and perform RFECV using SVM with rbf kernel
    #
    #     # example code:
    #     # selector = RFECV(
    #     #                 SGDRegressor(random_state=1),
    #     #                 cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
    #     #                 ).fit(X_encoded, t)
    #
    #     #             display(X_encoded.loc[:, selector.support_])
    #     return features
    
    def features_concat(self, signal, features_type='dwt+csp'):
        # extract features from the signal
        # if 'cnn' in features_type:
        #     self.extract_cnn_features(signal)
        self.extract_dwt_features(signal)
        self.extract_csp_features(signal)

        # concat features
        if features_type == 'cnn':
            features = self.cnn_features.reshape(self.cnn_features.shape[0], -1)
        elif features_type == 'dwt':
            features = self.dwt_features.reshape(self.dwt_features.shape[0], -1)
        elif features_type == 'csp':
            features = self.csp_features.reshape(self.csp_features.shape[0], -1)
        elif features_type == 'dwt+csp':
            features = np.concatenate((self.dwt_features.reshape(self.dwt_features.shape[0], -1), self.csp_features.reshape(self.csp_features.shape[0], -1)), axis=-1) # concatenate DWT and CSP features
        elif features_type =='dwt+cnn':
            features = np.concatenate((self.dwt_features.reshape(self.dwt_features.shape[0], -1), self.cnn_features.reshape(self.cnn_features.shape[0], -1)), axis=-1) # concatenate DWT and latent CNN features
        elif features_type == 'csp+cnn':
            features = np.concatenate((self.cnn_features.reshape(self.cnn_features.shape[0], -1), self.csp_features.reshape(self.csp_features.shape[0], -1)), axis=-1)
        # elif features_type == 'dwt+csp+cnn':
        else:
            features = np.concatenate((self.dwt_features.reshape(self.dwt_features.shape[0], -1), self.csp_features.reshape(self.csp_features.shape[0], -1), self.cnn_features.reshape(self.cnn_features.shape[0], -1)), axis=-1) # concatenate DWT, CSP and latent CNN features
        return features


class SVModel:
    # def __init__(self, train_days, start_day, end_day, bs):
    def __init__(self):
        # self.days_labels = days_labels
        # self.train_days = train_days
        # self.bs = bs
        # self.start_day = start_day
        # self.end_day = end_day
        # self.start_test = start_test
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.rbf_model = None
        self.losses = []
        self.accuracies = []
        self.roc_aucs = None
        self.rbf_val_acc = None 
        self.y_pred_rbf = None 
        self.rbf_test_accuracy = None
        self.y_pred_rbf_prob = None
        self.fpr = None
        self.tpr = None
        self.roc_aucs_by_day = None
        self.fpr_by_day = None
        self.tpr_by_day = None


    def split_dataset(self, features, labels):
        self.rbf_val_acc = [] 
        self.y_pred_rbf = [] 
        self.rbf_test_accuracy = [] 
        self.y_pred_rbf_prob = []
        if np.unique(labels).shape[0] > 2:
            self.roc_aucs_by_day = dict()
            self.fpr_by_day = dict()
            self.tpr_by_day = dict()
            self.tpr = dict()
            self.fpr = dict()
            self.roc_aucs = dict()
        else:
            self.roc_aucs = []
            self.fpr = []
            self.tpr = [] 
        
        x_temp, self.X_test, y_temp, self.y_test = train_test_split(features, labels, test_size=0.1, random_state=32, stratify=labels)
        # X_train, X_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
        # dwt_data = dwt_features.reshape(dwt_features.shape[0], -1)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(x_temp, y_temp, test_size=0.2, random_state=7, stratify=y_temp)
        # self.y_train, y_test = labels[:train_sessions], labels[train_sessions:]
        self.train_model()
        # unique_days, counts = np.unique(self.days_labels, return_counts=True)
        # train and test within session
        # for ii, day in enumerate(range(self.train_days, self.end_day)):
        #     # signal_features_by_day = features[np.where(self.days_labels == day)[0]]
        #     # print(signal_features_by_day.shape)
        #     # Split the dataset to train and test
        #     # test_sessions_start = np.where(days_labels == day)[0][0]
        #     # test_sessions = np.where(self.days_labels == day)[0][0]  # Number of sessions in the first M days
        #
        #     # print(train_sessions_start)
        #     # if self.bs == 'within':
        #     #     dataset_by_day = X_test[test_sessions_start - train_sessions:np.sum(counts[:day]) - train_sessions]
        #     #     labels_by_day = y_test[test_sessions_start - train_sessions:np.sum(counts[:day]) - train_sessions]
        #     # else:
        #     #     dataset_by_day = X_test[:np.sum(counts[:day]) - train_sessions]
        #     #     labels_by_day = y_test[:np.sum(counts[:day]) - train_sessions]
        #     rbf_val_acc, y_pred_rbf, rbf_test_accuracy, y_pred_rbf_prob = self.test_model(dataset_by_day, labels_by_day)
        #     self.rbf_val_acc.append(rbf_val_acc)
        #     self.y_pred_rbf.append(y_pred_rbf)
        #     self.rbf_test_accuracy.append(rbf_test_accuracy)
        #     self.y_pred_rbf_prob.append(y_pred_rbf_prob)
        #     if np.unique(y_test).shape[0] > 2:
        #         self.fpr[day] = self.fpr_by_day
        #         self.tpr[day] = self.tpr_by_day
        #         self.roc_aucs[day] = self.roc_aucs_by_day
        #     # self.roc_aucs.append(roc_auc)
        # return self.rbf_val_acc, self.y_pred_rbf, self.rbf_test_accuracy, self.y_pred_rbf_prob
        #
            
    def train_model(self, calibrate=False):
        # Give more weight to class 0 (minority class)
        class_weights = {0: 1, 1: 1}  # You can tune these values 0:1.05
        self.rbf_model =  SVC(class_weight=class_weights, kernel='rbf', C=33.0, gamma='scale', probability=True)  # C=5/15 for CNN+CSP+SWT/CSP+DWT
        # self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(features)
        if calibrate:
        # Load the trained model weights from N-1 days
            self.rbf_model = joblib.load('svm_weights/svm_rbf_model.pkl')
        self.rbf_model.fit(self.X_train, self.y_train)
        # Get decision function scores (distance from hyperplane)
        # decision_scores = self.rbf_model.decision_function(self.X_val)
        #
        # # Try different thresholds
        # threshold = 0.1  # default is 0.0
        # y_pred_adjusted = (decision_scores >= threshold).astype(int)
        #
        # print(y_pred_adjusted)
        # Save the trained model to a file
        joblib.dump(self.rbf_model, 'svm_weights/svm_rbf_model.pkl')

        # Predict on validation set
        val_y_pred = self.rbf_model.predict(self.X_val)
        train_y_pred = self.rbf_model.predict(self.X_train)
        # print(f'y_pred adjust: {y_pred_adjusted},\n y pred: {y_pred},\n labels val: {self.y_val}')

        # Calculate accuracy
        train_acc = accuracy_score(self.y_train, train_y_pred)
        val_acc = accuracy_score(self.y_val, val_y_pred)
        # adjust_acc = accuracy_score(self.y_val, y_pred_adjusted)
        # print(f"Validation Accuracy: {acc:.4f}, val acc adjust: {adjust_acc:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Compute and display confusion matrix
        train_cm = confusion_matrix(self.y_train, train_y_pred)
        print("Train Confusion Matrix:")
        print(train_cm)
        val_cm = confusion_matrix(self.y_val, val_y_pred)
        print("Val Confusion Matrix:")
        print(val_cm)

    def test_model(self, X_test, y_test):
        # Evaluate on the test set
        rbf_val_acc = self.rbf_model.score(X_test, y_test)
        y_pred_rbf = self.rbf_model.predict(X_test)
        rbf_test_accuracy = accuracy_score(y_test, y_pred_rbf)
        y_pred_rbf_prob = self.rbf_model.predict_proba(X_test) 
        # roc_auc = roc_auc_score(y_test, y_pred_rbf_prob)
        if y_pred_rbf_prob.shape[1] > 2:
            tpr = dict()
            fpr = dict()
            y_test_bin = label_binarize(y_test, classes=[1, 2, 3, 4])
            for j in range(y_pred_rbf_prob.shape[1]):
                self.fpr_by_day[j], self.tpr_by_day[j], _ = roc_curve(y_test_bin[:, j], y_pred_rbf_prob[:, j])
                self.roc_aucs_by_day[j] = auc(self.fpr_by_day[j], self.tpr_by_day[j])
        else:
            fpr, tpr, _  = roc_curve(y_test, torch.sigmoid(torch.tensor(y_pred_rbf_prob, dtype=torch.float32)).numpy()[:, 1])
            self.fpr.append(fpr)
            self.tpr.append(tpr)
            roc_auc = auc(fpr, tpr)
            self.roc_aucs.append(roc_auc)
        return rbf_val_acc, y_pred_rbf, rbf_test_accuracy, y_pred_rbf_prob

    def test_model_online(self, X_test):
        self.rbf_model = joblib.load('svm_weights/svm_rbf_model.pkl')
        y_pred_rbf = self.rbf_model.predict(X_test)
        return y_pred_rbf


# Parameters
# wavelet = 'haar'

#
# # Instantiate dicts
# raw_eeg_dict = {}
# eeg_asr_dict = {}
# eeg_asr_ae_dict = {}
# eeg_ae_dict = {}
# realizations = 10
#
#






    # --------- train/test the classifier ---------
    # raw eeg, trained on 30 days and test on one test day each time
    # raw_eeg_within_model = SVModel(days_labels, train_days, start_day, end_day, bs='within')
    # eeg_rbf_val_acc, eeg_y_pred_rbf, eeg_rbf_test_accuracy, eeg_y_pred_rbf_prob = raw_eeg_within_model.split_dataset(eeg_features, y_label)
    
    # # eeg + ASR, train on 30 days and test on ascending test days each time
    # eeg_asr_within_model = SVModel(days_labels, train_days, start_day, end_day, bs='within')
    # eeg_asr_rbf_val_acc, eeg_asr_y_pred_rbf, eeg_asr_rbf_test_accuracy, eeg_asr_y_pred_rbf_prob = eeg_asr_within_model.split_dataset(eeg_asr_features, y_label)

    # # eeg + ASR + AE, trained on 30 days and test on ascending test days each time
    # eeg_ae_asr_within_model = SVModel(days_labels, train_days, start_day, end_day, bs='within')
    # eeg_ae_asr_rbf_val_acc, eeg_ae_asr_y_pred_rbf, eeg_ae_asr_rbf_test_accuracy, eeg_ae_asr_y_pred_rbf_prob = eeg_ae_asr_within_model.split_dataset(eeg_ae_asr_features, y_label)

    # # eeg + AE, trained on 30 days and test on ascending test days each time
    # eeg_ae_within_model = SVModel(days_labels, train_days, start_day, end_day, bs='within')
    # eeg_ae_rbf_val_acc, eeg_ae_y_pred_rbf, eeg_ae_rbf_test_accuracy, eeg_ae_y_pred_rbf_prob = eeg_ae_within_model.split_dataset(eeg_ae_features, y_label)

    # # raw eeg, trained on 30 days and test on one test day each time
    # raw_eeg_between_model = SVModel(days_labels, train_days, start_day, end_day, bs='between')
    # eeg_between_val_acc, eeg_y_pred_between, eeg_rbf_between_test_accuracy, eeg_y_pred_between_prob = raw_eeg_between_model.split_dataset(eeg_features, y_label)
    # raw_eeg_dict[reali] = [raw_eeg_between_model, eeg_rbf_between_test_accuracy]
    # # eeg + ASR, train on 30 days and test on ascending test days each time
    # eeg_asr_between_model = SVModel(days_labels, train_days, start_day, end_day, bs='between')
    # eeg_between_asr_rbf_val_acc, eeg_between_asr_y_pred_rbf, eeg_between_asr_rbf_test_accuracy, eeg_between_asr_y_pred_rbf_prob = eeg_asr_between_model.split_dataset(eeg_asr_features, y_label)
    # eeg_asr_dict[reali] = [eeg_asr_between_model, eeg_between_asr_rbf_test_accuracy]
    # # eeg + ASR + AE, trained on 30 days and test on ascending test days each time
    # eeg_ae_asr_between_model = SVModel(days_labels, train_days, start_day, end_day, bs='between')
    # eeg_between_ae_asr_rbf_val_acc, eeg_between_ae_asr_y_pred_rbf, eeg_between_ae_asr_rbf_test_accuracy, eeg_between_ae_asr_y_pred_rbf_prob = eeg_ae_asr_between_model.split_dataset(eeg_ae_asr_features, y_label)
    # eeg_asr_ae_dict[reali] = [eeg_ae_asr_between_model, eeg_between_ae_asr_rbf_test_accuracy]
    # # eeg + AE, trained on 30 days and test on ascending test days each time
    # eeg_ae_between_model = SVModel(days_labels, train_days, start_day, end_day, bs='between')
    # eeg_between_ae_rbf_val_acc, eeg_between_ae_y_pred_rbf, eeg_between_ae_rbf_test_accuracy, eeg_between_ae_y_pred_rbf_prob = eeg_ae_between_model.split_dataset(eeg_ae_features, y_label)
    # eeg_ae_dict[reali] = [eeg_ae_between_model, eeg_between_ae_rbf_test_accuracy]
#
# eeg_models = [raw_eeg_between_model, eeg_asr_between_model, eeg_ae_asr_between_model, eeg_ae_between_model]
# dicts = [raw_eeg_dict, eeg_asr_dict, eeg_asr_ae_dict, eeg_ae_dict]
# model_name = ['Raw EEG', 'EEG + ASR', 'EEG + AE + ASR', 'EEG+AE']

# Check if the directory exists, if not, create it
# if not os.path.exists(directory):
#     os.makedirs(directory)

# plot accuracy (within)
# plt.figure(figsize=(12, 8))
# plt.plot(range(train_days, len(eeg_asr_rbf_test_accuracy) + train_days), eeg_asr_rbf_test_accuracy, '-xb', label='EEG with ASR')
# plt.plot(range(train_days, len(eeg_rbf_test_accuracy) + train_days), eeg_rbf_test_accuracy,'-*r', label='Raw EEG')
# plt.plot(range(train_days, len(eeg_ae_rbf_test_accuracy) + train_days), eeg_ae_rbf_test_accuracy, '-^g', label='EEG with AE')
# plt.plot(range(train_days, len(eeg_ae_asr_rbf_test_accuracy) + train_days), eeg_ae_asr_rbf_test_accuracy, '-o', color='orange', label='EEG with ASR + AE')
# plt.xlabel("Days")
# plt.ylabel("Accuracy")
# plt.grid()
# plt.legend(loc='best')
# # plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_test_acc_graph_day_by_day.png')
# # plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_test_acc_graph_day_by_day.pdf')
# plt.show()


# plot accuracy (between) without realizations

# Assuming dicts are named raw_eeg_dict, eeg_asr_dict, eeg_asr_ae_dict, eeg_ae_dict
# dicts = [raw_eeg_dict, eeg_asr_dict, eeg_asr_ae_dict, eeg_ae_dict]
# dict_names = ['Raw EEG', 'EEG + ASR', 'EEG + AE + ASR', 'EEG + AE']
#
# # Helper function to compute mean and standard deviation across all keys
# def compute_mean_std_across_keys(data_dict):
#     # Stack all second argument lists into a 2D array (shape: num_keys x length_of_values_vector)
#     all_values = np.array([value[1] for value in data_dict.values()])
#     # Compute mean and std along the first axis (across keys)
#     mean_values = np.mean(all_values, axis=0)
#     std_values = np.std(all_values, axis=0)
#     return mean_values, std_values
#
# # 1. Figure 1: Mean and Standard Deviation Plot with Shaded Area
# plt.figure(figsize=(12, 8))
#
# # Assuming all value vectors have the same length, get x-axis length from the first dictionary
# x_values = np.arange(len(list(raw_eeg_dict.values())[0][1]))  # Length of the values vector
# fig_form = ['-x', '-*', '-^', '-o']
# fig_color = ['blue', 'red', 'green', 'orange']
# for i, data_dict in enumerate(dicts):
#     # Compute mean and std for the current dictionary
#     means, stds = compute_mean_std_across_keys(data_dict)
#     # Plot mean with a shaded region for ±std
#     plt.plot(x_values+train_days+1, means, fig_form[i], color=fig_color[i], label=f"{dict_names[i]}")
#     plt.fill_between(x_values+train_days+1, means - stds, means + stds, color=fig_color[i], alpha=0.2)
#
# # Add labels, title, and legend
# plt.xlabel("Days")
# plt.ylabel("Accuracy")
# #plt.title("Mean and Standard Deviation of Values Vector Across Keys")
# plt.legend(loc='lower right', fontsize="xx-large")
# plt.grid()
# plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_test_acc_graph_all_days_ten_reali.png')
# plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_test_acc_graph_all_days_ten_reali.pdf')
# plt.show()
#
# # plot thr auc-roc
# plt.figure(figsize=(12, 8))
# for i, data_dict in enumerate(dicts):
#     roc_aucs_values = np.array([value[0].roc_aucs for value in data_dict.values()])
#     mean_values = np.mean(roc_aucs_values, axis=0)
#     stds = np.std(roc_aucs_values, axis=0)
#     plt.plot(x_values+train_days+1, mean_values, fig_form[i], color=fig_color[i], label=f"{dict_names[i]}")
#     plt.fill_between(x_values+train_days+1, mean_values - stds, mean_values + stds, color=fig_color[i], alpha=0.2)
# # Add labels, title, and legend
# plt.xlabel("Days", fontsize='18')
# plt.ylabel("Accuracy", fontsize='18')
# #plt.title("Mean and Standard Deviation of Values Vector Across Keys")
# # Adjust font size of axis tick values
# plt.tick_params(axis='both', which='major', labelsize=14)  # Increase font size for major ticks
# #plt.title("Mean and Standard Deviation of Values Vector Across Keys")
# plt.legend(loc='lower right', fontsize='xx-large')
# plt.grid()
# plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_test_auroc_graph_all_days_ten_reali.png')
# plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_test_auroc_graph_all_days_ten_reali.pdf')
# plt.show()

    

# # plot accuracy and au-roc for 2 subfigures in the same figure with realizations

# fig, axes = plt.subplots(1, 2, figsize=(18, 8))  # Create a 1x2 grid of subplots

# # Plot 1: Mean and Standard Deviation Plot with Shaded Area
# x_values = np.arange(len(list(raw_eeg_dict.values())[0][1]))  # Length of the values vector
# fig_form = ['-x', '-*', '-^', '-o']
# fig_color = ['blue', 'red', 'green', 'orange']

# for i, data_dict in enumerate(dicts):
#     means, stds = compute_mean_std_across_keys(data_dict)
#     axes[0].plot(x_values + train_days + 1, means, fig_form[i], color=fig_color[i], label=f"{dict_names[i]}")
#     axes[0].fill_between(x_values + train_days + 1, means - stds, means + stds, color=fig_color[i], alpha=0.2)

# # Configure the first subplot
# axes[0].set_xlabel("Days")
# axes[0].set_ylabel("Accuracy")
# axes[0].legend(loc='lower right', fontsize="xx-large")
# axes[0].grid()
# axes[0].text(0.01, 0.99, "A)", transform=axes[0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# # Plot 2: AUROC Plot
# for i, data_dict in enumerate(dicts):
#     roc_aucs_values = np.array([value[0].roc_aucs for value in data_dict.values()])
#     mean_values = np.mean(roc_aucs_values, axis=0)
#     stds = np.std(roc_aucs_values, axis=0)
#     axes[1].plot(x_values + train_days + 1, mean_values, fig_form[i], color=fig_color[i], label=f"{dict_names[i]}")
#     axes[1].fill_between(x_values + train_days + 1, mean_values - stds, mean_values + stds, color=fig_color[i], alpha=0.2)

# # Configure the second subplot
# axes[1].set_xlabel("Days")
# axes[1].set_ylabel("AUROC")
# axes[1].legend(loc='lower right', fontsize='xx-large')
# axes[1].grid()
# axes[1].text(0.01, 0.99, "B)", transform=axes[1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# # Save and show the figure
# plt.tight_layout()
# plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_subplots_all_days_ten_reali.png')
# plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_subplots_all_days_ten_reali.pdf')
# plt.show()
# # Adjusted code for 2 subfigures
# fig, axes = plt.subplots(1, 2, figsize=(18, 8))  # Create a 1x2 grid of subplots

# # Plot 1: Mean and Standard Deviation Plot with Shaded Area
# x_values = np.arange(len(list(raw_eeg_dict.values())[0][1]))  # Length of the values vector
# fig_form = ['-x', '-*', '-^', '-o']
# fig_color = ['blue', 'red', 'green', 'orange']

# for i, data_dict in enumerate(dicts):
#     means, stds = compute_mean_std_across_keys(data_dict)
#     axes[0].plot(x_values + train_days + 1, means, fig_form[i], color=fig_color[i], label=f"{dict_names[i]}")
#     axes[0].fill_between(x_values + train_days + 1, means - stds, means + stds, color=fig_color[i], alpha=0.2)

# # Configure the first subplot
# axes[0].set_xlabel("Days")
# axes[0].set_ylabel("Accuracy")
# axes[0].legend(loc='lower right', fontsize="xx-large")
# axes[0].grid()
# axes[0].text(0.01, 0.99, "A)", transform=axes[0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# # Plot 2: AUROC Plot
# for i, data_dict in enumerate(dicts):
#     roc_aucs_values = np.array([value[0].roc_aucs for value in data_dict.values()])
#     mean_values = np.mean(roc_aucs_values, axis=0)
#     stds = np.std(roc_aucs_values, axis=0)
#     axes[1].plot(x_values + train_days + 1, mean_values, fig_form[i], color=fig_color[i], label=f"{dict_names[i]}")
#     axes[1].fill_between(x_values + train_days + 1, mean_values - stds, mean_values + stds, color=fig_color[i], alpha=0.2)

# # Configure the second subplot
# axes[1].set_xlabel("Days")
# axes[1].set_ylabel("AUROC")
# axes[1].legend(loc='lower right', fontsize='xx-large')
# axes[1].grid()
# axes[1].text(0.01, 0.99, "B)", transform=axes[1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# # Save and show the figure
# plt.tight_layout()
# plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_subplots_all_days_ten_reali.png')
# plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_subplots_all_days_ten_reali.pdf')
# plt.show()

# Create a colormap starting from day 31
# num_days = len(eeg_asr_rbf_test_accuracy)
# cmap = cm.get_cmap("viridis", len(eeg_asr_rbf_test_accuracy))  # Choose the 'viridis' colormap, you can also try others like 'plasma', 'inferno', etc.
#
# # Set up the plot
# # plt.figure(figsize=(10, 8))
# fig, ax = plt.subplots(figsize=(10, 8))  # Create the figure and axes
# # Initialize arrays for computing mean TPR and FPR
# all_fpr = np.linspace(0, 1, 100)  # Common FPR range
# mean_tpr = np.zeros_like(all_fpr)
#
# # Loop through the data and plot the ROC curves
# for idx, day_data in enumerate(eeg_asr_rbf_test_accuracy):
#     # Compute ROC curve and ROC-AUC score
#     tpr = eeg_ae_asr_between_model.tpr
#     fpr = eeg_ae_asr_between_model.fpr
#     roc_auc = eeg_ae_asr_between_model.roc_aucs
#
#         # Interpolate TPR to align FPR values for averaging
#     interpolated_tpr = np.interp(all_fpr, fpr[idx], tpr[idx])
#     interpolated_tpr[0] = 0.0  # Ensure the curve starts at (0, 0)
#     mean_tpr += interpolated_tpr
#
#
#     # Plot the ROC curve with the corresponding color
#     # ax.plot(fpr[idx], tpr[idx], label=f'Day {idx + 1} (AUC = {roc_auc[idx]:.2f})', color=cmap(idx), alpha=0.7)  # Using colormap for color
#     ax.plot(fpr[idx], tpr[idx], color=cmap(idx), alpha=0.7)  # Using colormap for color
# # Calculate the mean TPR across all days
# mean_tpr /= num_days
# mean_auc = np.trapz(mean_tpr, all_fpr)  # Compute the AUC for the mean curve
#
# # Plot the mean ROC curve
# ax.plot(all_fpr, mean_tpr, color='r', lw=2.5, label=f'Mean ROC (AUC = {mean_auc:.2f})')
#
#
# # Plot a random classifier line (diagonal line)
# ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
#
# # Add labels and title
# ax.set_xlabel('False Positive Rate')
# ax.set_ylabel('True Positive Rate')
# ax.set_title('ROC-AUC Curves for Different Days')
#
# # Add a legend
# # ax.legend(loc='lower right')
#
# # Add color bar to indicate the mapping of the colors to the days
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=train_days, vmax=train_days + len(eeg_asr_rbf_test_accuracy)-1))
# sm.set_array([])  # Create an empty array for the colorbar
# plt.colorbar(sm, ax=ax, label='Days')  # Colorbar to indicate the days
#
# # Show grid for readability
# ax.grid(True)
# plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_test_roc_graph_all_days.png')
# plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_test_roc_graph_all_days.pdf')
#
# # Show the plot
# plt.show()
#
# # plot ROC
# if np.unique(y_label).shape[0] > 2:
#     n_classes = np.unique(y_label)
#     tpr = eeg_ae_asr_between_model.tpr
#     fpr = eeg_ae_asr_between_model.fpr
#     roc_auc = eeg_ae_asr_between_model.roc_aucs
#
#     # Initialize the plot
#     plt.figure(figsize=(12, 8))
#
#     # Define the colormap
#     cmap = plt.get_cmap("viridis")
#
#     # Create a list of colors based on the number of unique days and classes
#     days = list(fpr.keys())
#     num_days = len(days)
#     num_classes = len(fpr[days[0]])  # Assuming the number of classes is the same for each day
#
#     # Normalize to get a range of values for the colormap
#     norm = plt.Normalize(0, num_days * num_classes - 1)
#
#     # Loop over each day and class to plot the ROC curves
#     color_idx = 0  # Start from 0 for color mapping
#     for day, class_data in fpr.items():
#         for cls, fpr_values in class_data.items():
#             # Get the corresponding tpr values
#             tpr_values = tpr[day][cls]
#
#             # Compute AUC
#             roc_auc = auc(fpr_values, tpr_values)
#
#             # Map the color based on the colormap and the index
#             color = cmap(norm(color_idx))
#
#             # Plot the ROC curve
#             plt.plot(fpr_values, tpr_values, label=f'{day} - Class {cls+1} (AUC = {roc_auc:.2f})', color=color)
#
#             # Update the color index for the next class
#             color_idx += 1
#
#     # Add plot details
#     plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line for reference
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate', fontsize=14)
#     plt.ylabel('True Positive Rate', fontsize=14)
#     plt.title('ROC Curve for Multiple Test Days and Classes with Colormap', fontsize=16)
#     plt.legend(loc='best', fontsize=10)
#     plt.grid()
#     plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_test_roc_graph_all_days.png')
#     plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_test_roc_graph_all_days.pdf')
#     # Show the plot
#     plt.show()
#
# else:
#     # Create a colormap starting from day 31
#     num_days = len(eeg_rbf_between_test_accuracy)
#     cmap = cm.get_cmap("greens", len(eeg_rbf_between_test_accuracy))  # Choose the 'viridis' colormap, you can also try others like 'plasma', 'inferno', etc.
#
#     # Set up the plot
#     # plt.figure(figsize=(10, 8))
#     fig, ax = plt.subplots(figsize=(10, 8))  # Create the figure and axes
#     # Initialize arrays for computing mean TPR and FPR
#     all_fpr = np.linspace(0, 1, 100)  # Common FPR range
#     mean_tpr = np.zeros_like(all_fpr)
#
#     for mode in range(len(eeg_models)):
#         # Compute ROC curve and ROC-AUC score
#         tpr = eeg_models[mode].tpr
#         fpr = eeg_models[mode].fpr
#         roc_auc = eeg_models[mode].roc_aucs
#         # Loop through the data and plot the ROC curves
#         for idx, day_data in enumerate(eeg_asr_rbf_test_accuracy):
#
#
#             # Interpolate TPR to align FPR values for averaging
#             interpolated_tpr = np.interp(all_fpr, fpr[idx], tpr[idx])
#             interpolated_tpr[0] = 0.0  # Ensure the curve starts at (0, 0)
#             mean_tpr += interpolated_tpr
#
#
#             # Plot the ROC curve with the corresponding color
#             # ax.plot(fpr[idx], tpr[idx], label=f'Day {idx + 1} (AUC = {roc_auc[idx]:.2f})', color=cmap(idx), alpha=0.7)  # Using colormap for color
#             ax.plot(fpr[idx], tpr[idx], color=cmap(idx), alpha=0.7)  # Using colormap for color
#         # Calculate the mean TPR across all days
#         mean_tpr /= num_days
#         mean_auc = np.trapz(mean_tpr, all_fpr)  # Compute the AUC for the mean curve
#
#         # Plot the mean ROC curve
#         ax.plot(all_fpr, mean_tpr, color='r', lw=2.5, label=f'Mean ROC (AUC = {mean_auc:.2f})')
#
#
#         # Plot a random classifier line (diagonal line)
#         ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
#
#         # Add labels and title
#         ax.set_xlabel('False Positive Rate')
#         ax.set_ylabel('True Positive Rate')
#         ax.set_title('ROC-AUC Curves for Different Days')
#
#         # Add a legend
#         # ax.legend(loc='lower right')
#
#         # Add color bar to indicate the mapping of the colors to the days
#         sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=train_days, vmax=train_days + len(eeg_asr_rbf_test_accuracy)-1))
#         sm.set_array([])  # Create an empty array for the colorbar
#         plt.colorbar(sm, ax=ax, label='Days')  # Colorbar to indicate the days
#
#         # Show grid for readability
#         ax.grid(True)
#         plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_{model_name[mode]}_test_roc_graph_all_days.png')
#         plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_{model_name[mode]}_test_roc_graph_all_days.pdf')
#
#         # Show the plot
#         plt.show()
#




# self.losses = []
# self.accuracies = []
# rbf_model =  SVC(kernel='rbf', C=7.0, gamma='scale', probability=True)  # C=5/15 for CNN+CSP+SWT/CSP+DWT
# for day in range(np.unique(days_labels).shape[0]):
#     signal_features_by_day = eeg_asr_features[np.where(days_labels == day)[0]]
#     self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(signal_features_by_day)
#     rbf_model.fit(self.X_train, self.y_train)
#     # Evaluate on the validation set
#     rbf_val_acc = rbf_model.score(self.X_val, self.y_val)
#     y_pred_rbf = rbf_model.predict(self.X_test)
#     rbf_test_accuracy = accuracy_score(self.y_test, self.y_pred_rbf)
#     y_pred_rbf_prob = rbf_model.predict_proba(self.X_test)



# # train_sessions_start = 0
# # for day in range(1, 31):
# # signal_features_by_day = eeg_asr_features[np.where(days_labels == train_days)[0]]
# # print(signal_features_by_day.shape)
# #     # Split the dataset to train and test
# train_sessions = np.where(days_labels == day)[0][0]  # Number of sessions in the first M days
# # if day > 1:
# #     train_sessions_start = np.where(days_labels == day-1)[0][0]
# dataset_by_day = eeg_asr_features[:train_sessions]
# # dwt_data = dwt_features.reshape(dwt_features.shape[0], -1)
# X_train, X_test = dataset_by_day[:dataset_by_day.shape[0]-np.ceil(dataset_by_day.shape[0] * 0.2)], dataset_by_day[dataset_by_day.shape[0]-np.ceil(dataset_by_day.shape[0] * 0.2):]
# print(f'day: {day} | train dataset size: {X_train.shape}, test dataset size: {X_test.shape}')
    
# # check the PSD
# # f_test, Pxx_den_test = scipy.signal.welch(X[0][0], sampling_rate, nperseg=1024, average='median')
# # # f_res, Pxx_den_res = scipy.signal.welch(res_signal[trial][electrode], fs, nperseg=1024, average='median')
# # f_rec, Pxx_den_rec = scipy.signal.welch(denoised_signal[0][0], sampling_rate, nperseg=1024, average='median')
# # plt.figure()
# # plt.semilogy(f_test, Pxx_den_test, ls='-', color='C0', label='Original signal')
# # # plt.semilogy(f_res, Pxx_den_res , ls='-.', color='C1', label='Residual signal')
# # plt.semilogy(f_rec, Pxx_den_rec  , ls='--', color='C3', label='Reconstructed signal')
# # plt.grid()
# # plt.legend(loc='best')
# # plt.xlabel('Frequency [Hz]')
# # plt.ylabel(r'Power $[V^2/Hz]$')
# # plt.show()






# # y_train[np.where(y_train == 0)] = -1 # Convert the labels from {0, 1} to {-1, 1}
# # y_val[np.where(y_val == 0)] = -1 # Convert the labels from {0, 1} to {-1, 1}
# # y_test[np.where(y_test == 0)] = -1 # Convert the labels from {0, 1} to {-1, 1}
# # Define the SVM models with different kernels and L2 regularization using the 'C' parameter
# # Linear model
# # linear_model = LinearSVC(C=1.0, max_iter=5000, random_state=42)  # L2 regularization 
# # # LinearSVC(C=1.0, max_iter=1000, random_state=42)
# # # linear_model = SVC(kernel='linear', C=1.0)
# # linear_model.fit(X_train, y_train)
# # # Evaluate on the validation set
# # linear_val_acc = linear_model.score(X_val, y_val)
# # y_pred = linear_model.predict(X_test)
# # linear_test_accuracy = accuracy_score(y_test, y_pred)
# # # Quadratic model
# # quadratic_model = SVC(kernel='poly', C=20.0, degree=2, gamma='scale')
# # quadratic_model.fit(X_train, y_train)
# # # Evaluate on the validation set
# # quadratic_val_acc = quadratic_model.score(X_val, y_val)
# # y_pred = quadratic_model.predict(X_test)
# # quad_test_accuracy = accuracy_score(y_test, y_pred)
# # # Cubic model
# # cubic_model = SVC(kernel='poly', C=30.0, degree=3, gamma='scale')
# # cubic_model.fit(X_train, y_train)
# # # Evaluate on the validation set
# # cubic_val_acc = cubic_model.score(X_val, y_val)
# # y_pred = cubic_model.predict(X_test)
# # cubic_test_accuracy = accuracy_score(y_test, y_pred)
# # rbf model
# rbf_model =  SVC(kernel='rbf', C=7.0, gamma='scale', probability=True)  # C=5/15 for CNN+CSP+SWT/CSP+DWT
# rbf_model.fit(X_train, y_train)
# # Evaluate on the validation set
# rbf_val_acc = rbf_model.score(X_val, y_val)
# y_pred_rbf = rbf_model.predict(X_test)
# rbf_test_accuracy = accuracy_score(y_test, y_pred_rbf)
# y_pred_rbf_prob = rbf_model.predict_proba(X_test)

# print(f'linear acc: {linear_test_accuracy}, quadratic acc: {quad_test_accuracy}, cubic acc: {cubic_test_accuracy}, rbf acc: {rbf_test_accuracy}')

# # X_train, X_temp, y_train, y_temp = train_test_split(dwt_features.reshape(dwt_features.shape[0], -1), y_label, test_size=0.3, random_state=42)
# # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# # Define the hyperparameter grid
# # param_grid = {
# #         'lr': [5e-3],
# #         'momentum': [0.8],
# #         'batch_size': [18],
# #         'weight_decay': [0.02]  # Add batch sizes if you want to vary this too
# #     }


# # Train the svm model
# # best_model = utils.grid_search_svm(X_train, y_train, X_val, y_val, param_grid, epochs=2000)

# # Test the svm model
# # preds = best_model(torch.tensor(X_test, dtype=torch.float32))
# # y_test[np.where(y_test == 0)] = -1 # Convert the labels from {0, 1} to {-1, 1}
# # cm = confusion_matrix(y_test, (torch.sign(preds).detach().numpy()))
# cm = confusion_matrix(y_test, y_pred_rbf)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# test_acc = (cm[0,0]+ cm[1,1])/ np.sum(cm.flatten()) * 100
# print(f'test acc: {(cm[0,0]+ cm[1,1])/ np.sum(cm.flatten()) * 100}')


# # with torch.no_grad():
# #     # y_prob = torch.sigmoid(best_model(torch.tensor(X_test, dtype=torch.float32))).numpy() 
# #     y_prob = torch.sigmoid(torch.tensor(y_pred_rbf_prob, dtype=torch.float32)).numpy() 

# fpr, tpr, thresholds = roc_curve(y_test, y_pred_rbf_prob[:, 1])  
# roc_auc = auc(fpr, tpr)

# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# # Plot the confusion matrix in the first subplot
# disp.plot(ax=axes[0], colorbar=False) # Plot confusion matrix
# axes[0].set_title(f'Test accuracy: {test_acc:.2f}%')

# # Plot the ROC curve in the second subplot
# axes[1].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='darkorange', lw=2)
# axes[1].plot([0, 1], [0, 1], color='navy', linestyle='--', lw=1)  # Diagonal line
# axes[1].set_title('ROC Curve')
# axes[1].set_xlabel('False Positive Rate')
# axes[1].set_ylabel('True Positive Rate')
# axes[1].legend(loc='best')
# axes[1].grid()

# # Add a suptitle for the figure
# plt.suptitle(f'Performance Evaluation on {end_day-train_days} days, with CNN Features', fontsize=16)
# # Adjust layout and show the plot
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/evaluation_on_{end_day-start_day}_cnn_features.png')
# plt.show()
# # Plot ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

# # a=5















# fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)  # Create a 2x2 grid for subplots
# all_fpr = np.linspace(0, 1, 100)  # Common FPR range for interpolation
#
# for i, data_dict in enumerate(dicts):
#     row, col = divmod(i, 2)  # Determine subplot position
#     ax = axs[row, col]
#
#     # Extract tpr, fpr, and auc for all days
#     tpr_list = dicts[i][0][0].tpr
#     fpr_list = dicts[i][0][0].fpr
#     auc_list = dicts[i][0][0].roc_aucs
#
#     # Set up colormap for days
#     num_days = len(tpr_list)  # Number of days in the dataset
#     cmap = cm.get_cmap("Greens", num_days)
#
#     mean_tpr = np.zeros_like(all_fpr)  # Initialize mean TPR
#     auc_values = []  # Store AUC for each day
#
#     for day_idx in range(num_days):
#         try:
#             # Access TPR, FPR, and AUC for the current day
#             tpr = np.array(tpr_list[day_idx], dtype=float)
#             fpr = np.array(fpr_list[day_idx], dtype=float)
#             auc = auc_list[day_idx]
#
#             # Interpolate TPR to align with common FPR range
#             interpolated_tpr = np.interp(all_fpr, fpr, tpr)
#             interpolated_tpr[0] = 0.0  # Ensure the curve starts at (0, 0)
#             mean_tpr += interpolated_tpr  # Add to mean TPR computation
#             auc_values.append(auc)  # Collect AUC
#
#             # Plot individual ROC curve
#             # ax.plot(fpr, tpr, color=cmap(day_idx), alpha=0.7, label=f"Day {day_idx + 1} (AUC = {auc:.2f})")
#             ax.plot(fpr, tpr, color=cmap(day_idx), alpha=0.7)
#
#
#         except Exception as e:
#             print(f"Skipping day {day_idx} in dataset {i} due to error: {e}")
#             continue
#
#     # Compute and plot mean ROC curve
#     if len(auc_values) > 0:
#         mean_tpr /= num_days  # Average TPR across all days
#         mean_auc = np.mean(auc_values)  # Mean AUC across all days
#
#         # Plot the mean ROC curve
#         ax.plot(all_fpr, mean_tpr, color='red', lw=2.5, label=f"Mean AUC = {mean_auc:.2f}")
#         ax.plot([0, 1], [0, 1], 'k--', label="Random AUC = 0.5")
#     else:
#         print(f"No valid data for dataset {i}")
#
#     # Add subplot title and grid
#     ax.set_title(f"{model_name[i]}")
#     ax.grid(True)
#     ax.legend(loc='lower right')
#
# # Add shared axes labels
# fig.supxlabel("False Positive Rate")
# fig.supylabel("True Positive Rate")
#
# # Add color bar to indicate the mapping of the colors to the days
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=train_days, vmax=train_days + num_days-1))
# sm.set_array([])  # Create an empty array for the colorbar
# fig.tight_layout()
# fig.subplots_adjust(right=0.9)
# cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
# fig.colorbar(sm, cax=cbar_ax, label='Days')  # Colorbar to indicate the days
#
# # Show grid for readability
# ax.grid(True)
# plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_test_roc_graph_all_days_reali_5.png')
# plt.savefig(f'C:/Users/owner/Desktop/Git_Repo/Motor imagery skill/Figures/Yoav/{data}/{subject_id}/accuracy/{subject_id}_test_roc_graph_all_days_reali_5.pdf')
#
#
# # Adjust layout
# #plt.tight_layout()
# plt.show()