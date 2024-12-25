
"""Module: parameters

This module defines various parameters and functions for a Brain-Computer Interface (BCI) project
using the MNE library.

The data of this project (MafatBCI) is located in the following Google drive link:
https://drive.google.com/drive/folders/1b9YM1MjC1b5-gJWsq_VC1p9NYoWBJP9Z?usp=sharing

"""

from datetime import datetime
# stream data
import os
import numpy as np
import scipy

# Head Set Params - CHNAGE WHEN REPLACING HEADSET! ############################################
unicorn_electrodes = ["Fz","C3", "Cz", "C4", "Pz", "P07", "P08", "Oz", "trigger"]
samplingRate = 250  # Sampling frequency (Hz)
# filter parameters
ch_types = ['eeg'] * 9
streamName = 'UN-2023.06.26' # MIRIAMS 'UN-2023.06.07'# MY FIRST DEVICE 'UN-2023.06.26'
streamType = 'EEG'
##############################################################################################

# experiment params
output_files = "output_files/"
markers_before_folder_name = "markers_before_folder/"
markers_before_file_name = "markers_before_parse.csv"
markers_after_folder_name = "markers_after_folder/"
markers_after_file_name = "markers_after_parse.csv"
raw_data_folder_name = "Raw/"
filter_data_folder_name = "Filtered/"
filter_data_file_name = "filtered.fif"
models_folder_name = "models/"
csp_weights_folder_name = "csp_weights/"
epochs_total_file_name = "epochs_total.fif"
epochs_right_file_name = "epochs_RIGHT.fif"
epochs_left_file_name = "epochs_LEFT.fif"
baseline_inter_time = 2
exp_inter_time = 4
exp_inter_time_4_sanity_check = 10
right_arrow = "images//right-arrow.jpg"
left_arrow = "images//left-arrow.jpg"
square = "images//square.jpg"
num_of_epochs_per_condition = 10
inbetween_inter_time = 3
baseline_min = 0
baseline_max = baseline_inter_time
instruction_duration = 5
test_inter_time_save = 0.7
set_exp_period_for_segmentation = baseline_inter_time + exp_inter_time + inbetween_inter_time
set_exp_period_for_segmentation_test = baseline_inter_time + exp_inter_time + inbetween_inter_time
test_num_of_epochs = 4

# Specify the file path for saving the XML
mata_data_file_path = "stream_meta_data.xml"

# concurrency params
clear_queue_param = 400
# chunk_max_samples = 100

# feature extraction params
# features_methods = ['kurtosis', 'ptp_amp', 'std', "spect_entropy"]
# features_methods = ['kurtosis', 'std', "spect_entropy"]
features_methods = ['std', "spect_entropy"]
# features_methods = ['kurtosis']
lowerBound = 4 # 3
upperBound = 40 # 40

# model params
# recent_csp_model_filename = 'csp_recent.joblib'
# veteran_csp_model_filename = 'csp_veteran.joblib'
csp_model_filename = 'csp_model.joblib'
model_filename = 'lda_model.joblib'
FBCSP_model_filename = 'FBCSPModel.joblib'


def createFolder(path):
    os.makedirs(path, exist_ok=True)
    return


def getFoldersList(main_folder, starter):
    # get all the experiment folders
    exp_folders = [f for f in os.listdir(main_folder) if
                   os.path.isdir(os.path.join(main_folder, f)) and f.startswith(starter)]


    return exp_folders

def getDataFile(main_folder, end):
    # get all the experiment folders
    exp_folders = [f for f in os.listdir(main_folder) if
                   os.path.isfile(os.path.join(main_folder, f)) and f.endswith(end)]

    return exp_folders


def findLatestDatedFolder(folder_list):
    """
    Find the latest dated folder from a list of folders.

    Args:
        folder_list (list): List of folder paths.

    Returns:
        str: The path to the latest dated folder.
    """
    # Initialize variables to store the latest date and folder path
    latest_date = None
    latest_folder = None

    # Iterate through each folder in the list
    for folder in folder_list:
        first_digit_index = 0
        for idx, ch in enumerate(folder):
            if ch.isdigit():
                first_digit_index = idx
                break
        folder_datetime_str = folder[first_digit_index:]

        # Convert the date string to a datetime object
        try:
            folder_date = datetime.strptime(folder_datetime_str, "%d_%m_%Y at %I_%M_%S_%p")

            # Update the latest_date and latest_folder if the current folder's date is newer
            if latest_date is None or folder_date > latest_date:
                latest_date = folder_date
                latest_folder = folder
        except:
            print("no folder has date")
    # Return the path to the latest dated folder
    return latest_folder



def bandpass(trials, lo, hi, sample_rate):
    a, b = scipy.signal.iirfilter(6, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])
    ntrials = trials.shape[2]
    nsamples = trials.shape[1]
    nchannels = trials.shape[0]
    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:,:,i] = scipy.signal.filtfilt(a, b, trials[:,:,i], axis=1)
    return trials_filt



def cov(trials):
    ntrials = trials.shape[2]
    nsamples = trials.shape[1]
    covs = [trials[:,:,i].dot(trials[:,:,i].T) / nsamples for i in range(ntrials)]
    return np.mean(covs, axis=0)

def whitening(sigma):
    U, l, _ = np.linalg.svd(sigma)
    return U.dot(np.diag(l ** -0.5))

def csp(trials_r, trials_f):
    cov_r = cov(trials_r)
    cov_f = cov(trials_f)
    P = whitening(cov_r + cov_f)
    B, _, _ = np.linalg.svd(P.T.dot(cov_r).dot(P))
    W = P.dot(B)
    return W

def apply_mix(W, trials):
    ntrials = trials.shape[2]
    nchannels = trials.shape[0]
    nsamples = trials.shape[1]
    trials_csp = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
    return trials_csp