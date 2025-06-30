
import numpy as np
import mne
import os
import sys
from scipy import io

from .properties import Shu_properties as props

def extract_shu_data(sub, filterLim = [8,30], fs = 250, data_dir = 'data/shu_dataset/'):

    data_dir = os.path.abspath('data/shu_dataset')
    all_days_dirs = os.listdir(data_dir)
    sub_files = [file_str for file_str in all_days_dirs if sub in file_str]

    all_days_data = []
    
    for file in sub_files:
        file_path = data_dir + '/' + file
        d = io.loadmat(file_path)
        segmentedEEG = mne.filter.filter_data(d['data'].astype(float), fs, filterLim[0], filterLim[1], verbose=0)
        labels = d['labels'][0]

        stackedDict = {'segmentedEEG': segmentedEEG, 'labels': labels, 'fs': fs,
               'chanLabels': None, 'trigLabels': ['left', 'right'], 'trials_N': len(labels)}

        all_days_data.append(stackedDict)

    return all_days_data


def get_EEG_dict():
    all_sub_EEG_dict = {}

    try:
        for sub in props['sub_list']:
            all_sub_EEG_dict[sub] = extract_shu_data(
                sub, props['filterLim'], props['fs'], props['data_dir'])
        
        return all_sub_EEG_dict
    
    except Exception as e:
        print(e)
        print(f'Could\'nt load data files')
    