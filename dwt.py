# from torch.utils.data import random_split, DataLoader, Dataset
# import mne
# import sklearn
# import matplotlib.pyplot as plt
# # from properties import sub201_properties as params
#
# import numpy as np
# import pandas as pd
# import pywt
# from scipy import signal
# import os
import numpy as np
import pywt
from torch.utils.data import Dataset
from scipy.stats import entropy



class DWT(Dataset):
    def __init__(self, denoised_signal, mode, wavelet='haar'):
        self.denoised_signal = denoised_signal
        self.mode = mode
        if mode == 'offline':
            self.num_trials, self.num_electrodes, self.num_samples = denoised_signal.shape
        else:
            self.num_electrodes, self.num_samples = denoised_signal.shape
            self.num_trials = 1
        self.wavelet = wavelet

    def decompose_band(self, signal, wavelet='haar'):
        cA, cD = pywt.dwt(signal, wavelet,mode ='zero')
        return cA, cD

    def custom_wavelet_decomposition_to_array(self, signal, wavelet='haar'):
        # Step 1: Decompose into 0-32 Hz (Low-pass) and 32-64 Hz (High-pass) [Gamma Band]
        cA_32, gamma = self.decompose_band(signal, wavelet=wavelet)

        # Step 2: Decompose 0-32 Hz into 0-16 Hz (Low-pass) and 16-32 Hz (High-pass)
        cA_16, beta_16_32 = self.decompose_band(cA_32, wavelet=wavelet)

        # Step 3: Decompose 0-16 Hz into 0-8 Hz (Low-pass) and 8-16 Hz (High-pass)
        cA_8, cD_8_16 = self.decompose_band(cA_16, wavelet=wavelet)

        # Step 4: Decompose 8-16 Hz into Alpha (8-12 Hz) and Beta-part (12-16 Hz)
        alpha, beta_12_16 = self.decompose_band(cD_8_16, wavelet=wavelet)

        # Step 5: Combine Beta (12-16 Hz) and Beta (16-32 Hz) to get the full Beta (12-32 Hz)
        beta = np.concatenate((beta_12_16, beta_16_32))

        # Step 6: Theta (4-8 Hz) and Delta (0-4 Hz)
        delta, theta = self.decompose_band(cA_8, wavelet=wavelet)

        # Collect all bands in a list or array
        bands = np.array([delta, theta, alpha, beta])

        return bands

    def get_dwt_features(self):
        # Define the number of trials, electrodes, and samples
        num_bands = 4  # Delta, Theta, Alpha, Beta
        num_features = 4  # Mean, Std, Median, Entorpy, Energy, Peak to Peak

        trials, electrodes, samples = self.num_trials, self.num_electrodes, self.num_samples

        # Initialize array to store features for all trials and electrodes
        bands_multichannel_features = np.zeros((trials, electrodes, num_bands+1, num_features))
        bands_multichannel = np.zeros((trials, electrodes), dtype=object)

        for trial in range(trials):
            for electrode in range(electrodes):
                # Apply wavelet decomposition to the signal for each electrode
                # bands = self.custom_wavelet_decomposition_to_array(self.denoised_signal[trial, electrode, :], wavelet=self.wavelet)
                if self.mode == 'online':
                    bands = pywt.wavedec(self.denoised_signal[electrode, :], self.wavelet, level=num_bands)
                else:
                    bands = pywt.wavedec(self.denoised_signal[trial, electrode, :], self.wavelet, level=num_bands)

                bands_multichannel[trial,electrode] = bands
                # Calculate features for each band (Delta, Theta, Alpha, Beta)
                for band_idx, band in enumerate(bands):
                    # Extract features (mean, var, energy)
                    band_mean = np.mean(band)
                    band_var = np.var(band)
                    band_median = np.median(band)
                    band_entropy = entropy(np.abs(band), base=2)
                    # band_energy = np.sum(band ** 2)
                    # band_peak_to_peak = np.ptp(band)




                    # Assign features to the corresponding location
                    bands_multichannel_features[trial, electrode, band_idx, 0] = band_mean
                    bands_multichannel_features[trial, electrode, band_idx, 1] = band_var
                    bands_multichannel_features[trial, electrode, band_idx, 2] = band_median
                    bands_multichannel_features[trial, electrode, band_idx, 3] = band_entropy
                    # bands_multichannel_features[trial, electrode, band_idx, 4] = band_energy
                    # bands_multichannel_features[trial, electrode, band_idx, 5] = band_peak_to_peak

        # if self.mode == 'online':
        #     print(bands_multichannel_features.shape)
        #     bands_multichannel_features = np.squeeze(bands_multichannel_features, axis=0)
        #

        return bands_multichannel_features, bands_multichannel
    

    def dwt_eeg_band_features_multiday(self, days_labels=None):

        # If days_labels are provided, separate by days
        if days_labels is not None:
            unique_days = np.unique(days_labels)
            bands_per_day = []
            features_per_day = []
            for day in unique_days:
                # Get the indices of the trials corresponding to this day
                day_indices = np.where(days_labels == day)[0]
                # Extract the trials for this day
                day_data = self.denoised_signal[day_indices]
                # Use the existing get_dwt_features method to get the DWT features
                day_features,bands = DWT(day_data, self.wavelet).get_dwt_features()
                # day_features,bands = self.get_dwt_features()
                bands_per_day.append(bands)
                features_per_day.append(day_features)

            # Convert to a numpy array with shape (days, trials_per_day, electrodes, bands, features)
            dwt_bands = np.array(bands_per_day)
            dwt_features_by_day = np.array(features_per_day)
            return dwt_features_by_day,dwt_bands
        else:
            # No day labels, treat as single day
            return self.get_dwt_features()

    # def plot_band_power_spectra(self, bands, day, trial, electrode, fs=128, nperseg=256, save_path=None):
    #     Electrodes = ['FC3', 'C1', 'C3', 'C5', 'CP3', 'O1', 'FC4', 'C2', 'C4', 'C6', 'CP4']
    #     band_names = ['Delta (0-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-12 Hz)', 'Beta (12-32 Hz)']
    #     expected_ranges = [(0, 4), (4, 8), (8, 12), (12, 32)]
    #
    #     fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    #     axs = axs.ravel()
    #
    #     for i, (band_name, expected_range) in enumerate(zip(band_names, expected_ranges)):
    #         band_data = bands[day]
    #         band_data = band_data[trial,electrode]
    #         if i == 0 or i == 2:
    #             band_data_inv = pywt.idwt(band_data[i],np.zeros(band_data[i].shape[0]), wavelet='haar',mode='smooth')
    #         else:
    #             band_data_inv = pywt.idwt(np.zeros(band_data[i].shape[0]), band_data[i], wavelet='haar',mode ='smooth')
    #
    #         # Calculate power spectrum using Welch's method
    #         nperseg = band_data[i].shape[0]
    #         plt.plot(band_data_inv)
    #         f, Pxx = signal.welch(band_data_inv, fs=fs, nperseg=nperseg)
    #
    #         # Plot power spectrum
    #         axs[i].semilogy(f, Pxx)
    #         axs[i].set_xlabel('Frequency [Hz]')
    #         axs[i].set_ylabel('PSD [V**2/Hz]')
    #         axs[i].set_title(f'{band_name} Power Spectrum')
    #         axs[i].set_xlim(0, fs / 2)  # Limit x-axis to Nyquist frequency
    #         axs[i].grid(True)
    #
    #         # Highlight expected frequency range
    #         axs[i].axvspan(expected_range[0], expected_range[1], color='r', alpha=0.3)
    #
    #     plt.tight_layout()
    #     plt.suptitle(f'Power Spectra for Day {day}, Trial {trial}, Electrode {Electrodes[electrode]}', fontsize=16)
    #     plt.subplots_adjust(top=0.92)
    #     if save_path:
    #         plt.savefig(save_path)
    #         plt.close(fig)
    #     else:
    #         plt.show()
    #
    # def plot_all_bands_power_spectra(self, bands, fs=128, nperseg=256, output_dir='power_spectra_plots'):
    #     """
    #     Plot power spectra for all days, trials, and electrodes.
    #
    #     Parameters:
    #     bands (np.ndarray): Array of shape (days, trials, electrodes, bands, samples)
    #     fs (int): Sampling frequency in Hz
    #     nperseg (int): Length of each segment for Welch's method
    #     """
    #     days = bands.shape[0]
    #     trials, electrodes = bands[0].shape
    #     os.makedirs(output_dir, exist_ok=True)
    #     Electrodes = ['FC3', 'C1', 'C3', 'C5', 'CP3', 'O1', 'FC4', 'C2', 'C4', 'C6', 'CP4']
    #
    #     for day in range(days):
    #         for trial in range(trials):
    #             for electrode in range(electrodes):
    #                 filename = f'power_spectra_day{day}_trial{trial}_electrode{Electrodes[electrode]}.png'
    #                 save_path = os.path.join(output_dir, filename)
    #                 self.plot_band_power_spectra(bands, day, trial, electrode, fs, nperseg, save_path)
    #     print(f"All power spectra plots have been saved in the directory: {output_dir}")

