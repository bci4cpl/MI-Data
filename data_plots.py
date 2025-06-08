import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from preprocessing import binary_to_direction


def plot_idle(idle_data_c3_c4):
    plt.figure()
    plt.plot(idle_data_c3_c4[0][2000:-1000], label='C3')
    plt.plot(idle_data_c3_c4[1][2000:-1000], label='C4')
    plt.legend()
    plt.xlabel('Time [ms]')
    plt.ylabel('Amplitude [uV]')
    plt.title('Idle C3 and C4')
    plt.grid()
    plt.show()


def plot_trial(trial, fs, left_data, right_data, labels):
    seconds = np.arange(int(4.5 * fs)) / fs
    plt.figure()
    plt.subplot(211)
    plt.plot(seconds, left_data[trial, 0], label='C3')
    plt.plot(seconds, left_data[trial, 1], label='C4')
    plt.legend()
    plt.title(f'Left MI trial {binary_to_direction(labels[trial])}')
    plt.grid()

    plt.subplot(212)
    plt.plot(seconds, right_data[trial, 0], label='C3')
    plt.plot(seconds, right_data[trial, 1], label='C4')
    plt.legend()
    plt.title(f'Right MI trial {binary_to_direction(labels[trial])}')
    plt.grid()
    plt.show()


def plot_mean(fs, mean_left, mean_right):
    seconds = np.arange(mean_left.shape[1]) / fs
    plt.figure()
    plt.plot(seconds, mean_left[0], label='C3 - Left')
    plt.plot(seconds, mean_left[1], label='C4 - Left')
    plt.plot(seconds, mean_right[0], label='C3 - Right')
    plt.plot(seconds, mean_right[1], label='C4 - Right')
    plt.legend()
    plt.title('Mean MI Trials')
    plt.grid()
    plt.show()


def plot_psd(fs, left_trial, right_trial, idle_c3_c4, trial):
    C3_freq, C3_psd = signal.welch(left_trial[trial, 0], fs, nperseg=1024)
    C4_freq, C4_psd = signal.welch(left_trial[trial, 1], fs, nperseg=1024)
    C3_freq_r, C3_psd_r = signal.welch(right_trial[trial, 0], fs, nperseg=1024)
    C4_freq_r, C4_psd_r = signal.welch(right_trial[trial, 1], fs, nperseg=1024)
    C3_freq_idle, C3_psd_idle = signal.welch(idle_c3_c4[0], fs, nperseg=1024)
    C4_freq_idle, C4_psd_idle = signal.welch(idle_c3_c4[1], fs, nperseg=1024)

    plt.figure()
    plt.semilogy(C3_freq, C3_psd, label='C3 - Left')
    plt.semilogy(C4_freq, C4_psd, label='C4 - Left')
    plt.semilogy(C3_freq_r, C3_psd_r, label='C3 - Right')
    plt.semilogy(C4_freq_r, C4_psd_r, label='C4 - Right')
    plt.semilogy(C3_freq_idle, C3_psd_idle, label='C3 - Idle')
    plt.semilogy(C4_freq_idle, C4_psd_idle, label='C4 - Idle')
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.legend()
    plt.grid()
    plt.show()

def plot_spectrograms(fs, left_trials, right_trials, labels, trial=0, nperseg = 256, noverlap = 128, nfft = 512):
    """
    Plot spectrograms for C3 and C4 channels for one trial and the idle signal.

    Args:
        fs (float): Sampling frequency.
        left_trials (ndarray): Shape (n_trials, 2, n_samples) for left-hand trials (C3, C4).
        right_trials (ndarray): Same as above for right-hand trials.
        idle_c3_c4 (ndarray): Shape (2, n_samples), idle data for C3 and C4.
        labels (ndarray): Label array for all trials.
        trial (int): Index of trial to plot.
    """
    # Select trial data
    label = labels[trial]
    trial_data = left_trials[trial] if label == 0 else right_trials[trial]
    direction = binary_to_direction(label)

    # Spectrogram parameters
    window = 'hann'
    beta = None  # Not using custom window tuple

    # Spectrograms for C3 and C4 (trial)
    c3_freqs, c3_times, c3_Sxx = signal.spectrogram(
        trial_data[0], fs, window=window, nperseg=nperseg,
        noverlap=noverlap, nfft=nfft, scaling='density'
    )
    c4_freqs, c4_times, c4_Sxx = signal.spectrogram(
        trial_data[1], fs, window=window, nperseg=nperseg,
        noverlap=noverlap, nfft=nfft, scaling='density'
    )

    # Plot trial spectrograms
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    im1 = axs[0].pcolormesh(c3_times, c3_freqs, 10 * np.log10(c3_Sxx), shading='gouraud')
    axs[0].set_title(f'Spectrogram - C3 (Trial {trial}, {direction})')
    axs[0].set_ylabel('Frequency [Hz]')
    axs[0].set_xlabel('Time [s]')
    fig.colorbar(im1, ax=axs[0], label='Power [dB]')

    im2 = axs[1].pcolormesh(c4_times, c4_freqs, 10 * np.log10(c4_Sxx), shading='gouraud')
    axs[1].set_title(f'Spectrogram - C4 (Trial {trial}, {direction})')
    axs[1].set_ylabel('Frequency [Hz]')
    axs[1].set_xlabel('Time [s]')
    fig.colorbar(im2, ax=axs[1], label='Power [dB]')

    plt.suptitle(f'{direction} Motor Imagery Trial {trial}')
    plt.tight_layout()
    plt.show()


def plot_mean_spectrograms(data_arr, data_labels, fs, nperseg=512, noverlap=256, nfft=1024):
    """
    Compute and plot the mean spectrograms for C3 and C4 channels over all left and right trials.

    Args:
        data_arr (ndarray): EEG data of shape (n_trials, n_channels, n_samples).
        data_labels (ndarray): Corresponding labels (0 = left, 1 = right).
        fs (float): Sampling frequency.
        nperseg (int): Segment length for spectrogram.
        noverlap (int): Overlap between segments.
        nfft (int): FFT size.
    """

    # Split trials by label
    left_trials = data_arr[data_labels == 0]
    right_trials = data_arr[data_labels == 1]

    def compute_mean_spectrogram(trials, channel_index):
        spectrograms = []
        for trial in trials:
            f, t, Sxx = signal.spectrogram(
                trial[channel_index], fs,
                window='hann', nperseg=nperseg,
                noverlap=noverlap, nfft=nfft, scaling='density'
            )
            spectrograms.append(Sxx)
        mean_Sxx = np.mean(np.stack(spectrograms), axis=0)
        return f, t, mean_Sxx

    # Compute spectrograms for channels C3 (index 5) and C4 (index 4)
    f, t, left_c3_mean = compute_mean_spectrogram(left_trials, 1)
    _, _, left_c4_mean = compute_mean_spectrogram(left_trials, 3)
    _, _, right_c3_mean = compute_mean_spectrogram(right_trials, 1)
    _, _, right_c4_mean = compute_mean_spectrogram(right_trials, 3)

    # Limit frequency axis to 0–40 Hz
    freq_mask = f <= 40
    f = f[freq_mask]
    left_c3_mean = left_c3_mean[freq_mask, :]
    left_c4_mean = left_c4_mean[freq_mask, :]
    right_c3_mean = right_c3_mean[freq_mask, :]
    right_c4_mean = right_c4_mean[freq_mask, :]

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    vmin = min(np.min(10 * np.log10(left_c3_mean)), np.min(10 * np.log10(right_c3_mean)))
    vmax = max(np.max(10 * np.log10(left_c3_mean)), np.max(10 * np.log10(right_c3_mean)))

    im = axs[0, 0].pcolormesh(t, f, 10 * np.log10(left_c3_mean), shading='gouraud', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title("Left Trials - C3")
    axs[0, 1].pcolormesh(t, f, 10 * np.log10(left_c4_mean), shading='gouraud', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title("Left Trials - C4")

    axs[1, 0].pcolormesh(t, f, 10 * np.log10(right_c3_mean), shading='gouraud', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title("Right Trials - C3")
    axs[1, 1].pcolormesh(t, f, 10 * np.log10(right_c4_mean), shading='gouraud', vmin=vmin, vmax=vmax)
    axs[1, 1].set_title("Right Trials - C4")

    for ax in axs.flat:
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]')

    fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04, label='Power [dB]')
    plt.suptitle("Mean Spectrograms Over All Trials (0–40 Hz)")
    plt.tight_layout()
    plt.show()


def detect_high_mu_power_segments(data_arr, fs, ch_index, band=(4, 14), window_sec=1.0, nperseg=256, noverlap=128, nfft=512):
    """
    Detects time windows with high 8–14 Hz power in the spectrogram for a given EEG channel.
    Args:
        data_arr (ndarray): EEG data of shape (n_trials, n_channels, n_samples).
        fs (float): Sampling rate in Hz.
        ch_index (int): EEG channel index (e.g., 5 for C3).
        band (tuple): Frequency band to analyze (default: 8–14 Hz).
        window_sec (float): Length of window to scan for high power (in seconds).
    """
    all_band_powers = []
    all_times = []

    for trial in data_arr:
        f, t, Sxx = signal.spectrogram(
            trial[ch_index], fs=fs, nperseg=nperseg,
            noverlap=noverlap, nfft=nfft, scaling='density'
        )

        # Select only 8–14 Hz
        band_mask = (f >= band[0]) & (f <= band[1])
        band_power = np.mean(Sxx[band_mask, :], axis=0)  # mean across band

        all_band_powers.append(band_power)
        all_times.append(t)

    all_band_powers = np.array(all_band_powers)  # shape: (n_trials, n_time_bins)
    mean_power = np.mean(all_band_powers, axis=0)
    time_vector = all_times[0]

    # Sliding window over mean power
    window_len = int(window_sec * fs / (nperseg - noverlap))  # convert seconds to spectrogram bins
    scores = np.convolve(mean_power, np.ones(window_len), mode='valid')

    # Detect top windows (e.g., top 3 highest power segments)
    top_k = 3
    top_indices = np.argsort(scores)[-top_k:]

    # Compute spectrogram for plotting
    f_plot, t_plot, Sxx_plot = signal.spectrogram(
        data_arr[0, ch_index], fs=fs, nperseg=nperseg,
        noverlap=noverlap, nfft=nfft, scaling='density'
    )

    # Apply frequency mask
    freq_mask = f_plot <= 40
    f_plot = f_plot[freq_mask]
    Sxx_plot = Sxx_plot[freq_mask, :]


    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t_plot, f_plot, 20 * np.log10(Sxx_plot), shading='gouraud')
    if ch_index == 1:
        plt.title(f"C3 - Example Trial Spectrogram with High 8–14 Hz Segments")
    else:
        plt.title(f"C4 - Example Trial Spectrogram with High 8–14 Hz Segments")

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    # Overlay detected windows
    for idx in top_indices:
        start_t = time_vector[idx]
        end_t = time_vector[min(idx + window_len, len(time_vector) - 1)]
        plt.axvspan(start_t, end_t, color='red', alpha=0.3, label='High Mu Power')

    plt.colorbar(label='Power [dB]')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Top high-power time segments (seconds):")
    for idx in sorted(top_indices):
        print(f"  {time_vector[idx]:.2f}–{time_vector[min(idx + window_len, len(time_vector) - 1)]:.2f} sec")

