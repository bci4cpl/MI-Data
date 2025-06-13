import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from preprocessing import binary_to_direction
from scipy.signal import spectrogram
import matplotlib as mpl



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

#
# def detect_high_mu_power_segments(data_arr, fs, ch_index,
#                                   band=(4, 14), window_sec=1.0,
#                                   nperseg=256, noverlap=128, nfft=512):
#     """
#     Detects time windows with high 8–14 Hz power in the spectrogram for a given EEG channel,
#     and overlays vertical lines at the times of maximum summed alpha (8–14 Hz) and beta (14–40 Hz) power.
#
#     Args:
#         data_arr (ndarray): EEG data of shape (n_trials, n_channels, n_samples).
#         fs (float): Sampling rate in Hz.
#         ch_index (int): EEG channel index (e.g., 1 for C3).
#         band (tuple): Frequency band to analyze for sliding-window detection (default: 4–14 Hz).
#         window_sec (float): Length of window to scan for high power (in seconds).
#         nperseg (int): Length of each segment for spectrogram.
#         noverlap (int): Number of points to overlap between segments.
#         nfft (int): Number of FFT points.
#     """
#     # 1) Compute mean power time series for the sliding-window band
#     all_band_powers = []
#     all_times = []
#     for trial in data_arr:
#         f, t, Sxx = signal.spectrogram(
#             trial[ch_index], fs=fs,
#             nperseg=nperseg, noverlap=noverlap, nfft=nfft,
#             scaling='density'
#         )
#         band_mask = (f >= band[0]) & (f <= band[1])
#         band_power = np.mean(Sxx[band_mask, :], axis=0)
#         all_band_powers.append(band_power)
#         all_times.append(t)
#     all_band_powers = np.array(all_band_powers)  # (n_trials, n_timebins)
#     mean_power = np.mean(all_band_powers, axis=0)
#     time_vector = all_times[0]
#
#     # 2) Sliding-window scoring (optional, not plotted here)
#     window_len = int(window_sec * fs / (nperseg - noverlap))
#     scores = np.convolve(mean_power, np.ones(window_len), mode='valid')
#     top_k = 3
#     top_indices = np.argsort(scores)[-top_k:]
#
#     # 3) Compute spectrogram for plotting (example trial)
#     f_plot, t_plot, Sxx_plot = signal.spectrogram(
#         data_arr[0, ch_index], fs=fs,
#         nperseg=nperseg, noverlap=noverlap, nfft=nfft,
#         scaling='density'
#     )
#     freq_mask = f_plot <= 40
#     f_plot = f_plot[freq_mask]
#     Sxx_plot = Sxx_plot[freq_mask, :]
#
#     # 4) Find times of maximum summed alpha and beta power
#     alpha_mask = (f_plot >= 8) & (f_plot <= 14)
#     beta_mask  = (f_plot >= 14) & (f_plot <= 40)
#
#     alpha_time_series = np.sum(Sxx_plot[alpha_mask, :], axis=0)
#     beta_time_series  = np.sum(Sxx_plot[beta_mask,  :], axis=0)
#
#     alpha_peak_idx = np.argmax(alpha_time_series)
#     beta_peak_idx  = np.argmax(beta_time_series)
#
#     alpha_time = t_plot[alpha_peak_idx]
#     beta_time  = t_plot[beta_peak_idx]
#
#     # 5) Plot
#     plt.figure(figsize=(12, 6))
#     plt.pcolormesh(t_plot, f_plot, 20 * np.log10(Sxx_plot), shading='gouraud')
#
#     # vertical lines at peak times
#     plt.axvline(alpha_time, color='red',   linestyle='--',
#                 label=f'α peak at {alpha_time:.2f} s')
#     plt.axvline(beta_time,  color='gray',  linestyle='--',
#                 label=f'β peak at {beta_time:.2f} s')
#
#     # title
#     if ch_index == 1:
#         plt.title("C3 – Right Hand MI Spectrogram")
#     else:
#         plt.title("C4 – Left Hand MI Spectrogram")
#
#     plt.xlabel("Time [s]")
#     plt.ylabel("Frequency [Hz]")
#     plt.colorbar(label='Power [dB]')
#
#     # legend
#     plt.legend(loc='upper right')
#
#     plt.tight_layout()
#     plt.clim(vmax=-220)
#     plt.show()
#
#     # print top sliding-window segments
#     print("Top high-power time segments (seconds):")
#     for idx in sorted(top_indices):
#         start = time_vector[idx]
#         end   = time_vector[min(idx + window_len, len(time_vector)-1)]
#         print(f"  {start:.2f}–{end:.2f} sec")
#
#

#
# def detect_high_mu_power_segments(data_arr, fs, ch_index,
#                                  alpha_band=(8, 14),
#                                  beta_band=(14, 40),
#                                  window_sec=1.0,
#                                  nperseg=256, noverlap=128, nfft=512):
#     """
#     Plot spectrogram of channel `ch_index` and mark:
#      - red line at time of max 8–14 Hz power
#      - gray line at time of max 14–40 Hz power
#     """
#     # --- 1) Collect band‐power time‐courses across trials ---
#     all_alpha = []
#     all_beta  = []
#     for trial in data_arr:
#         f, t, Sxx = signal.spectrogram(
#             trial[ch_index], fs, nperseg=nperseg,
#             noverlap=noverlap, nfft=nfft, scaling='density'
#         )
#         # masks
#         m_alpha = (f >= alpha_band[0]) & (f <= alpha_band[1])
#         m_beta  = (f >= beta_band[0])  & (f <= beta_band[1])
#         # mean over freq‐axis
#         all_alpha.append(np.mean(Sxx[m_alpha, :], axis=0))
#         all_beta .append(np.mean(Sxx[m_beta,  :], axis=0))
#
#     all_alpha = np.vstack(all_alpha)  # (n_trials, n_timebins)
#     all_beta  = np.vstack(all_beta)
#
#     mean_alpha = np.mean(all_alpha, axis=0)
#     mean_beta  = np.mean(all_beta,  axis=0)
#
#     # --- 2) Find the time‐index of maximum mean power for each band ---
#     alpha_idx = np.argmax(mean_alpha)
#     beta_idx  = np.argmax(mean_beta)
#     time_vector = t  # same for both
#
#     alpha_time = time_vector[alpha_idx]
#     beta_time  = time_vector[beta_idx]
#
#     # --- 3) Plot the spectrogram ---
#     f_plot, t_plot, Sxx_plot = signal.spectrogram(
#         data_arr[0, ch_index], fs, nperseg=nperseg,
#         noverlap=noverlap, nfft=nfft, scaling='density'
#     )
#     # restrict to 0–40 Hz
#     mask40 = f_plot <= 40
#     f_plot = f_plot[mask40]
#     Sxx_plot = Sxx_plot[mask40, :]
#
#     plt.figure(figsize=(12, 6))
#     plt.pcolormesh(
#         t_plot, f_plot,
#         10 * np.log10(Sxx_plot),
#         shading='gouraud'
#     )
#     plt.xlabel("Time [s]")
#     plt.ylabel("Frequency [Hz]")
#     plt.title(f"Channel {ch_index}: spectrogram with α/β peaks")
#
#     # --- 4) Overlay vertical lines ---
#     plt.axvline(alpha_time, color='red',   linestyle='--', label='max α (8–14 Hz)')
#     plt.axvline(beta_time,  color='gray',  linestyle='--', label='max β (14–40 Hz)')
#
#     plt.colorbar(label="Power [dB]")
#     plt.legend(loc='upper right')
#     plt.tight_layout()
#     plt.show()
#
#     print(f"Max α power at {alpha_time:.2f} s, Max β power at {beta_time:.2f} s")
#



# def detect_high_mu_power_segments(data_arr, fs, ch_index,
#                                   window_sec=1.0,
#                                   nperseg=256, noverlap=128, nfft=512):
#     """
#     Computes the *mean* spectrogram across all trials for channel `ch_index`,
#     then finds and plots the times of maximal summed alpha (8–14 Hz) and beta (14–40 Hz)
#     power on that mean spectrogram.
#     """
#
#     nperseg = fs // 2  # 0.25 s windows
#     noverlap = int(nperseg * 0.70)
#
#     # 1) Compute per-trial spectrograms and average them
#     Sxx_list = []
#     for trial in data_arr:
#         f, t, Sxx = signal.spectrogram(
#             trial[ch_index], fs,
#             nperseg=nperseg, noverlap=noverlap, nfft=nfft,
#             scaling='density'
#         )
#         Sxx_list.append(Sxx)
#     mean_Sxx = np.mean(np.stack(Sxx_list), axis=0)   # shape: (n_freqs, n_times)
#
#     # 2) Mask to 0–40 Hz
#     freq_mask = f <= 40
#     f_plot = f[freq_mask]
#     Sxx_plot = mean_Sxx[freq_mask, :]
#
#     # 3) Compute α and β sums over time
#     alpha_mask = (f_plot >= 8)  & (f_plot <= 14)
#     beta_mask  = (f_plot >= 14) & (f_plot <= 40)
#
#     alpha_ts = np.sum(Sxx_plot[alpha_mask, :], axis=0)
#     beta_ts  = np.sum(Sxx_plot[beta_mask,  :], axis=0)
#
#     alpha_idx = np.argmax(alpha_ts)
#     beta_idx  = np.argmax(beta_ts)
#
#     alpha_time = t[alpha_idx]
#     beta_time  = t[beta_idx]
#
#     # 4) Plot mean spectrogram + vertical lines
#     plt.figure(figsize=(12, 6))
#     plt.pcolormesh(t, f_plot, 20 * np.log10(Sxx_plot), shading='gouraud')
#
#     plt.axvline(alpha_time, color='red',   linestyle='--',
#                 label=f'α peak at {alpha_time:.2f}s')
#     plt.axvline(beta_time,  color='gray',  linestyle='--',
#                 label=f'β peak at {beta_time:.2f}s')
#
#     title = "C3" if ch_index == 1 else "C4"
#     plt.title(f"{title} – Mean Spectrogram with α & β Peaks")
#     plt.xlabel("Time [s]")
#     plt.ylabel("Frequency [Hz]")
#     plt.colorbar(label='Power [dB]')
#     plt.legend(loc='upper right')
#     plt.tight_layout()
#     plt.show()
#
#     # 5) Print matching peak times
#     print(f"α-peak time    = {alpha_time:.2f} s")
#     print(f"β-peak time    = {beta_time:.2f} s")


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_alpha_beta_spectrograms(data_arr, fs, ch_index,
                                 nperseg=256, noverlap=128, nfft=512):
    """
    For channel `ch_index`, compute the mean spectrogram across all trials,
    then plot two panels: (1) α band (8–14 Hz), (2) β band (14–40 Hz).
    Each panel’s y-axis is zoomed to its respective band.
    """
    # 1) Compute per-trial spectrograms and average
    specs = []
    for tr in data_arr:
        f, t, Sxx = spectrogram(
            tr[ch_index], fs,
            window='hann', nperseg=nperseg,
            noverlap=noverlap, nfft=nfft,
            scaling='density'
        )
        specs.append(Sxx)
    mean_Sxx = np.mean(np.stack(specs), axis=0)  # (n_freqs, n_times)

    # 2) Define masks and band limits
    bands = [
        ((f >= 8) & (f <= 14), 8, 14, "Alpha (8–14 Hz)"),
        ((f >= 14) & (f <= 40), 14, 40, "Beta  (14–40 Hz)")
    ]

    # 3) Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, (mask, fmin, fmax, band_name) in zip(axs, bands):
        f_band  = f[mask]
        Sxx_band = mean_Sxx[mask, :]
        im = ax.pcolormesh(
            t, f_band, 10 * np.log10(Sxx_band),
            shading='gouraud', cmap='viridis'
        )
        ax.set_title(f"Channel {ch_index} – {band_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(fmin, fmax)            # *** zoom y-axis to the band
        ax.grid(True)

    axs[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axs.ravel().tolist(),
                 label='Power [dB]', fraction=0.02, pad=0.04)

    plt.suptitle(f"Mean Spectrograms for Channel {ch_index}: α vs. β bands")
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

def detect_high_mu_power_segments(data_arr, fs, ch_index,
                                  window_sec=1.0,
                                  nperseg=None, noverlap=None, nfft=512):
    """
    Computes the mean spectrogram across all trials for channel `ch_index`,
    then finds and plots the times of maximal summed alpha (8–14 Hz) and beta (14–40 Hz)
    power on that mean spectrogram, with α/β envelopes below.
    """
    # Use 0.5s windows with 70% overlap by default
    if nperseg is None:
        nperseg = fs // 2
    if noverlap is None:
        noverlap = int(nperseg * 0.60)

    # 1) Compute per-trial spectrograms and average them
    Sxx_list = []
    for trial in data_arr:
        f, t, Sxx = signal.spectrogram(
            trial[ch_index], fs,
            nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            scaling='density'
        )
        Sxx_list.append(Sxx)
    mean_Sxx = np.mean(np.stack(Sxx_list), axis=0)   # (n_freqs, n_times)

    # 2) Mask frequencies to 0–40 Hz
    freq_mask = f <= 40
    f_plot = f[freq_mask]
    Sxx_plot = mean_Sxx[freq_mask, :]

    # 3) Compute α/β power time series
    alpha_mask = (f_plot >= 8)  & (f_plot <= 14)
    beta_mask  = (f_plot >= 14) & (f_plot <= 40)

    alpha_ts = np.sum(Sxx_plot[alpha_mask, :], axis=0)
    beta_ts  = np.sum(Sxx_plot[beta_mask,  :], axis=0)

    # Find their peak times
    alpha_time = t[np.argmax(alpha_ts)]
    beta_time  = t[np.argmax(beta_ts)]

    # 4) Plot: top = spectrogram, bottom = normalized envelopes
    fig, (ax_spec, ax_env) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
        figsize=(12, 6)
    )

    # Spectrogram on ax_spec
    im = ax_spec.pcolormesh(
        t, f_plot, 20 * np.log10(Sxx_plot),
        shading='gouraud'
    )

    ax_spec.set_xlabel("Time [s]")
    ax_spec.tick_params(axis='x', labelbottom=True)

    ax_spec.set_ylabel("Frequency [Hz]")
    ax_spec.set_title(f"{'C3 - Right Hand MI' if ch_index==1 else 'C4 - Left Hand MI'} – Mean Spectrogram with α & β Peaks")
    ax_spec.grid(True)

    # Envelope plot on ax_env
    ax_env.plot(t, alpha_ts / np.max(alpha_ts),
                color='red',   label='α envelope')
    ax_env.plot(t, beta_ts  / np.max(beta_ts),
                color='gray',  label='β envelope', alpha=0.7)
    ax_env.set_ylabel("Norm. power")
    ax_env.set_xlabel("Time [s]")
    ax_env.legend(loc='upper right')
    ax_env.grid(True)

    # Colorbar for spectrogram
    fig.colorbar(im, ax=ax_spec, label='Power [dB]', pad=0.02)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Print peak times
    print(f"α-peak time = {alpha_time:.2f} s")
    print(f"β-peak time = {beta_time:.2f} s")

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_band_derivatives(data_arr, fs,
                          ch_indices=(1,3),
                          nperseg=None, noverlap=None, nfft=512):
    """
    For each channel in ch_indices, compute the mean α- and β-power
    time series across all trials, then plot their numerical derivatives.
    """
    # default windowing: 0.5s windows @ 70% overlap
    if nperseg is None:
        nperseg  = fs // 2
    if noverlap is None:
        noverlap = int(nperseg * 0.6)

    # Prepare figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, ch in zip(axs, ch_indices):
        # 1) average spectrogram across trials
        Sxx_list = []
        for tr in data_arr:
            f, t, Sxx = spectrogram(tr[ch], fs,
                                     window='hann',
                                     nperseg=nperseg,
                                     noverlap=noverlap,
                                     nfft=nfft,
                                     scaling='density')
            Sxx_list.append(Sxx)
        mean_Sxx = np.mean(np.stack(Sxx_list), axis=0)

        # 2) limit to 0–40 Hz
        freq_mask = f <= 40
        f_plot    = f[freq_mask]
        Sxx_plot  = mean_Sxx[freq_mask, :]

        # 3) extract band-power time series
        alpha_mask = (f_plot >= 8)  & (f_plot <= 14)
        beta_mask  = (f_plot >= 14) & (f_plot <= 40)

        alpha_ts = np.sum(Sxx_plot[alpha_mask, :], axis=0)
        beta_ts  = np.sum(Sxx_plot[beta_mask,  :], axis=0)

        # 4) numerical derivatives
        d_alpha = np.gradient(alpha_ts, t)
        d_beta  = np.gradient(beta_ts,  t)

        # 5) plot
        ax.plot(t, d_alpha, color='red',   label="dα/dt")
        ax.plot(t, d_beta,  color='gray',  label="dβ/dt", alpha=0.7)
        ax.set_title(f"{'C3' if ch==1 else 'C4'} derivative of band-power")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("d(Power)/dt")
        ax.legend(loc='upper right')
        ax.grid(True)

    plt.suptitle("Time-derivatives of α and β Power (mean across trials)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



# — helper to plot one confusion matrix on a given Axes —
def plot_cm_ax(ax, y_true, y_pred, classes, title,confusion_matrix,accuracy_score):
    cm  = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred) * 100

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "custom_teal", ["#ffffff", "#3d9583"], N=256
    )

    im  = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}\nAcc = {acc:.1f}%")

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j],
                    ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")


def plot_mean_spectrograms_all_channels(data_arr, fs,
                                        channel_names=None,
                                        nperseg=None, noverlap=None,
                                        fmax=40, cmap='viridis'):
    """
    Plot the mean spectrogram (averaged across trials) for each channel.

    Parameters
    ----------
    data_arr : ndarray, shape (n_trials, n_channels, n_samples)
        Your EEG data.
    fs : float
        Sampling frequency in Hz.
    channel_names : list of str, optional
        Names for each channel (length == n_channels). Defaults to indices.
    nperseg : int, optional
        Window length for spectrogram (defaults to fs//2).
    noverlap : int, optional
        Overlap for spectrogram (defaults to nperseg*0.6).
    fmax : float, optional
        Maximum frequency to display.
    cmap : str, optional
        Colormap for the pcolormesh.
    """
    n_trials, n_ch, n_samps = data_arr.shape

    if nperseg is None:
        nperseg = fs // 2
    if noverlap is None:
        noverlap = int(nperseg * 0.6)
    if channel_names is None:
        channel_names = [f"Ch{ch}" for ch in range(n_ch)]

    # Compute mean spectrogram per channel
    mean_Sxxs = []
    for ch in range(n_ch):
        specs = []
        for tr in data_arr:
            f, t, Sxx = spectrogram(
                tr[ch], fs,
                window='hann', nperseg=nperseg,
                noverlap=noverlap, nfft=nperseg*2,
                scaling='density'
            )
            specs.append(Sxx)
        mean_Sxxs.append(np.mean(np.stack(specs), axis=0))

    # Mask frequencies up to fmax
    freq_mask = f <= fmax
    f_plot = f[freq_mask]

    # Prepare figure: 2 rows × 4 cols
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot each channel
    for ax, Sxx_mean, name in zip(axes, mean_Sxxs, channel_names):
        Sxxm = Sxx_mean[freq_mask, :]
        im = ax.pcolormesh(
            t, f_plot, 10 * np.log10(Sxxm),
            shading='gouraud', cmap=cmap
        )
        ax.set_title(name)
        ax.grid(True)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(f_plot[0], f_plot[-1])

    # Global colorbar
    cbar = fig.colorbar(im, ax=axes.tolist(), orientation='vertical',
                        fraction=0.02, pad=0.04)
    cbar.set_label('Power [dB]')

    # Shared axis labels
    for ax in axes[4:]:
        ax.set_xlabel("Time (s)")
    for ax in axes[::4]:
        ax.set_ylabel("Frequency (Hz)")

    plt.suptitle("Mean Spectrograms Across Trials — All Channels")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


from matplotlib.gridspec import GridSpec

def plot_overall_mean_spectrogram_with_envelopes(data_arr, fs,
                                                 window_sec=0.5,
                                                 overlap_frac=0.6,
                                                 nfft=512,
                                                 fmax=40,
                                                 cmap='viridis'):
    """
    Compute and plot the mean spectrogram across ALL channels and ALL trials,
    with a side colorbar and normalized alpha/beta envelopes beneath.

    Parameters
    ----------
    data_arr : ndarray, shape (n_trials, n_channels, n_samples)
        Your EEG data.
    fs : float
        Sampling rate [Hz].
    window_sec : float
        Length of each spectrogram window in seconds.
    overlap_frac : float
        Fractional overlap between windows (0 < overlap_frac < 1).
    nfft : int
        Number of FFT points.
    fmax : float
        Maximum frequency to display [Hz].
    cmap : str
        Matplotlib colormap name.
    """
    # 1) Spectrogram parameters
    nperseg = int(window_sec * fs)
    noverlap = int(nperseg * overlap_frac)

    # 2) Compute all single‐channel spectrograms and collect
    all_Sxx = []
    for trial in data_arr:
        for ch in range(trial.shape[0]):
            f, t, Sxx = spectrogram(
                trial[ch], fs,
                window='hann', nperseg=nperseg,
                noverlap=noverlap, nfft=nfft,
                scaling='density'
            )
            all_Sxx.append(Sxx)
    # Stack and average over trials × channels
    mean_Sxx = np.mean(np.stack(all_Sxx), axis=0)  # shape: (n_freqs, n_times)

    # 3) Frequency mask up to fmax
    freq_mask = f <= fmax
    f_plot = f[freq_mask]
    Sxx_plot = mean_Sxx[freq_mask, :]

    # 4) Compute alpha/beta envelopes
    alpha_mask = (f_plot >= 8) & (f_plot <= 14)
    beta_mask  = (f_plot >= 14) & (f_plot <= 40)
    alpha_ts = np.sum(Sxx_plot[alpha_mask, :], axis=0)
    beta_ts  = np.sum(Sxx_plot[beta_mask,  :], axis=0)
    alpha_env = alpha_ts / np.max(alpha_ts)
    beta_env  = beta_ts  / np.max(beta_ts)

    # 5) Layout: 2 rows × 2 cols, colorbar in right column spanning both rows
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, width_ratios=[20,1], height_ratios=[3,1],
                  wspace=0.3, hspace=0.3)
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_env  = fig.add_subplot(gs[1, 0], sharex=ax_spec)
    cax     = fig.add_subplot(gs[:, 1])

    # 6) Plot spectrogram
    im = ax_spec.pcolormesh(t, f_plot, 10*np.log10(Sxx_plot),
                             shading='gouraud', cmap=cmap)
    ax_spec.set_ylim(f_plot[0], f_plot[-1])
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title("Overall Mean Spectrogram (all channels & trials)")
    ax_spec.grid(True)

    # 7) Plot envelopes
    ax_env.plot(t, alpha_env, color='red',   label='α envelope')
    ax_env.plot(t, beta_env,  color='gray',  label='β envelope', alpha=0.7)
    ax_env.set_xlabel("Time (s)")
    ax_env.set_ylabel("Normalized power")
    ax_env.legend(loc='upper right')
    ax_env.grid(True)

    # 8) Colorbar (separate axis)
    fig.colorbar(im, cax=cax, label='Power (dB)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()