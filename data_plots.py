import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from preprocessing import binary_to_direction
from scipy.signal import spectrogram
import matplotlib as mpl
from matplotlib.gridspec import GridSpec


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

def plot_overall_mean_spectrogram_with_envelopes(data_arr, fs,
                                                 window_sec=0.5,
                                                 overlap_frac=0.6,
                                                 nfft=512,
                                                 fmax=40,
                                                 cmap='viridis'):
    """
    Compute and plot the mean spectrogram across ALL channels and ALL trials,
    with a side colorbar and normalized alpha/beta envelopes beneath.
    """

    nperseg = int(window_sec * fs)
    noverlap = int(nperseg * overlap_frac)

    all_Sxx = []
    for trial in data_arr:
        for ch in range(trial.shape[0]):
            f, t, Sxx = spectrogram(
                trial[ch], fs,
                window='hann', nperseg=nperseg,
                noverlap=noverlap, nfft=nfft,
                scaling='density'
            )
            all_Sxx.append(10 * np.log10(Sxx + 1e-20))
    mean_Sxx = np.mean(np.stack(all_Sxx), axis=0)


    freq_mask = f <= fmax
    f_plot = f[freq_mask]
    Sxx_plot = mean_Sxx[freq_mask, :]

    alpha_mask = (f_plot >= 8) & (f_plot <= 14)
    beta_mask  = (f_plot >= 14) & (f_plot <= 40)
    alpha_ts = np.sum(Sxx_plot[alpha_mask, :], axis=0)
    beta_ts  = np.sum(Sxx_plot[beta_mask,  :], axis=0)
    alpha_env = alpha_ts / np.max(alpha_ts)
    beta_env  = beta_ts  / np.max(beta_ts)

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, width_ratios=[20,1,3], height_ratios=[3,1],
                  wspace=0.3, hspace=0.3)
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_env  = fig.add_subplot(gs[1, 0], sharex=ax_spec)
    cax     = fig.add_subplot(gs[:, 1])
    ax_legend = fig.add_subplot(gs[1, 2])
    ax_legend.axis('off')  # just use for legend

    im = ax_spec.pcolormesh(t, f_plot, Sxx_plot,
                             shading='gouraud', cmap=cmap)
    ax_spec.set_ylim(f_plot[0], f_plot[-1])
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title("Overall Mean Spectrogram (all channels & trials)")
    ax_spec.grid(True)

    # Plot envelopes
    line1, = ax_env.plot(t, alpha_env, color='red',   label='α envelope')
    line2, = ax_env.plot(t, beta_env,  color='gray',  label='β envelope', alpha=0.7)
    ax_env.set_xlabel("Time (s)")
    ax_env.set_ylabel("Normalized power")
    ax_env.grid(True)

    ax_env.legend(handles=[line1, line2],
                loc='upper center',
                bbox_to_anchor=(0.5, -0.25),
                ncol=2,
                fontsize=10,
                frameon=False)

    fig.colorbar(im, cax=cax, label='Power (dB)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return Sxx_plot, f_plot, t

def diff_mean_spectrogram(data_arr, fs,
                          window_sec=0.5, overlap_frac=0.6,
                          nfft=512, fmax=40, cmap='RdBu_r'):
    """
    Compute and plot the time‐difference of the mean spectrogram
    across all channels & trials, i.e. np.diff(mean_Sxx, axis=1).
    Overlays linear‐difference envelopes in alpha and beta bands.

    Parameters
    ----------
    data_arr : ndarray, shape (n_trials, n_channels, n_samples)
    fs       : float, sampling rate [Hz]
    window_sec : float, spectrogram window length [s]
    overlap_frac : float, fraction overlap between windows
    nfft     : int, FFT length
    fmax     : float, max frequency to display [Hz]
    cmap     : str, diverging colormap for difference
    """
    # 1) Spectrogram params
    nperseg  = int(window_sec * fs)
    noverlap = int(nperseg * overlap_frac)

    # 2) Collect all linear‐power spectrograms
    all_S = []
    for trial in data_arr:
        for ch in range(trial.shape[0]):
            f, t, Sxx = spectrogram(
                trial[ch], fs,
                window='hann', nperseg=nperseg,
                noverlap=noverlap, nfft=nfft,
                scaling='density'
            )
            all_S.append(Sxx)
    mean_Sxx = np.mean(np.stack(all_S), axis=0)  # (n_freqs, n_times)

    # 3) Mask frequencies up to fmax
    freq_mask = (f <= fmax)
    f_plot    = f[freq_mask]
    Sxx_plot  = mean_Sxx[freq_mask, :]

    # 4) Compute time‐differences
    dS = np.diff(Sxx_plot, axis=1)      # shape (n_freqs, n_times-1)
    t_diff = t[1:]

    # 5) Compute linear envelopes of the band‐differences
    α_mask = (f_plot >= 8)  & (f_plot <= 14)
    β_mask = (f_plot >= 14) & (f_plot <= 40)

    α_env = np.sum(dS[α_mask, :], axis=0)
    β_env = np.sum(dS[β_mask, :], axis=0)

    # 6) Plotting layout
    fig = plt.figure(figsize=(12, 6))
    gs  = GridSpec(2, 2, height_ratios=[4,1], hspace=0.3, wspace=0.3)

    ax_spec = fig.add_subplot(gs[0, 0])
    ax_env  = fig.add_subplot(gs[1, 0], sharex=ax_spec)
    ax_cb   = fig.add_subplot(gs[:, 1])

    # 7) Spectrogram‐difference plot
    im = ax_spec.pcolormesh(t_diff, f_plot, dS,
                            shading='gouraud', cmap=cmap,
                            vmin=-np.max(np.abs(dS)),
                            vmax= np.max(np.abs(dS)))
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title("Δ Mean Spectrogram (linear power diff)")
    ax_spec.grid(True)

    # 8) Envelope plot (linear sums, no dB)
    line1, = ax_env.plot(t_diff, α_env, color='C0', label='Δ α-band')
    line2, = ax_env.plot(t_diff, β_env, color='C1', label='Δ β-band')
    ax_env.set_xlabel("Time (s)")
    ax_env.set_ylabel("Sum Δ power")
    ax_env.grid(True)

    # 9) Legend below envelope
    ax_env.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
                  ncol=2, frameon=False)

    # 10) Colorbar
    fig.colorbar(im, cax=ax_cb, label='Δ Power (linear units)')

    plt.tight_layout()
    plt.show()

    # Return the difference matrix & axes for further use
    return dS, f_plot, t_diff

