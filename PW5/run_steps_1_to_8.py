from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

VP = 6.2  # km/s
VS = 3.6  # km/s


def bandpass(x: np.ndarray, fs: float, a: float, b: float) -> np.ndarray:
    """Keep frequencies in [a, b] Hz using FFT, zeroing outside, and IFFT."""
    n = len(x)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    spec = np.fft.fft(x)
    spec[(np.abs(freqs) < a) | (np.abs(freqs) > b)] = 0.0
    return np.fft.ifft(spec).real


def envelope(x: np.ndarray, fs: float, win_s: float) -> np.ndarray:
    """Moving average of |x| over a window of win_s seconds."""
    w = max(1, int(round(win_s * fs)))
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(np.abs(x), kernel, mode="same")


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Look at the record (in mm/s)
    d = pd.read_csv("data/seismogram_A.csv")
    t = d["time_s"].to_numpy(dtype=float)
    z = d["Z"].to_numpy(dtype=float)
    n = d["N"].to_numpy(dtype=float)
    e = d["E"].to_numpy(dtype=float)
    fs = 1.0 / (t[1] - t[0])
    print(f"Samples N={len(t)}, fs={fs} Hz, duration={t[-1] - t[0]:.2f} s")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    for ax, ch, label in zip(axes, [z, n, e], ["Z (mm/s)", "N (mm/s)", "E (mm/s)"]):
        ax.plot(t, ch * 1000.0, linewidth=0.6)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Seismogram A Channels (mm/s)")
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_channels.png", dpi=180)
    plt.close()

    mask_60 = t <= 60.0
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for ax, ch, label in zip(axes, [z, n, e], ["Z (mm/s)", "N (mm/s)", "E (mm/s)"]):
        ax.plot(t[mask_60], ch[mask_60] * 1000.0, linewidth=0.7)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("First 60 s Zoom (Background Noise)")
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_zoom_60s.png", dpi=180)
    plt.close()

    # 2. Spectrum of Z
    N = len(z)
    Z_fft = np.fft.fft(z)
    amp_z = np.abs(Z_fft) / N * 2.0
    freqs = np.fft.fftfreq(N, d=1.0 / fs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(np.arange(N), amp_z, linewidth=0.6)
    ax1.set_xlabel("FFT Array Index")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Z Spectrum vs. Index")
    ax1.grid(True, alpha=0.3)

    ax2.plot(freqs, amp_z, linewidth=0.6)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Z Spectrum vs. fftfreq")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig2a_spectrum_index_vs_freq.png", dpi=180)
    plt.close()

    pos = freqs > 0
    z_quiet = z[mask_60]
    Nq = len(z_quiet)
    fq = np.fft.fftfreq(Nq, d=1.0 / fs)
    amp_q = np.abs(np.fft.fft(z_quiet)) / Nq * 2.0
    pos_q = fq > 0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(freqs[pos], amp_z[pos], alpha=0.75, label="Whole record")
    ax.loglog(fq[pos_q], amp_q[pos_q], alpha=0.85, label="Quiet first 60 s")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Log-Log Spectrum: Quiet Window vs. Whole Record (Z)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig2b_spectrum_loglog.png", dpi=180)
    plt.close()

    # 3. Inverse FFT and manual band-pass filter (0.5 - 10 Hz)
    reconstructed_z = np.fft.ifft(np.fft.fft(z)).real
    print("Round-trip FFT->IFFT close to Z:", np.allclose(reconstructed_z, z))

    z_bp = bandpass(z, fs, 0.5, 10.0)
    n_bp = bandpass(n, fs, 0.5, 10.0)
    e_bp = bandpass(e, fs, 0.5, 10.0)
    removed_z = z - z_bp

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, removed_z * 1000.0, linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Removed Z (mm/s)")
    ax.set_title("Removed Signal (z - bandpass(z, 0.5, 10 Hz))")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig3c_removed_z.png", dpi=180)
    plt.close()

    # 4. Envelope & window comparison
    env_z_05 = envelope(z_bp, fs, 0.5)
    env_z_005 = envelope(z_bp, fs, 0.05)
    env_z_5 = envelope(z_bp, fs, 5.0)
    h_bp = np.sqrt(n_bp**2 + e_bp**2)
    env_h_05 = envelope(h_bp, fs, 0.5)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, env_z_005 * 1000.0, alpha=0.7, linewidth=0.7, label="Window 0.05 s (noisy)")
    ax.plot(t, env_z_05 * 1000.0, linewidth=1.2, label="Window 0.5 s (balanced)")
    ax.plot(t, env_z_5 * 1000.0, linewidth=1.4, label="Window 5.0 s (over-smoothed)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Envelope of Filtered Z (mm/s)")
    ax.set_title("Envelope Window Comparison on Filtered Z")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig4_envelope_windows.png", dpi=180)
    plt.close()

    # 5. Arrival times (P and S) & peak comparison
    idx_p = int(np.argmax(env_z_05 > 0.05 * np.max(env_z_05)))
    t_p = float(t[idx_p])

    start_s_idx = int(np.searchsorted(t, t_p + 2.0))
    idx_s = start_s_idx + int(np.argmax(env_h_05[start_s_idx:] > 0.10 * np.max(env_h_05)))
    t_s = float(t[idx_s])

    peaks, _ = signal.find_peaks(env_h_05)
    t_peak_max = float(t[peaks[np.argmax(env_h_05[peaks])]]) if len(peaks) > 0 else float(t[np.argmax(env_h_05)])

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    for ax, ch_f, label in zip(axes, [z_bp, n_bp, e_bp], ["Filtered Z (mm/s)", "Filtered N (mm/s)", "Filtered E (mm/s)"]):
        ax.plot(t, ch_f * 1000.0, linewidth=0.6)
        ax.axvline(t_p, color="green", linestyle="--", label="t_P" if label.startswith("Filtered Z") else "")
        ax.axvline(t_s, color="red", linestyle="--", label="t_S" if label.startswith("Filtered Z") else "")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"Arrival Times: t_P = {t_p:.2f} s, t_S = {t_s:.2f} s")
    plt.tight_layout()
    plt.savefig(out_dir / "fig5_arrivals.png", dpi=180)
    plt.close()

    # 6. Distance R and origin time
    R = (t_s - t_p) / (1.0 / VS - 1.0 / VP)
    dR_1s = 1.0 / (1.0 / VS - 1.0 / VP)
    t_origin = t_p - R / VP
    print(f"t_P={t_p:.2f} s, t_S={t_s:.2f} s, t_peak_H={t_peak_max:.2f} s")
    print(f"R={R:.2f} km, +/-1s in t_S changes R by +/-{dR_1s:.2f} km, t_origin={t_origin:.2f} s")

    # 7. Spectrogram up to 10 Hz
    f_spec, ts_spec, Sxx = signal.spectrogram(z, fs=fs, nperseg=256)
    f_mask = f_spec <= 10.0
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pcm = ax.pcolormesh(ts_spec, f_spec[f_mask], 10.0 * np.log10(Sxx[f_mask, :] + 1e-30), shading="auto")
    ax.axvline(t_p, color="white", linestyle="--", linewidth=1.0, label="t_P")
    ax.axvline(t_s, color="yellow", linestyle="--", linewidth=1.0, label="t_S")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram of Raw Z (0 - 10 Hz)")
    fig.colorbar(pcm, ax=ax, label="Power (dB)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "fig7_spectrogram.png", dpi=180)
    plt.close()

    # 8. PGV (cm/s) and Magnitude M
    h_raw_cms = np.sqrt(n**2 + e**2) * 100.0
    pgv = float(np.max(h_raw_cms))
    mag = float((np.log10(pgv) + np.log10(R) + 1.0) / 0.55)
    print(f"PGV={pgv:.5f} cm/s, Magnitude M={mag:.3f}")


if __name__ == "__main__":
    main()