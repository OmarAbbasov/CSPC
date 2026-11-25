from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal

VP = 6.2  # km/s
VS = 3.6  # km/s


def bandpass(x: np.ndarray, fs: float, a: float, b: float) -> np.ndarray:
    n = len(x)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    spec = np.fft.fft(x)
    spec[(np.abs(freqs) < a) | (np.abs(freqs) > b)] = 0.0
    return np.fft.ifft(spec).real


def envelope(x: np.ndarray, fs: float, win_s: float) -> np.ndarray:
    w = max(1, int(round(win_s * fs)))
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(np.abs(x), kernel, mode="same")


def analyse(csv_file: str | Path) -> dict:
    path = Path(csv_file)
    df = pd.read_csv(path)
    t = df["time_s"].to_numpy(dtype=float)
    z = df["Z"].to_numpy(dtype=float)
    n = df["N"].to_numpy(dtype=float)
    e = df["E"].to_numpy(dtype=float)
    fs = 1.0 / (t[1] - t[0])

    # Step 3: Band-pass filter (0.5 - 10 Hz)
    z_f = bandpass(z, fs, 0.5, 10.0)
    n_f = bandpass(n, fs, 0.5, 10.0)
    e_f = bandpass(e, fs, 0.5, 10.0)
    h_f = np.sqrt(n_f**2 + e_f**2)

    # Step 4: Envelope (0.5 s window)
    env_z = envelope(z_f, fs, 0.5)
    env_h = envelope(h_f, fs, 0.5)

    # Step 5: P and S arrival times
    idx_p = int(np.argmax(env_z > 0.05 * np.max(env_z)))
    t_p = float(t[idx_p])

    start_s_idx = int(np.searchsorted(t, t_p + 2.0))
    idx_s = start_s_idx + int(np.argmax(env_h[start_s_idx:] > 0.10 * np.max(env_h)))
    t_s = float(t[idx_s])

    # Step 6: Distance R (km) and origin time
    R = float((t_s - t_p) / (1.0 / VS - 1.0 / VP))
    t_origin = float(t_p - R / VP)

    # Step 7: Spectrogram computation
    _f_spec, _ts_spec, _Sxx = signal.spectrogram(z, fs=fs, nperseg=256)

    # Step 8: PGV (cm/s) and Magnitude M
    pgv = float(np.max(np.sqrt(n**2 + e**2)) * 100.0)
    magnitude = float((np.log10(pgv) + np.log10(R) + 1.0) / 0.55)

    return {
        "event": path.stem,
        "t_p_s": t_p,
        "t_s_s": t_s,
        "R_km": R,
        "t_origin_s": t_origin,
        "pgv_cm_s": pgv,
        "magnitude": magnitude,
    }


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        analyse("data/seismogram_A.csv"),
        analyse("data/seismogram_B.csv"),
    ]
    events_df = pd.DataFrame(rows)
    events_df.to_csv(out_dir / "events.csv", index=False)
    print(events_df.to_string(index=False))


if __name__ == "__main__":
    main()