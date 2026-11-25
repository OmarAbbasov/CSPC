from pathlib import Path
import pandas as pd


def main() -> None:
    events_path = Path("results/events.csv")
    report_path = Path("results/report.txt")
    df = pd.read_csv(events_path)

    closest_row = df.loc[df["R_km"].idxmin()]
    strongest_row = df.loc[df["magnitude"].idxmax()]

    lines = ["Earthquake Summary Report", "=" * 26]
    for _, row in df.iterrows():
        lines.append(
            f"{row['event']}: t_P = {row['t_p_s']:.2f} s, t_S = {row['t_s_s']:.2f} s, "
            f"R = {row['R_km']:.2f} km, origin = {row['t_origin_s']:.2f} s, "
            f"PGV = {row['pgv_cm_s']:.4f} cm/s, M = {row['magnitude']:.2f}"
        )

    lines.append("")
    lines.append(f"Closer earthquake: {closest_row['event']} (R = {closest_row['R_km']:.2f} km)")
    lines.append(f"Stronger earthquake: {strongest_row['event']} (M = {strongest_row['magnitude']:.2f})")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()