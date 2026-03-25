import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import MDSplus
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from toksearch_d3d import PtDataSignal

from gadata import gadata


NUM_TOTAL_TARGETS = 15
VALID_PROFILE_TYPES = {"rot", "dens", "pres", "etemp"}
ACTUATOR_SIGNALS = [
    "bmspinj30l",
    "bmspinj30r",
    "bmspinj33l",
    "bmspinj33r",
    "bmspinj15l",
    "bmspinj15r",
    "bmspinj21l",
    "bmspinj21r",
    "gasA",
    "echpwrc",
]
ON_AXIS_BEAMS = {"bmspinj30l", "bmspinj30r", "bmspinj33l", "bmspinj33r"}
OFF_AXIS_BEAMS = {"bmspinj15l", "bmspinj15r", "bmspinj21l", "bmspinj21r"}
DEFAULT_RADIAL_INDICES = [0, 5, 10, 15, 20, 25, 30]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a multi-shot PDF report with profile and actuator plots."
    )
    parser.add_argument(
        "--config",
        required=False,
        default="report_config_example.json",
        help="Path to a JSON config file describing the shots to include.",
    )
    parser.add_argument(
        "--output",
        required=False,
        default="profile_report.pdf",
        help="Path to the output PDF report.",
    )
    parser.add_argument(
        "--targets-dir",
        default=str(Path(os.environ["HOME"]) / "rl_prof_control_exp" / "final_lp_targets"),
        help="Directory containing T1.txt ... T15.txt target files.",
    )
    parser.add_argument(
        "--profile-tau",
        type=float,
        default=20.0,
        help="Low-pass filter tau used on profiles.",
    )
    parser.add_argument(
        "--actuator-tau",
        type=float,
        default=1500.0,
        help="Low-pass filter tau used on actuators.",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        metavar=("START_MS", "END_MS"),
        default=[0.0, 6000.0],
        help="Time limits for actuator plots.",
    )
    return parser.parse_args()


def filter_lp(x, prev_y, dt, tau):
    return prev_y + dt / (dt + tau) * (x - prev_y)


def apply_lp_filter(data, dt, tau):
    filtered = np.zeros_like(data)
    filtered[0] = data[0]
    for i in range(1, len(data)):
        filtered[i] = filter_lp(data[i], filtered[i - 1], dt, tau)
    return filtered


def interp_full(psi_in, values_in, psi_out):
    in_size = len(psi_in)
    out_size = len(psi_out)
    out = np.zeros(out_size)

    j_min = [0, 0]
    j_max = [0, 0]

    j_min[0] = np.argmin(psi_in)
    j_max[0] = np.argmax(psi_in)

    j_min[1] = j_max[0]
    j_max[1] = j_min[0]

    for j in range(in_size):
        if psi_in[j] > psi_in[j_min[0]] and psi_in[j] <= psi_in[j_min[1]]:
            j_min[1] = j
        if psi_in[j] < psi_in[j_max[0]] and psi_in[j] >= psi_in[j_max[1]]:
            j_max[1] = j

    for i in range(out_size):
        psi_t = psi_out[i]
        j_high = j_max[0]
        j_low = j_min[0]

        for j in range(in_size):
            if psi_in[j] > psi_t and psi_in[j] < psi_in[j_high]:
                j_high = j
            elif psi_in[j] < psi_t and psi_in[j] > psi_in[j_low]:
                j_low = j

        if j_high == j_max[0]:
            j_low = j_max[1]
        elif j_low == j_min[0]:
            j_high = j_min[1]

        out_high = values_in[j_high]
        out_low = values_in[j_low]
        psi_n_diff = psi_in[j_high] - psi_in[j_low]
        weight = (psi_t - psi_in[j_low]) / psi_n_diff if abs(psi_n_diff) >= 1e-5 else 0.0
        out[i] = out_low + weight * (out_high - out_low)

    return out


def interp_profile(values_in, out_size):
    psi_in = np.linspace(0, 1, len(values_in))
    psi_out = np.linspace(0, 1, out_size)
    return interp_full(psi_in, values_in, psi_out)


def convert_profiles_101_to_33(profiles_101):
    original_shape = profiles_101.shape
    profiles_flat = profiles_101.reshape(-1, original_shape[-1])
    profiles_33 = np.array([interp_profile(profile, 33) for profile in profiles_flat])
    return profiles_33.reshape(original_shape[:-1] + (33,))


def load_targets(targets_dir):
    base = Path(targets_dir)
    target_rows = {"rot": [], "dens": [], "etemp": [], "pres": []}

    for i in range(NUM_TOTAL_TARGETS):
        path = base / f"T{i + 1}.txt"
        data = np.loadtxt(path, skiprows=2)
        target_rows["rot"].append(data[0])
        target_rows["dens"].append(data[1])
        target_rows["etemp"].append(data[2])
        target_rows["pres"].append(data[3])

    return {name: np.stack(rows) for name, rows in target_rows.items()}


def validate_shot_config(config):
    required = {"shot", "prof_type", "targets_indxs", "target_set_times"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}")

    prof_type = config["prof_type"]
    if prof_type not in VALID_PROFILE_TYPES:
        raise ValueError(f"Invalid prof_type '{prof_type}'. Expected one of {sorted(VALID_PROFILE_TYPES)}.")

    targets_indxs = config["targets_indxs"]
    target_set_times = config["target_set_times"]
    if len(targets_indxs) != len(target_set_times) - 1:
        raise ValueError("targets_indxs must have length len(target_set_times) - 1.")
    if any(idx < 1 or idx > NUM_TOTAL_TARGETS for idx in targets_indxs):
        raise ValueError(f"targets_indxs must be between 1 and {NUM_TOTAL_TARGETS}.")
    if sorted(target_set_times) != list(target_set_times):
        raise ValueError("target_set_times must be sorted in ascending order.")


def load_report_config(config_path):
    config_path = Path(config_path)
    data = json.loads(config_path.read_text())
    if isinstance(data, dict) and "shots" in data:
        shared_radial_indices = data.get("radial_indices")
        shots = data["shots"]
    elif isinstance(data, list):
        shared_radial_indices = None
        shots = data
    else:
        raise ValueError("Config must be either a list of shot configs or an object with a 'shots' key.")

    for shot_config in shots:
        validate_shot_config(shot_config)
        if "radial_indices" not in shot_config and shared_radial_indices is not None:
            shot_config["radial_indices"] = shared_radial_indices
        if "radial_indices" not in shot_config:
            shot_config["radial_indices"] = list(DEFAULT_RADIAL_INDICES)

    return shots


def fetch_profile_signals(shot, prof_type, tau):
    fetched = PtDataSignal("CKXOUT").fetch(shot)
    times = fetched["times"]
    data = fetched["data"]

    ny_model = 2
    ny_out_pts = 101
    ny_prof = 7
    prof_data = data.reshape((-1, ny_model, ny_out_pts, ny_prof))

    profiles_by_type = {
        "dens": prof_data[:, 1, :, 3],
        "rot": prof_data[:, 1, :, 6],
        "pres": prof_data[:, 1, :, 0],
        "etemp": prof_data[:, 1, :, 4],
    }

    converted = {
        name: apply_lp_filter(convert_profiles_101_to_33(values), dt=20.0, tau=tau)
        for name, values in profiles_by_type.items()
    }

    return times, converted[prof_type]


def fetch_actuator_data(shot, tau):
    conn = MDSplus.Connection("atlas.gat.com")
    act_data = {}
    on_axis_pwr = None
    off_axis_pwr = None

    for signal in ACTUATOR_SIGNALS:
        data = gadata(signal, shot, connection=conn)
        x = np.asarray(data.xdata)
        y = np.asarray(data.zdata)

        x = x[::20]
        y = y[::20]
        y = apply_lp_filter(y, dt=20.0, tau=tau)

        act_data[signal] = y
        act_data[f"{signal}_time"] = x

        if signal in ON_AXIS_BEAMS:
            if on_axis_pwr is None:
                on_axis_pwr = y.copy()
                act_data["on_axis_pwr_time"] = x
            else:
                on_axis_pwr += y
        elif signal in OFF_AXIS_BEAMS:
            if off_axis_pwr is None:
                off_axis_pwr = y.copy()
                act_data["off_axis_pwr_time"] = x
            else:
                off_axis_pwr += y

        act_data["on_axis_pwr"] = on_axis_pwr
        act_data["off_axis_pwr"] = off_axis_pwr

    return act_data


def build_target_trace(times, set_time_indices, targets, selected_target_indices, radial_index):
    x_times = times[set_time_indices[0]:set_time_indices[-1]]
    y_target = np.empty(x_times.size, dtype=float)
    start_idx = 0

    for j, target_idx in enumerate(selected_target_indices):
        segment_end = set_time_indices[j + 1] - set_time_indices[0]
        y_target[start_idx:segment_end] = targets[target_idx, radial_index]
        start_idx = segment_end

    return x_times, y_target


def plot_profile_page(config, targets, times, profile_signals, profile_tau):
    shot = config["shot"]
    prof_type = config["prof_type"]
    selected_target_indices = [idx - 1 for idx in config["targets_indxs"]]
    target_set_times = config["target_set_times"]
    radial_indices = config["radial_indices"]
    set_time_indices = [int(np.argmin(np.abs(times - t))) for t in target_set_times]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=False, sharey=False)
    axes = axes.ravel()

    for i, radial_index in enumerate(radial_indices):
        ax = axes[i]
        x_times, target_trace = build_target_trace(
            times=times,
            set_time_indices=set_time_indices,
            targets=targets,
            selected_target_indices=selected_target_indices,
            radial_index=radial_index,
        )
        signal_trace = profile_signals[set_time_indices[0]:set_time_indices[-1], radial_index]

        ax.plot(x_times, signal_trace, label="signal")
        ax.plot(x_times, target_trace, linestyle="--", color="r", label="target")
        for set_time in target_set_times:
            ax.axvline(set_time, color="0.8", linewidth=0.8)
        ax.set_title(f"rad idx {radial_index}", fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for i in range(len(radial_indices), len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Profile: {prof_type} | Shot: {shot} | Tau: {profile_tau:g}", fontsize=16)
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    return fig


def plot_actuator_page(config, act_data, actuator_tau, xlim):
    shot = config["shot"]
    target_set_times = config["target_set_times"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for i, signal in enumerate(["on_axis_pwr", "off_axis_pwr", "gasA", "echpwrc"]):
        ax = axes[i]
        x = act_data.get(f"{signal}_time")
        y = act_data.get(signal)
        if x is None or y is None:
            ax.set_title(f"{signal} unavailable")
            ax.axis("off")
            continue

        ax.plot(x, y, label=signal)
        for set_time in target_set_times:
            ax.axvline(set_time, color="0.8", linewidth=0.8)
        ax.set_title(signal)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Time (ms)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f"Actuator Signals | Shot: {shot} | Tau: {actuator_tau:g}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def render_report(shots, targets_by_type, output_path, profile_tau, actuator_tau, xlim):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for config in shots:
            shot = config["shot"]
            prof_type = config["prof_type"]
            print(f"Processing shot {shot} ({prof_type})")

            times, profile_signals = fetch_profile_signals(shot=shot, prof_type=prof_type, tau=profile_tau)
            profile_fig = plot_profile_page(
                config=config,
                targets=targets_by_type[prof_type],
                times=times,
                profile_signals=profile_signals,
                profile_tau=profile_tau,
            )
            pdf.savefig(profile_fig)
            plt.close(profile_fig)

            act_data = fetch_actuator_data(shot=shot, tau=actuator_tau)
            actuator_fig = plot_actuator_page(
                config=config,
                act_data=act_data,
                actuator_tau=actuator_tau,
                xlim=xlim,
            )
            pdf.savefig(actuator_fig)
            plt.close(actuator_fig)


def main():
    args = parse_args()
    shots = load_report_config(args.config)
    targets_by_type = load_targets(args.targets_dir)
    render_report(
        shots=shots,
        targets_by_type=targets_by_type,
        output_path=args.output,
        profile_tau=args.profile_tau,
        actuator_tau=args.actuator_tau,
        xlim=tuple(args.xlim),
    )
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
