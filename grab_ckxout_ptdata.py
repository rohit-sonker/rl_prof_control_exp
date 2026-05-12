"""
Fetch CKXOUT profiles via toksearch_d3d PtDataSignal (same reshape/slices as grab_cakenn_data.py).

Processing matches real-time_selection.ipynb: radial interp 101→33 (uniform ψ_N), then low-pass
along time with dt=tau=20 ms on the 20 ms CKXOUT cadence.

Stores per-shot times; channels are 101→33 then LP-filtered. The safety-factor channel is stored
as inverse q (1/q) under the key ``q``, with |q| floored before division to avoid blow-ups.
"""

import os
import pickle

import numpy as np
from toksearch_d3d import PtDataSignal

# Same as real-time_selection.ipynb (dt matches resampled / CKXOUT grid)
DT_MS = 20.0
TAU_MS = 20.0
# Minimum |q| before inversion (1/q); avoids inf/nan after smoothing.
Q_INVERT_EPS = 1e-12

shot_list = np.arange(206584, 206597)

ckxout_signal = PtDataSignal("CKXOUT")

NYMODEL = 2
NYOUT_PTS = 101
NYPROF = 7


def filter_lp(x, prev_y, dt, tau):
    return prev_y + dt / (dt + tau) * (x - prev_y)


def apply_lp_filter(data, dt, tau, start_from_zero=True):
    """Low-pass along axis 0 (time); supports 1D or 2D (n_time, n_radial) arrays."""
    filtered = np.zeros_like(data, dtype=float)
    filtered[0] = 0.0 if start_from_zero else data[0]

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
    in_size = len(values_in)
    psi_in = np.linspace(0, 1, in_size)
    psi_out = np.linspace(0, 1, out_size)
    return interp_full(psi_in, values_in, psi_out)


def convert_profiles_101_to_33(profiles_101):
    original_shape = profiles_101.shape
    profiles_flat = profiles_101.reshape(-1, original_shape[-1])
    profiles_33 = np.array([interp_profile(prof, 33) for prof in profiles_flat])
    new_shape = original_shape[:-1] + (33,)
    return profiles_33.reshape(new_shape)


def safe_invert_q(q_profiles, eps=Q_INVERT_EPS):
    """1/q with |q| floored to eps so values stay finite (sign preserved)."""
    q = np.asarray(q_profiles, dtype=float)
    return np.sign(q) / np.maximum(np.abs(q), eps)


def process_ckxout_profiles(raw_dict):
    """101→33 then LP; q is inverted after smoothing and saved only as ``q``."""
    out = {}
    for key in ("density", "rotation", "pressure", "etemp", "itemp"):
        p33 = convert_profiles_101_to_33(raw_dict[key])
        out[key] = apply_lp_filter(p33, dt=DT_MS, tau=TAU_MS)
    q33 = convert_profiles_101_to_33(raw_dict["q"])
    q33_f = apply_lp_filter(q33, dt=DT_MS, tau=TAU_MS)
    out["q"] = safe_invert_q(q33_f)
    return out


all_shot_data = {}
for shotn in shot_list:
    print(f"Grabbing shot {shotn}")
    try:
        x = ckxout_signal.fetch(int(shotn))
        times = np.asarray(x["times"])
        data = np.asarray(x["data"])

        prof_data = data.reshape((-1, NYMODEL, NYOUT_PTS, NYPROF))

        raw = {
            "density": prof_data[:, 1, :, 3],
            "rotation": prof_data[:, 1, :, 6],
            "q": prof_data[:, 1, :, 2],
            "pressure": prof_data[:, 1, :, 0],
            "etemp": prof_data[:, 1, :, 4],
            "itemp": prof_data[:, 1, :, 5],
        }
        processed = process_ckxout_profiles(raw)

        all_shot_data[int(shotn)] = {
            "times": times,
            **processed,
        }
        print(f"  Successfully retrieved data for shot {shotn}")
    except Exception as e:
        print(f"  Error with shot {shotn}: {type(e).__name__}: {str(e)}")
        print(f"  Skipping shot {shotn}")
        continue

output_dir = "cakenn_data"
os.makedirs(output_dir, exist_ok=True)

out_path = os.path.join(output_dir, "ckxout_ptdata_profiles_mar26exp.pkl")
payload = {
    "meta": {
        "dt_ms": DT_MS,
        "tau_ms": TAU_MS,
        "radial_points_33_channels": 33,
        "q_invert_eps": Q_INVERT_EPS,
        "description": "CKXOUT via PtDataSignal. Profiles: 101→33 then LP (dt=tau=20 ms). "
        "Key q is inverse safety factor 1/q after smoothing (|q| floored before divide).",
    },
    "shots": all_shot_data,
}
with open(out_path, "wb") as f:
    pickle.dump(payload, f)

print(f"Wrote {len(all_shot_data)} shots to {out_path}")
