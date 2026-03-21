
import matplotlib.pyplot as plt

import os
import numpy as np

tue_red = "#C71918"
red_compl = "#18C6C7"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.2,
})

def quick_plot(DO_PLOTS, t_ax, val, label, out_path=None, xlabel="Time [ms]", ylabel=None):
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=160)
    ax.plot(t_ax, val, label=label, color=tue_red, lw=1.1)

    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.legend()
    ax.minorticks_on()
    ax.tick_params(direction='in', which='both')

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=300)

    if DO_PLOTS:
        plt.show()

    plt.close()





def _to_numpy(x):
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


def _finite(x):
    x = _to_numpy(x)
    return x[np.isfinite(x)]


def _stats(x):
    x = _finite(x)
    if x.size == 0:
        return {
            "min": np.nan,
            "max": np.nan,
            "avg": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }

    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "avg": float(np.average(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def _gas_name(working_gas):
    if working_gas == 0:
        return "helium"
    if working_gas == 1:
        return "hydrogen"
    return f"unknown ({working_gas})"


def _fmt(v, precision=6):
    if v is None:
        return "None"
    try:
        if not np.isfinite(v):
            return "nan"
    except Exception:
        return str(v)

    av = abs(v)
    if av != 0 and (av >= 1e4 or av < 1e-3):
        return f"{v:.{precision}e}"
    return f"{v:.{precision}f}"


def _latex_value_unit(value, unit, precision=3):
    if value is None or not np.isfinite(value):
        return "nan"

    av = abs(value)
    if av != 0 and (av >= 1e4 or av < 1e-2):
        mantissa, exponent = f"{value:.{precision}e}".split("e")
        exponent = int(exponent)
        if unit:
            return rf"{mantissa} \times 10^{{{exponent}}}\,\mathrm{{{unit}}}"
        return rf"{mantissa} \times 10^{{{exponent}}}"
    else:
        if unit:
            return rf"{value:.{precision}f}\,\mathrm{{{unit}}}"
        return rf"{value:.{precision}f}"

def _write_summary_table(f, ip_stats, uloop_stats, te_stats, tau_stats):
    rows = [
        ("Ip [A]", ip_stats),
        ("Uloop [V]", uloop_stats),
        ("Te [eV]", te_stats),
        ("tau_E [us]", tau_stats),
    ]

    name_w = max(len(r[0]) for r in rows)
    val_w = 14

    f.write("==== SUMMARY TABLE ====\n")
    f.write(
        f"{'quantity':<{name_w}}  "
        f"{'min':>{val_w}}  "
        f"{'max':>{val_w}}  "
        f"{'mean':>{val_w}}  "
        f"{'avg':>{val_w}}\n"
    )
    f.write("-" * (name_w + 2 + 4 * (val_w + 2)) + "\n")

    for name, stats in rows:
        f.write(
            f"{name:<{name_w}}  "
            f"{_fmt(stats['min']):>{val_w}}  "
            f"{_fmt(stats['max']):>{val_w}}  "
            f"{_fmt(stats['mean']):>{val_w}}  "
            f"{_fmt(stats['avg']):>{val_w}}\n"
        )

    f.write("\n")


def write_timeconf_meta(
    out_dir,
    shot_dir,
    shot_num,
    working_gas,
    time_mask_padding,
    drift_correction,
    do_plots,
    major_radius,
    minor_radius,
    volume,
    plasma_valid,
    start_idx,
    end_idx,
    start_time,
    end_time,
    time,
    Ip,
    Uloop,
    Rp,
    Te,
    tau_E,
    n_e,
    file_name="meta.txt",
):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)

    time_stats = _stats(time)
    ip_stats = _stats(Ip)
    uloop_stats = _stats(Uloop)
    rp_stats = _stats(Rp)
    te_stats = _stats(Te)
    tau_stats = _stats(tau_E)

    gas_name = _gas_name(working_gas)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("==== SETTINGS ====\n")
        f.write(f"shot_num: {shot_num}\n")
        f.write(f"shot_dir: {shot_dir}\n")
        f.write(f"out_dir: {out_dir}\n")
        f.write(f"working_gas: {gas_name}\n")
        f.write(f"working_gas_code: {working_gas}\n")
        f.write(f"time_mask_padding: {_fmt(time_mask_padding)}\n")
        f.write(f"drift_correction: {drift_correction}\n")
        f.write(f"do_plots: {do_plots}\n")
        f.write("\n")

        f.write("==== DETECTED PLASMA WINDOW ====\n")
        f.write(f"valid_plasma: {plasma_valid}\n")
        f.write(f"start_index: {start_idx}\n")
        f.write(f"end_index: {end_idx}\n")
        f.write(f"start_time_ms: {_fmt(float(start_time))}\n")
        f.write(f"end_time_ms: {_fmt(float(end_time))}\n")
        f.write(f"window_duration_ms: {_fmt(float(end_time) - float(start_time))}\n")
        f.write("\n")

        f.write("==== CONSTANTS ====\n")
        f.write(f"major_radius_m: {_fmt(major_radius)}\n")
        f.write(f"minor_radius_m: {_fmt(minor_radius)}\n")
        f.write(f"volume_m3: {_fmt(volume)}\n")
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"density_m^-3: {_fmt(float(n_e))}\n")
        f.write("\n")

        _write_summary_table(f, ip_stats, uloop_stats, te_stats, tau_stats)

        f.write("-- time [ms] --\n")
        f.write("\n")

        f.write("-- time [ms] --\n")
        f.write(f"time_min_ms: {_fmt(time_stats['min'])}\n")
        f.write(f"time_max_ms: {_fmt(time_stats['max'])}\n")
        f.write(f"time_avg_ms: {_fmt(time_stats['avg'])}\n")
        f.write(f"time_mean_ms: {_fmt(time_stats['mean'])}\n")
        f.write(f"time_std_ms: {_fmt(time_stats['std'])}\n")
        f.write("\n")

        f.write("-- Ip [A] --\n")
        f.write(f"Ip_min_A: {_fmt(ip_stats['min'])}\n")
        f.write(f"Ip_max_A: {_fmt(ip_stats['max'])}\n")
        f.write(f"Ip_avg_A: {_fmt(ip_stats['avg'])}\n")
        f.write(f"Ip_mean_A: {_fmt(ip_stats['mean'])}\n")
        f.write(f"Ip_std_A: {_fmt(ip_stats['std'])}\n")
        f.write("\n")

        f.write("-- Uloop [V] --\n")
        f.write(f"Uloop_min_V: {_fmt(uloop_stats['min'])}\n")
        f.write(f"Uloop_max_V: {_fmt(uloop_stats['max'])}\n")
        f.write(f"Uloop_avg_V: {_fmt(uloop_stats['avg'])}\n")
        f.write(f"Uloop_mean_V: {_fmt(uloop_stats['mean'])}\n")
        f.write(f"Uloop_std_V: {_fmt(uloop_stats['std'])}\n")
        f.write("\n")

        f.write("-- Rp [ohm] --\n")
        f.write(f"Rp_min_ohm: {_fmt(rp_stats['min'])}\n")
        f.write(f"Rp_max_ohm: {_fmt(rp_stats['max'])}\n")
        f.write(f"Rp_avg_ohm: {_fmt(rp_stats['avg'])}\n")
        f.write(f"Rp_mean_ohm: {_fmt(rp_stats['mean'])}\n")
        f.write(f"Rp_std_ohm: {_fmt(rp_stats['std'])}\n")
        f.write("\n")

        f.write("-- Te [eV] --\n")
        f.write(f"Te_min_eV: {_fmt(te_stats['min'])}\n")
        f.write(f"Te_max_eV: {_fmt(te_stats['max'])}\n")
        f.write(f"Te_avg_eV: {_fmt(te_stats['avg'])}\n")
        f.write(f"Te_mean_eV: {_fmt(te_stats['mean'])}\n")
        f.write(f"Te_std_eV: {_fmt(te_stats['std'])}\n")
        f.write("\n")

        f.write("-- tau_E [us] --\n")
        f.write(f"tauE_min_us: {_fmt(tau_stats['min'])}\n")
        f.write(f"tauE_max_us: {_fmt(tau_stats['max'])}\n")
        f.write(f"tauE_avg_us: {_fmt(tau_stats['avg'])}\n")
        f.write(f"tauE_mean_us: {_fmt(tau_stats['mean'])}\n")
        f.write(f"tauE_std_us: {_fmt(tau_stats['std'])}\n")
        f.write("\n")

        f.write("==== LATEX ====\n")
        f.write(
            "density: "
            + rf"$n_e = {_latex_value_unit(float(n_e), 'm^{-3}')}$"
            + "\n"
        )
        f.write(
            "temperature: "
            + rf"$T_e = {_latex_value_unit(te_stats['mean'], 'eV')}$"
            + "\n"
        )
        mu_s = "\\mu s"
        f.write(
            "time_confinement: "
            + rf"$\tau_E = {_latex_value_unit(tau_stats['mean'], mu_s)}$"
            + "\n"
        )

        f.write("\n")
        f.write("latex_table:\n")

        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Quantity & Min & Max & Mean & Avg \\\\\n")
        f.write("\\hline\n")

        f.write(
            f"$I_p$ [A] & "
            f"{_fmt(ip_stats['min'])} & "
            f"{_fmt(ip_stats['max'])} & "
            f"{_fmt(ip_stats['mean'])} & "
            f"{_fmt(ip_stats['avg'])} \\\\\n"
        )

        f.write(
            f"$U_{{loop}}$ [V] & "
            f"{_fmt(uloop_stats['min'])} & "
            f"{_fmt(uloop_stats['max'])} & "
            f"{_fmt(uloop_stats['mean'])} & "
            f"{_fmt(uloop_stats['avg'])} \\\\\n"
        )

        f.write(
            f"$T_e$ [eV] & "
            f"{_fmt(te_stats['min'])} & "
            f"{_fmt(te_stats['max'])} & "
            f"{_fmt(te_stats['mean'])} & "
            f"{_fmt(te_stats['avg'])} \\\\\n"
        )

        f.write(
            f"$\\tau_E$ [$\\mu$s] & "
            f"{_fmt(tau_stats['min'])} & "
            f"{_fmt(tau_stats['max'])} & "
            f"{_fmt(tau_stats['mean'])} & "
            f"{_fmt(tau_stats['avg'])} \\\\\n"
        )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Summary of plasma parameters}\n")
        f.write("\\end{table}\n")

    return out_path