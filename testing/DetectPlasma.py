# # Plasma detection from plasma current signal
# 
# This is a quick plasma detection code from the plasma current signal.
# This procedure is run as soon as possible in order to provide the information on plasma existence to other routines.
# 
# (author: L. Lobko)

import numpy as np
import os
from scipy import signal as sigproc
from scipy import integrate
import matplotlib.pyplot as plt

shot_no = 50855

def update_db_current_shot(field_name, value):
    os.system('export PGPASSWORD=`cat /golem/production/psql_password`;psql -c "UPDATE operation.discharges SET '+field_name+'='+str(value)+'WHERE shot_no IN(SELECT max(shot_no) FROM operation.discharges)" -q -U golem golem_database')
    os.system('export PGPASSWORD=`cat /golem/production/psql_password`;psql -c "UPDATE diagnostics.basicdiagnostics SET '+field_name+'='+str(value)+'WHERE shot_no IN(SELECT max(shot_no) FROM diagnostics.basicdiagnostics)" -q -U golem golem_database')

os.makedirs('Results', exist_ok=True)

def save_scalar(phys_quant, value, format_str='%.3f'):
    with open("Results/"+phys_quant, 'w') as f:
        f.write(format_str % value)
    update_db_current_shot(phys_quant,value)

# ds = np.DataSource('/tmp')  # temporary storage for downloaded files
ds = np.lib.npyio.DataSource('/tmp')
data_rog_URL = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/PlasmaDetection/dIp_dt.csv'
data_ULoop_URL = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/PlasmaDetection/V_loop.csv'
t_cd_URL = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Operation/Discharge/t_cd_discharge_request'
K_RogowskiCoil_URL = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Production/Parameters/SystemParameters/K_RogowskiCoil'
L_chamber_URL = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Production/Parameters/SystemParameters/L_chamber'
R_chamber_URL = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Production/Parameters/SystemParameters/R_chamber'

# Load data

def load_data(shot_no, signal_URL):
    fname = ds.open(signal_URL.format(shot_no=shot_no)).name
    data = np.loadtxt(fname, delimiter=',')
    # data[:, 0] = data[:, 0] * 1e3 # from micros to ms
    return data

def load_param(shot_no, param):
    data = float(ds.open(param.format(shot_no=shot_no)).read())
    # data = data * 1e-3
    return data

U_rogcoil = load_data(shot_no, data_rog_URL)
U_Loop = load_data(shot_no, data_ULoop_URL)
t_cd = load_param(shot_no, t_cd_URL)
K_RogowskiCoil = load_param(shot_no, K_RogowskiCoil_URL)
L_chamber = load_param(shot_no, L_chamber_URL)
R_chamber = load_param(shot_no, R_chamber_URL)

U_rogcoil[:, 1] *= -1

# fig, ax = plt.subplots()
# ax.plot(U_Loop[:, 0], U_Loop[:, 1], 'b-')
# plt.show()

def offset_remove(data):
    x_size, y_size = data.shape
    data_for_offset = data[0:int(x_size/100)]
    offset = np.mean(data_for_offset[:, 1])
    data[:, 1] -= offset
    return data

U_rogcoil = offset_remove(U_rogcoil)
U_Loop = offset_remove(U_Loop)

# fig, ax = plt.subplots()
# ax.plot(U_Loop[:, 0], U_Loop[:, 1], 'b-')
# plt.show()

# Integrate signal

U_integrated = np.copy(U_rogcoil)
U_integrated[:, 1] = (integrate.cumtrapz(U_rogcoil[:, 1], U_rogcoil[:, 0], initial=0))
U_integrated[:, 1] *= K_RogowskiCoil

# fig, ax = plt.subplots()
# ax.plot(U_integrated[:, 0], U_integrated[:, 1], 'b-')
# plt.show()

# Calculate chamber current

# def dIch_dt(t, Ich):
#     return (U_l_func(t) - R_chamber * Ich) / L_chamber
# 
# dIch_dt = (U_Loop - R_chamber *)/ L_chamber

Ich = np.copy(U_Loop)
Ich[:, 1] = U_Loop[:, 1] / R_chamber

# fig, ax = plt.subplots()
# ax.plot(Ich[:, 0], Ich[:, 1], 'b-')
# plt.show()

Ip = np.copy(Ich)
Ip[:, 1] = U_integrated[:, 1] - Ich[:, 1]

# fig, ax = plt.subplots()
# ax.plot(Ip[:, 0], Ip[:, 1], 'b-')
# plt.show()

# Smooth signal

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

Ip[:, 1] = smooth(Ip[:, 1], 200)
Ip[:, 0] = Ip[:, 0] * 1e3

# fig, ax = plt.subplots()
# ax.plot(Ip[:, 0], Ip[:, 1], 'b-')
# plt.show()

def find_peaks(data):
    peaks_indexes, _ = sigproc.find_peaks(data[:, 1], prominence=1e-1)
    return np.vstack((data[peaks_indexes, 0], data[peaks_indexes, 1])).T

# Calculate plasma boundaries from signal derivation

def calc_plasma_boundaries(data, position):
    deriv = data.copy()
    deriv[:, 1] = np.gradient(data[:, 1])
    deriv[:, 1] = smooth(deriv[:, 1], 1000)
    if position == 'start':
        index = np.where(deriv[:, 1] >= np.max(deriv[:, 1])/5)
        deriv = deriv[index]
        return deriv[0, 0]
    else:
        deriv = np.abs(deriv)
        # max_time = np.max(deriv[:, 0])
        # deriv = deriv[deriv[:, 0] <= (max_time-0.5)]
        peaks = find_peaks(deriv)
        return (peaks[-1, 0]+0.5)
    



# Cut data before t_cd (before is no plasma)

Ip_plasma_check = Ip[Ip[:, 0] <= (t_cd/1000+5)]

Ip = Ip[Ip[:, 0] > (t_cd/1000)]

# fig, ax = plt.subplots()
# ax.plot(Ip_plasma_check[:, 0], Ip_plasma_check[:, 1], 'b-')
# plt.show()

if np.max(Ip_plasma_check[:, 1]) < 100:
    print('No plasma in vacuum chamber.')
    t_plasma_start = -1.0
    t_plasma_end = -1.0
else:
    t_plasma_start = calc_plasma_boundaries(Ip, 'start')
    print('Plasma starts at {:.2f} ms.'.format(t_plasma_start))
    t_plasma_end = calc_plasma_boundaries(Ip, 'end')
    print('Plasma ends at {:.2f} ms.'.format(t_plasma_end))

b_plasma = int(t_plasma_start > 0 and t_plasma_end > 0)

if b_plasma:
    t_plasma_duration = t_plasma_end - t_plasma_start
    print('Plasma duration is {:.2f} ms.'.format(t_plasma_duration))
else:
    t_plasma_duration = -1.0  # convention instead of nan

plasma_endpoints = [t_plasma_start, t_plasma_end]

fig, axes = plt.subplots()
axes.plot(Ip[:, 0], Ip[:, 1]/1000, label ='Ip calculated')
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--')
axes.set(xlabel='$time$ [ms]', ylabel='$I_p$ [kA]')
plt.legend()
plt.grid()
plt.show()

# Save data

save_scalar("b_plasma", b_plasma)
save_scalar("t_plasma_start", t_plasma_start)
save_scalar("t_plasma_end", t_plasma_end)
save_scalar("t_plasma_duration", t_plasma_duration)

# ==========================================
# Quasi-stationary (flat-top) plasma phase
#   - first: amplitude-based detection around |Ip|max
#   - then: trim start/end using dIp/dt to avoid ramp-up and disruption tail
# ==========================================

# Default values (no plasma / no flat-top)
t_plasma_qs_start = -1.0
t_plasma_qs_end = -1.0
t_plasma_qs_duration = -1.0

if b_plasma:
    # Use Ip only within detected plasma interval [t_plasma_start, t_plasma_end]
    t_all = Ip[:, 0]   # time [ms]
    Ip_all = Ip[:, 1]  # current [A]

    mask_window = (t_all >= t_plasma_start) & (t_all <= t_plasma_end)
    t_win = t_all[mask_window]
    Ip_win = Ip_all[mask_window]

    if len(t_win) > 5:
        Ip_abs = np.abs(Ip_win)
        Ip_max = Ip_abs.max()

        if Ip_max > 0:
            # Index (position) of maximum |Ip|
            idx_max = Ip_abs.argmax()

            # Plasma duration in this interval [ms]
            plasma_duration = t_plasma_end - t_plasma_start

            # Minimum desired flat-top duration (fraction of plasma duration)
            min_flat_fraction = 0.2          # e.g. 20 % of plasma duration
            min_flat_duration = min_flat_fraction * plasma_duration

            # First pass: amplitude-based detection
            threshold_levels = [0.9, 0.8, 0.7, 0.6, 0.5]

            t_qs_start = None
            t_qs_end = None
            longest_segment = None

            for level in threshold_levels:
                thr = level * Ip_max
                mask = Ip_abs >= thr

                indices = np.where(mask)[0]
                if len(indices) == 0:
                    continue

                # Split mask into continuous segments
                splits = np.where(np.diff(indices) != 1)[0] + 1
                groups = np.split(indices, splits)

                # Keep only segments that contain the Ip maximum
                candidate_segments = [
                    g for g in groups if (g[0] <= idx_max <= g[-1])
                ]
                if not candidate_segments:
                    continue

                # For this level, pick the longest segment containing Ip_max
                longest = max(candidate_segments, key=len)

                t_start_candidate = t_win[longest[0]]
                t_end_candidate   = t_win[longest[-1]]
                duration_candidate = t_end_candidate - t_start_candidate

                if duration_candidate >= min_flat_duration:
                    t_qs_start = t_start_candidate
                    t_qs_end   = t_end_candidate
                    longest_segment = longest
                    # Accept the first level that satisfies the duration condition
                    break

            # Fallback if no amplitude-based flat-top was found
            if t_qs_start is None or t_qs_end is None:
                n = len(t_win)
                i1 = int(0.33 * n)
                i2 = int(0.66 * n)
                t_qs_start = t_win[i1]
                t_qs_end   = t_win[i2]
                longest_segment = np.arange(i1, i2 + 1)

            # ------------------------------------------
            # Second pass: trim start/end using dIp/dt
            # ------------------------------------------
            # Compute derivative on the same window
            dIp_dt = np.gradient(Ip_win, t_win)  # [A/ms]

            seg_idx = longest_segment
            slopes_seg = np.abs(dIp_dt[seg_idx])

            # Typical "flat" slope level inside this segment (median)
            slope_med = np.median(slopes_seg)
            # Allow slopes up to a few times the median
            slope_thr = 3.0 * slope_med

            # Mark subsegment where slope is still "flat"
            flat_mask_seg = slopes_seg <= slope_thr
            flat_indices_seg = seg_idx[flat_mask_seg]

            # If trimming would completely kill the segment, keep original
            if len(flat_indices_seg) > 0:
                t_trim_start = t_win[flat_indices_seg[0]]
                t_trim_end   = t_win[flat_indices_seg[-1]]
                duration_trim = t_trim_end - t_trim_start

                # Require that trimmed interval is not too short
                if duration_trim >= 0.5 * min_flat_duration:
                    t_qs_start = t_trim_start
                    t_qs_end   = t_trim_end

            t_plasma_qs_start = float(t_qs_start)
            t_plasma_qs_end   = float(t_qs_end)
            t_plasma_qs_duration = t_plasma_qs_end - t_plasma_qs_start

print(
    "Quasi-stationary plasma interval: "
    f"{t_plasma_qs_start:.3f} → {t_plasma_qs_end:.3f} ms "
    f"(Δt = {t_plasma_qs_duration:.3f} ms)"
)

# Optional quick check plot (can be commented out in production)
fig, ax = plt.subplots()
ax.plot(Ip[:, 0], Ip[:, 1] / 1000, label='Ip')
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--')
for x in [t_plasma_qs_start, t_plasma_qs_end]:
    if x > 0:
        ax.axvline(x=x, color='red', linestyle='--', label='qs boundary')
ax.set_xlabel('time [ms]')
ax.set_ylabel('$I_p$ [kA]')
ax.grid()
plt.legend()
plt.show()
# Save quasi-stationary phase scalars
save_scalar("t_plasma_qs_start", t_plasma_qs_start)
save_scalar("t_plasma_qs_end", t_plasma_qs_end)
save_scalar("t_plasma_qs_duration", t_plasma_qs_duration)




