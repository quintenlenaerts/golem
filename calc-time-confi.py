import os, sys
import download_shot as ds
from shot_data import ShotData

import numpy as np
from scipy import constants


import matplotlib.pyplot as plt

from validate_plasma import validate_plasma, get_plasma_start_and_end_indices

from golem import *

MAJOR_RADIUS = 0.4
MINOR_RADIUS = 0.085
VOLUME = 2 * np.pi ** 2 * MAJOR_RADIUS * MINOR_RADIUS ** 2

# COMPUTATION OPTIONS
TIME_MASK_PADDING = 0.05
DO_PLOTS = True
# WORKING GAS OPTIONS
HELIUM_GAS = 0
HYDROGEN_GAS = 1



tue_red = "#C71918"
red_compl = "#18C6C7"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.2,
})



def calc_timeconf(shot_dir, out_dir="time_results", shot_num=""):
    os.makedirs(out_dir, exist_ok=True)
    shot = ShotData(shot_dir)

    # 1. check if plasma succesfull & extract workable range 
    p_valid, p_start, p_end = validate_plasma(shot)
    if not p_valid:
        gsad("Plasma not valid. Aborting.")
        return
    
    time = shot["U_loop.csv"].iloc[:,0]
    s_idx, e_idx = get_plasma_start_and_end_indices(p_start, p_end, time, TIME_MASK_PADDING)
    gprint(f"Time analysis found start/end indices : {s_idx}/{e_idx} and time values are {round(time[s_idx], 3)}/{round(time[e_idx],3)} ms. These values should match the above (minus padding). \n")
    time_mask = (np.arange(len(time)) >= s_idx) & (np.arange(len(time)) <= e_idx)
    time = time[time_mask]

    # 2. Defining / Getting variables
    Ip = (shot["Ip.csv"].iloc[time_mask,1] * 1e3)
    Uloop = (shot["U_loop.csv"].iloc[time_mask,1])

    # 3. calculating the R
    Rp = Uloop / Ip 
    quick_plot(time, Rp, "Rp [ohm]")
    quick_plot(time, Uloop, "Uloop [V]")

    # 4. Calculating temperatuur
    Te = 0.9 * Rp ** (-2/3)
    quick_plot(time, Te, "Temperature [ev]")
    gprint(f"Average/Mean plasma temperatue {round(np.average(Te),2)}/{round(np.mean(Te),2)} [eV]\n")

    # 5. Calculating density.
    n_e = calc_density(shot, HELIUM_GAS)
    gprint(f"Calculated density is {n_e}")

    # 6. Calculating time conf
    time_conf = constants.elementary_charge * n_e * Te * VOLUME / (3 * Uloop * Ip) *1e6
    quick_plot(time, time_conf, "Time confinement. [µs]")
    rounding = 2
    gprint(f"Time confinement min/max/avg/mean : {round(np.min(time_conf),rounding)}/{round(np.max(time_conf),rounding)}/{round(np.average(time_conf),rounding)}/{round(np.mean(time_conf),rounding)} [µs]")



def quick_plot(t_ax, val, label):
    if not DO_PLOTS:
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=160)
    ax.plot(t_ax, val, label=label, color=tue_red, lw=1.1)
    ax.legend()
    plt.show()
    plt.close()




def calc_time(shot_dir, out_dir="time_results"):
    os.makedirs(out_dir, exist_ok=True)

    shot = ShotData(shot_dir)
    L_p = calc_electron_inductance() + calc_ion_inductance()


    #
    #   TODO LIST : (TESSSA asgiend) 
    #

    # 1. Oke hier moet eerst worden gececked ofda dIdpt uberhaupt beschickbaar was
    #           => dit veranderd hoe we PH kunne/moete berekenen
    # 2. vervolgens moet de effective/workable range van het plasma bepaald worde door Ip
    #           overal waar deez klein / negatief is => weg 
    #           dan R berekenen
    # 3. time moet gekozen worden door valida_plasma lifetime niet door scuffed mask
    # 4. paper beginne scrhijven (liefst wa rap)
    # 5. HYDROGEN / HELIUM plasma type
    # 6. minor radius zou mss wel een bitteke 0.1 kunne zijn

    


    Ip = shot["Ip.csv"].iloc[:,1] * 1e3
    time = shot["Ip.csv"].iloc[:,0]
    # dIpdt = shot["dIpdt.csv"].iloc[:,1]
    Uloop = shot["U_loop.csv"].iloc[:,1]
    
    mask = Ip > 0.5 * np.max(Ip)

    Rp = Uloop[mask] / Ip[mask]
    Te = 0.9 * Rp ** (-2/3)

    time_valid = time[mask]
    print(time_valid)


    n_e = calc_density(shot, HELIUM_GAS)
    
    # om te zien waar het plasma leeft enzo
    validate_plasma(shot)

    time_conf = constants.elementary_charge * n_e * Te * VOLUME / (3 * Uloop[mask] * Ip[mask])


    # Rp = Uloop / Ip

    # print("\nRp")
    # print(Rp)

    # print(L_p)
    # BIG METHOD IF DIPDT IS GIVEN
    # P_mag = L_p * Ip * dIpdt
    # P_H = Uloop * Ip - P_mag
    # POOR PERSON METHODS
    # P_H = Uloop * Ip

    # T_e = 0.9 * Rp ** (-2/3)

    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=160)
    ax.plot(time, Ip, lw=1.1, color = tue_red, label="Ip")
    # ax.plot(time_valid, time_conf, lw=1.1, color = tue_red, label="energy conf")

    # ax.plot(time_valid, Te, lw=1.1, color = tue_red, label="temperature")
    # ax.plot(time, dIpdt*-1, lw=1.1, color = red_compl, label="dIp")
    # ax.plot(time, Uloop, lw=1.1, color = red_compl, label="Uloop")
    # ax.plot(time, Rp)
    # ax.plot(time, Uloop, lw=1)
    # ax.plot(time, Uloop, lw=1)
    ax.legend()
    plt.show()


    
    # print(calc_density(shot))


def calc_electron_inductance():
    return constants.mu_0 * MAJOR_RADIUS * np.log(8 * MAJOR_RADIUS / MINOR_RADIUS - 7/2)

def calc_ion_inductance():
    nu = 2 # the peaking factor : assumeing 1 for parabolic or 2 for peaked (more likely to accurately represnet thre reeal worls )
    norm_internal_inductance = np.log(1.65 + 0.89 * nu)
    return constants.mu_0 * MAJOR_RADIUS * norm_internal_inductance / 2 


def calc_density(shot, working_gas=HELIUM_GAS):
    #eerst dichtheid van working gas bepalen

    #std is Hydro
    k_a = 2
    k_e = 1
    if (working_gas == HELIUM_GAS):
        k_a = 2
        k_e = 2

    #volume van torus
    T0 = 300 # roomp temp temperatuur
    V0 = 60e-3
    # print(f"should be around the 600kubiekemterkes he {Vp}")

    p0 = shot["pre_dis_p"] * 1e-3

    N_e = k_a * k_e * (p0 * V0) / (T0 * constants.k)

    return N_e / VOLUME


def handle_shot_download():
    # standard shot is 51333
    num = 51333

    # shot num is first argu
    if (len(sys.argv) > 1):
        num = sys.argv[1]
    
    should_download_shot = True
    if (len(sys.argv)>2):
        should_download_shot = sys.argv[2] == "1"

    gprint(f"calculating time conf of shot num {num} \t downloading : {should_download_shot} ")

    if (should_download_shot):
        ds.download_shot(ds.TIME_CONFINEMENT_FILES, num)

    return num

def handle_should_plot_arg():
    global DO_PLOTS
    if len(sys.argv) > 3:
        DO_PLOTS = sys.argv[3] == "1"

if __name__ == "__main__":
    gstart_large()
    handle_should_plot_arg()

    num = handle_shot_download()
    
    calc_timeconf(f"shot_{num}", num)

    gend()
