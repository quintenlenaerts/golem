import os, sys
import download_shot as ds
from shot_data import ShotData

import numpy as np
from scipy import constants


import matplotlib.pyplot as plt

MAJOR_RADIUS = 0.4
MINOR_RADIUS = 0.085
VOLUME = 2 * np.pi ** 2 * MAJOR_RADIUS * MINOR_RADIUS ** 2

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

def calc_time(shot_dir, out_dir="time_results", shot_no=51333):
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
    


    Ip = shot["Ip.csv"].iloc[:,1] * 1e3
    time = shot["Ip.csv"].iloc[:,0]
    # dIpdt = shot["dIpdt.csv"].iloc[:,1]
    Uloop = shot["U_loop.csv"].iloc[:,1]
    
    mask = Ip > 0.5 * np.max(Ip)

    Rp = Uloop[mask] / Ip[mask]
    Te = 0.9 * Rp ** (-2/3)

    time_valid = time[mask]


    n_e = calc_density(shot)
    

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
    # ax.plot(time, Ip, lw=1.1, color = tue_red, label="Ip")
    ax.plot(time_valid, time_conf, lw=1.1, color = tue_red, label="energy conf")
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


if __name__ == "__main__":
    num = 51333 #ieuw
    # num = 48251 # guce 
    # num = 41881
    
    # num = 44824 # guci
    # num = 44805 # tres bien
    # num = 44779
    # num = 44805


    ds.download_shot(ds.TIME_CONFINEMENT_FILES, num)
    calc_time(f"shot_{num}", shot_no=num)
