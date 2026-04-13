import os, sys
import download_shot as ds
from shot_data import ShotData

import numpy as np
from scipy import constants


from validate_plasma import validate_plasma, get_plasma_start_and_end_indices

from golem import *
from export_meta import quick_plot, write_timeconf_meta

MAJOR_RADIUS = 0.4
MINOR_RADIUS = 0.1
# MINOR_RADIUS = 0.085
VOLUME = 2 * np.pi ** 2 * MAJOR_RADIUS * MINOR_RADIUS ** 2

# COMPUTATION OPTIONS
TIME_MASK_PADDING = 0.15
# TIME_MASK_PADDING = 0.2
DO_PLOTS = True
DRIFT_CORRECTION = True

# WORKING GAS OPTIONS
HELIUM_GAS = 0
HYDROGEN_GAS = 1

def calc_timeconf(shot_dir, out_dir="time_results"):
    os.makedirs(out_dir, exist_ok=True)
    shot = ShotData(shot_dir)

    # 1. check if plasma succesfull & extract workable range 
    plasma_dat = validate_plasma(shot)
    
    if not plasma_dat:
        gsad("Plasma not valid. Aborting.")
        return
    
    p_valid, p_start, p_end = plasma_dat
    
    time = shot["U_loop.csv"].iloc[:,0]
    
    # fullIp = (shot["Ip.csv"].iloc[:,1] )          # using calc ip
    fullIp = (shot["Ip.csv"].iloc[:,1] * 1e3)       # using std ip
    fullDrift = (fullIp[len(fullIp)-1] - fullIp[0])/(time[len(time)-1] - time[0]) * time  

    s_idx, e_idx = get_plasma_start_and_end_indices(p_start, p_end, time, TIME_MASK_PADDING)
    gprint(f"Time analysis found start/end indices : {s_idx}/{e_idx} and time values are {round(time[s_idx], 3)}/{round(time[e_idx],3)} ms. These values should match the above (minus padding). \n")
    time_mask = (np.arange(len(time)) >= s_idx) & (np.arange(len(time)) <= e_idx)
    time = time[time_mask]

    # 2. Defining / Getting variables
    # Ip = (shot["Ip.csv"].iloc[time_mask,1])         # FOR USING calculated IP
    Ip = (shot["Ip.csv"].iloc[time_mask,1] * 1e3) # FOR USING THE std IP
    Uloop = (shot["U_loop.csv"].iloc[time_mask,1])
    drift = fullDrift[time_mask]

    # Remove drift
    if DRIFT_CORRECTION:
        Ip = Ip - drift
    quick_plot(DO_PLOTS,time, Ip, "Ip", ylabel="Ip [A]", out_path=f"{out_dir}/Ip.png")

    # 3. calculating the R
    Rp = Uloop / Ip 
    quick_plot(DO_PLOTS,time, Rp, "Rp", ylabel="Rp [ohm]", out_path=f"{out_dir}/Rp.png")
    quick_plot(DO_PLOTS,time, Uloop, "U Loop", ylabel="U [V]", out_path=f"{out_dir}/Uloop.png")

    # 4. Calculating temperatuur
    Te = 0.9 * Rp ** (-2/3)
    # A = np.pi * MINOR_RADIUS**2
    # L = 2 * np.pi * MAJOR_RADIUS
    # lnLambda = 10 # Estimate
    # Z = 1 # Hydrogen
    # # if (working_gas == HELIUM_GAS):
    # #     Z = 2 # place holder value
        
    # eta = Rp * A / L

    # Te = (4 * np.sqrt(2 * np.pi)* Z* constants.elementary_charge**2* np.sqrt(constants.electron_mass)* lnLambda 
    # / ( 3*(4 * np.pi * constants.epsilon_0)**2* eta* constants.k**1.5))**(2/3)
    quick_plot(DO_PLOTS,time, Te, "Temperature", ylabel="Temperature [eV]", out_path=f"{out_dir}/T.png")
    gprint(f"Average/Mean plasma temperatue {round(np.average(Te),2)}/{round(np.mean(Te),2)} [eV]\n")

    # 5. Calculating density.
    n_e = calc_density(shot, HYDROGEN_GAS)
    gprint(f"Calculated density is {n_e}")

    # 6. Calculating time conf
    # P = Ip**2 * Rp
    # time_conf = constants.elementary_charge * n_e * Te * VOLUME / (3 * P) * 1e6
    time_conf = constants.elementary_charge * n_e * Te * VOLUME / (3 * Uloop * Ip) *1e6
    quick_plot(DO_PLOTS,time, time_conf, "Energy time confinement", ylabel="tau [µs]", out_path=f"{out_dir}/tau.png")
    rounding = 2
    gprint(f"Time confinement min/max/avg/mean : {round(np.min(time_conf),rounding)}/{round(np.max(time_conf),rounding)}/{round(np.average(time_conf),rounding)}/{round(np.mean(time_conf),rounding)} [µs]")


    triple = n_e * Te * time_conf * 1e-9
    quick_plot(DO_PLOTS,time, triple, "Triple", ylabel=" Triple product [eV s m^-3]", out_path=f"{out_dir}/triple.png")
    # print(f"tripple porducucut : : ::  {}")
    # exporting data
    write_timeconf_meta(
        out_dir=out_dir,
        shot_dir=shot_dir,
        shot_num=shot_dir.replace("shot_", ""),
        working_gas=HELIUM_GAS,
        time_mask_padding=TIME_MASK_PADDING,
        drift_correction=DRIFT_CORRECTION,
        do_plots=DO_PLOTS,
        major_radius=MAJOR_RADIUS,
        minor_radius=MINOR_RADIUS,
        volume=VOLUME,
        plasma_valid=p_valid,
        start_idx=s_idx,
        end_idx=e_idx,
        start_time=time.iloc[0] if hasattr(time, "iloc") else time[0],
        end_time=time.iloc[-1] if hasattr(time, "iloc") else time[-1],
        time=time,
        Ip=Ip,
        Uloop=Uloop,
        Rp=Rp,
        Te=Te,
        tau_E=time_conf,
        n_e=n_e,
    )


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

    # p0 = shot["pre_dis_p"] * 1e-3
    p0 = 9 * 1e-3
    if shot["pre_dis_p"] == None:
        print("SHOT has no pres ")
    else:
        p0 = shot["pre_dis_P"] * 1e-3
    

    gprint(f"used discharge is {p0} Pa")
    # p0 = 0.13 * 1e-3 
    # p0 = 10 * 1e-3


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

    calc_timeconf(f"shot_{num}", out_dir=f"time_results_{num}")

    gend()
