import os
import sys
import requests

BASE_URL = "http://golem.fjfi.cvut.cz/shots/{shot_no}/"



# idk deze heb ik uit http://golem.fjfi.cvut.cz/shots/0/Diagnostics/PlasmaDetection/DetectPlasma.html gehaald
# ze gebruiken deze parameters om de "Quasi statiionary" plasma interval te berekne.? 
STD_FILES = {
    "dIp_dt.csv": "Diagnostics/PlasmaDetection/dIp_dt.csv",
    "V_loop.csv": "Diagnostics/PlasmaDetection/V_loop.csv",
    "t_cd_discharge_request": "Operation/Discharge/t_cd_discharge_request",
    "K_RogowskiCoil": "Production/Parameters/SystemParameters/K_RogowskiCoil",
    "L_chamber": "Production/Parameters/SystemParameters/L_chamber",
    "R_chamber": "Production/Parameters/SystemParameters/R_chamber"
}

THINGS = {
    "t_plasma_start": "Diagnostics/PlasmaDetection/Results/t_plasma_start",
    "t_plasma_end": "Diagnostics/PlasmaDetection/Results/t_plasma_end",
    "t_plasma_duration": "Diagnostics/PlasmaDetection/Results/t_plasma_duration",
    "Ip.csv" : "Diagnostics/BasicDiagnostics/Basic/Results/Bt.csv"
}



# http://golem.fjfi.cvut.cz/shots/0/Diagnostics/BasicDiagnostics/analysis.html
# allemaal bestandjes nodig voor basic diagnotistics i g
BASIC_DIAG_FiLES = {
    "t_plasma_start": "Diagnostics/PlasmaDetection/Results/t_plasma_start",
    "t_plasma_end": "Diagnostics/PlasmaDetection/Results/t_plasma_end",
    "t_plasma_duration": "Diagnostics/PlasmaDetection/Results/t_plasma_duration"
}


PLASMA_CURRENT_FILES = {
    # needed for checking if the plasma is actually alive
    "b_plasma": "Diagnostics/PlasmaDetection/Results/b_plasma",
    "t_plasma_start": "Diagnostics/PlasmaDetection/Results/t_plasma_start",
    "t_plasma_end": "Diagnostics/PlasmaDetection/Results/t_plasma_end",
    "t_plasma_duration": "Diagnostics/PlasmaDetection/Results/t_plasma_duration",

    "t_Bt" : "Production/Parameters/TBt",
    "t_CD" : "Production/Parameters/Tcd",

    # voltage looop stuff       
    "V_loop.csv" : "Diagnostics/PlasmaDetection/V_loop.csv",
    "dBt_dt.csv" : "Diagnostics/PlasmaDetection/dBt_dt.csv",
    "CD_orientation" : "Production/Parameters/CD_orientation", # either CW or anticlock wise
    "Bt_orientation" : "Production/Parameters/Bt_orientation",


    "dIp_dt.csv": "Diagnostics/PlasmaDetection/dIp_dt.csv",


    # deze twee hebben we nodig als constants voor chamber current
    "R_chamber" : "Production/Parameters/SystemParameters/R_chamber",
    "L_chamber" : "Production/Parameters/SystemParameters/L_chamber"

    # #basic diagnostics
    # "Ip.csv" : "Diagnostics/BasicDiagnostics/Results/Ip.csv",
    # "Ip.csv" : "Diagnostics/BasicDiagnostics/Results/Ip.csv"
    
}

TIME_CONFINEMENT_FILES = {
    
    
    "Ip.csv" : "Diagnostics/BasicDiagnostics/Results/Ip.csv",
    "Ich.csv" : "Diagnostics/BasicDiagnostics/Results/Ich.csv",
    "U_loop.csv" : "Diagnostics/BasicDiagnostics/Results/U_loop.csv",
    "dIpdt.csv" : "Diagnostics/PlasmaDetection/dIp_dt.csv",

    # doing cools 
    "R_chamber" : "Production/Parameters/SystemParameters/R_chamber",
    "L_chamber" : "Production/Parameters/SystemParameters/L_chamber",


    # chamber params
    "pre_dis_p" : "Operation/Discharge/p_chamber_pressure_predischarge"

}




def download_file(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {save_path}")
    except requests.RequestException as e:
        print(f"Failed to download {url}")
        print(e)


def download_shot(files, shot_no):
    shot_dir = f"shot_{shot_no}"
    os.makedirs(shot_dir, exist_ok=True)

    for filename, relative_path in files.items():
        url = BASE_URL.format(shot_no=shot_no) + relative_path
        save_path = os.path.join(shot_dir, filename)
        download_file(url, save_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_shot.py <shot_number> <FILE type joenge =STD_FILES>")
        print("file types zijn : 'STD', 'BASIC_DIAG'")
        sys.exit(1)

    files_to_use = STD_FILES
    if len(sys.argv) > 2:
        print(sys.argv[2])
        if (sys.argv[2] == "BASIC_DIAG"):
            files_to_use = BASIC_DIAG_FiLES
        if (sys.argv[2] == "STD"):
            files_to_use = STD_FILES
    

    shot_number = sys.argv[1]
    download_shot(STD_FILES, shot_number )

# if __name__ == "__main__":
#     st = 51333

#     for i in range(2000):
#         shot_no = i + st
#         download_shot(THINGS, shot_no)

#         shot_dir = f"shot_{shot_no}"
#         if os.path.isdir(shot_dir) and not os.listdir(shot_dir):
#             os.rmdir(shot_dir)