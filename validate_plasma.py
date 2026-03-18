from shot_data import ShotData
import numpy as np

def validate_plasma(shot : ShotData):
    """
        Are gonna make sure that the plasma is alive and well ie a lifetime of more then 0s.
    """

    if (shot["b_plasma"] != 1):
        return False

    t_plasma_start = shot["t_plasma_start"]
    t_plasma_end = shot["t_plasma_end"]
    plasma_lifetime = shot["t_plasma_duration"]
    print(f"[^-^] plasma lifetime of {plasma_lifetime:.1f} ms, from {t_plasma_start:.1f} ms to {t_plasma_end:.1f} ms")

    return True, t_plasma_start, t_plasma_end


def get_plasma_start_and_end_indices(plasma_start,plasma_end, time_arr, padding=0.1):
    """
        Given values shuol be floats or else.

        Time array should be from the shot.

        Padding is in percent, this adds the value of the index ie 4102 -> 4101*1.1 is new start
        ending gets subtracted so 15000 -> 15000*.9 is new ending index
    """
    time_arr = np.asarray(time_arr)

    # find indices of closest values
    start_index = np.argmin(np.abs(time_arr - plasma_start))
    end_index   = np.argmin(np.abs(time_arr - plasma_end))

    # apply padding
    start_index = int(start_index * (1 + padding))
    end_index   = int(end_index * (1 - padding))

    return start_index, end_index



