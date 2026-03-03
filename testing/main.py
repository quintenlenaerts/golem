import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, signal, interpolate, constants
import holoviews as hv
hv.extension('bokeh')
import hvplot.pandas
import requests

# 180221 Dirigent
destination='BigThings/'
os.makedirs(destination, exist_ok=True)