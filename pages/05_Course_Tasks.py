import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from copy import deepcopy
from scipy.integrate import odeint
from scipy.signal import find_peaks
from scipy import signal

from streamlit_ace import st_ace, KEYBINDINGS, LANGUAGES, THEMES

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST API - Course tasks'
icon = Image.open("logo.ico")
st.set_page_config(page_title=apptitle, page_icon=icon)

# Title the app
st.title('OpenFAST analysis')
