import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 

import mpld3
import streamlit.components.v1 as components

from copy import deepcopy
from scipy.integrate import odeint
from scipy.signal import find_peaks
from scipy import signal

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST API'
icon = Image.open("logo.ico")
st.set_page_config(page_title=apptitle, page_icon=icon)

# Title the app
st.title('OpenFAST analysis')

st.markdown("""
 * Use the menu at left to select data from the different analysis possibilities
 * To tune the analysis parameters use the **Analysis** tab
""")

st.sidebar.success("Select a demo above.")

analysis_types = ['Files explorer' , 'Modes estimator' , 'Aerodynamic properties' ]
sel_analysis = st.selectbox('Available analysis', analysis_types)

if sel_analysis == 'Files explorer':
	subprocess.run(["python", "FilesExplorer.py"])

