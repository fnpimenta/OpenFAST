import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 

from copy import deepcopy
from scipy.integrate import odeint
from scipy.signal import find_peaks
from scipy import signal

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST API'
icon = Image.open("logo.ico")
st.set_page_config(page_title=apptitle, page_icon=icon , layout="wide")

# Title the app
st.title('OpenFAST analysis')

st.markdown("""
 * Use the menu at left to select data from the different analysis possibilities
 * To tune the analysis parameters use the **Ahttp://localhost:8501/Homenalysis** tab
""")



with st.sidebar:
    st.title("Gallery")

    with st.expander("✨ File explorer", True):
        st.checkbox('a',value=False)

    with st.expander("🧩 Course Tasks", True):
        st.checkbox('b',value=False)
page.show()