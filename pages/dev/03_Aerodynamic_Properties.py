import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import os 

from copy import deepcopy
from scipy.integrate import odeint
from scipy.signal import find_peaks
from scipy import signal

from io import StringIO
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from ipywidgets import *
from IPython.display import display, HTML
from AeroPy import *


pi = np.pi

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST API - Modes estimator'
icon = Image.open(".\logo.ico")
st.set_page_config(page_title=apptitle, page_icon=icon , layout="wide")


# -- File type definition (tab1)
file_mode = ["📈 Upload file" , "📈 Select from database"]
page = st.sidebar.selectbox('Input data file', file_mode)

# -- Load data files
@st.cache_data()
def load_data(uploaded_files,n1,n2):
    try:
        data = pd.read_csv(uploaded_file,skiprows=n1,nrows=n2-n1,delimiter="\s+",encoding_errors='replace')
        error_check += 1
    except:
        tab1.write('Please select the file for analysis')
        data = 0

    return data 



if page == file_mode[0]:
    uploaded_file = st.sidebar.file_uploader("Choose a file",accept_multiple_files=False)

    count = 0
    for line in uploaded_file:
        count += 1
        if bytes("DISTRIBUTED", 'utf-8') in line:
            n1 = count
            if "TOWER" in line:
                element_type = 'tower'
            else:
                element_type = 'blade'
        if bytes('MODE SHAPES', 'utf-8') in line:
            n2 = count

    uploaded_file.seek(0)
    tower = pd.read_csv(uploaded_file,skiprows=np.concatenate((np.arange(n1),[n1+1])),nrows=n2-n1-3,delimiter='\s+',on_bad_lines='skip',encoding_errors='ignore')


else:
    # -- Load data files
    ref_models = {'NREL 5MW':'01_NREL_5MW', 'WP 1.5MW':'02_WINDPACT_1500kW'}
    ref_model = st.sidebar.selectbox('Reference model', ref_models)
    ref_path = ref_models[ref_model]

    sel_dir = '03_AeroDyn/Airfoils'

    all_files = os.listdir('./OpenFAST_models/' + ref_path + '/' + sel_dir + '/')
    uploaded_file = st.sidebar.selectbox('Available aerofoils', all_files)


    x = np.array(pd.read_csv('./OpenFAST_models/' + ref_path + '/' + sel_dir + '/' + uploaded_file,delimiter='\s+',skiprows=1).iloc[:,0])
    y = np.array(pd.read_csv('./OpenFAST_models/' + ref_path + '/' + sel_dir + '/' + uploaded_file,delimiter='\s+',skiprows=1).iloc[:,1])  

n_calc = st.slider('Nº panels',10,100,20,step=2)
alpha = st.number_input('Angle of attack',-10,10,0)
u0 = st.number_input('Free stream velocity',1,None,10)
chord = st.number_input('Chord',None,None,1)


x = (x-x.min())/(x.max()-x.min())
y = (y/(x.max()-x.min()))

AeroCoefficients(x,y,n_calc,u0,alpha,chord)
