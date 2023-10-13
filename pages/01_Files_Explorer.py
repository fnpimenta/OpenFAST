import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 

# -- Side bar definition
tabs = st.sidebar.tabs(["📈 File download" , "📈 Data analysis" , "🌊 File generator"])

tab1 = tabs[0]
tab2 = tabs[1]
tab3 = tabs[2]

# -- Load data files
ref_models = {'NREL 5MW':'01_NREL_5MW', 'WP 1.5MW':'02_WINDPACT_1500kW'}
ref_model = tab1.selectbox('Reference model', ref_models)
ref_path = ref_models[ref_model]

all_dir = os.listdir('./OpenFAST_models/' + ref_path )
sel_dir = tab1.selectbox('Available modules', all_dir)

all_files = os.listdir('./OpenFAST_models/' + ref_path + '/' + sel_dir)
sel_file = tab1.selectbox('Available files', all_files)

explore_file = tab1.checkbox('Explore file content',False)

if explore_file:
	log = open('./OpenFAST_models/' + ref_path + '/' + sel_dir + '/' + sel_file, 'r')
	for line in log:
	    st.write(line)