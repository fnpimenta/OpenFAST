import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from copy import deepcopy
from scipy.integrate import odeint
from scipy.signal import find_peaks
from scipy import signal

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST API - Data analysis'
icon = Image.open("logo.ico")
st.set_page_config(page_title=apptitle, page_icon=icon)

# Title the app
st.title('OpenFAST analysis')

st.markdown("""
 * Use the menu at left to select data from the different analysis possibilities
 * To tune the analysis parameters use the **Analysis** tab
""")

# -- Side bar definition
tabs = st.sidebar.tabs(["📈 File upload" , "📈 Data analysis" , "🌊 File generator"])

tab1 = tabs[0]
tab2 = tabs[1]
tab3 = tabs[2]

# -- Load data files

@st.cache_data()
def load_data(uploaded_files):
	N = len(uploaded_files)

	data = []
	keys = []

	error_check = 0

	for uploaded_file in uploaded_files:
		try:
			data.append(pd.read_csv(uploaded_file,skiprows=[0,1,2,3,4,5,7],delimiter="\s+",encoding_errors='replace'))
			keys.append(list(data[-1]))
			
			error_check += 1
		except:
			tab1.write('Please select the file for analysis')

	return data , keys , error_check 

uploaded_files = tab1.file_uploader("Choose a file",accept_multiple_files=True)
N = len(uploaded_files)

data , keys , error_check  = load_data(uploaded_files)
cols = st.columns(4)
nvar = len(keys)

selected = []

for i in range(1,len(keys[0])):
	key = keys[0][i]
	selected.append(cols[int((i-1)%4)].checkbox(key))

sel_var = [keys[0][x+1] for x in range(len(keys[0])-1) if selected[x]]

n_sel_var = len(sel_var)

if (error_check == N) and (N>0) :

	sep_plots = st.checkbox('Separate plots',value=True)
	

	tcol = tab2.selectbox('Time column', data[0].columns,index=0)
	dof = tab2.selectbox('Data column', data[0].columns,index=4)

	# -- File type definition (tab1)
	fig = plt.figure(figsize=(12,8))
	gs = gridspec.GridSpec(N,n_sel_var,hspace=0.1,wspace=0.25)

	if sep_plots:
		for i in range(len(data)):
			t = np.array(data[i][tcol])

			for j in  range(n_sel_var):
				ax = plt.subplot(gs[i,j])
				dof = sel_var[j]
				y = np.array(data[i][dof])

				ax.plot(t,y)
				ax.set_xticklabels('')
	else:
		ax = plt.subplot(gs[0,0])
	
		for i in range(len(data)):
			t = np.array(data[i][tcol])

			for dof in sel_var:
				y = np.array(data[i][dof])
				ax.plot(t,y)	

	st.pyplot(fig,theme='streamlit')  	

	#st.plotly_chart(fig,theme='streamlit')  

