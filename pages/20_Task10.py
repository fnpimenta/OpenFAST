import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from scipy import signal, interpolate
import rainflow

from modes import *
from estimators import *
from plot_generators import *
from Print import *

from PIL import Image

import rainflow

# -- Set page config
apptitle = 'OpenFAST Course - Task 10'
icon = Image.open('feup_logo.ico')
st.set_page_config(page_title=apptitle, page_icon=icon)

st.title('Task 10 - Fatigue evaluation')

@st.cache_data()
def FatigueDamage(data,m=[3,5],detail = 80,Gama_f=1,Gama_m=1,Gama_n=1):
	a_bar_1 = detail**m[0]*2e6
	Sigma_D = (a_bar_1/(1e7))**(1/m[0]);
	a_bar_2 = 1e7*Sigma_D**m[1]

	# 1st column-> d; 2nd column-> D
	Damage = np.zeros((len(data),2));     
	for k in range(len(data)): 
		Damage[k,0] = data[k][1];
		if data[k][0] < Sigma_D:   # Inside the 2nd region
			Damage[k,1] = a_bar_2/(data[k][0]**m[1]);

		else:                      # Inside the 1st region
			Damage[k,1] = a_bar_1/(data[k][0]**m[0]);

	Damage[:,1] = Damage[:,1]/(Gama_f*Gama_m*Gama_n);      # Partial safety factor for materials

	Damage = sum(Damage[:,0]/Damage[:,1])
	return Damage

# -- Load data files
@st.cache_data()
def load_data(uploaded_files,n1,n2):
	try:
		data = pd.read_csv(uploaded_file,skiprows=n1,nrows=n2-n1,delimiter="\s+",encoding_errors='replace')
		error_check += 1
	except:
		c1.write('Please select the file for analysis')
		data = 0

	return data

with st.expander("**Objective**",True):

	st.write('''<div style="text-align: justify">
			\nTo study the impact of the pitch control parameters on the response, 
			repeat Task 7 but with a higher and lower sensitivity of the pitch system. 
			For clarity, it is suggested that you use an high wind speed (for instance, 16m/s), ensuring that the wind turbine is operating in Region 3, where the pitch control impact is more visible.
			\nYou may find a step-by-step procedure in the hints section.
			</div>''',unsafe_allow_html=True)

figs = []

PALETTE = [
	"#ff4b4b",
	"#ffa421",
	"#ffe312",
	"#21c354",
	"#00d4b1",
	"#00c0f2",
	"#1c83e1",
	"#803df5",
	"#808495",
]

onshore_color = "#cc000033"
offshore_color = "#0a75ad33"

ref_models = {'NREL 5MW':'01_NREL_5MW', 'WP 1.5MW':'02_WINDPACT_1500kW'}
ref_path = ref_models['WP 1.5MW']

file_OpenFAST = open('./OpenFAST_models/' + ref_path + '/' + 'TestFile.fst', 'r')
file_struct = open('./OpenFAST_models/' + ref_path + '/' + '01_ElastoDyn' + '/' + 'WP_ElastoDyn.dat', 'r')
file_wind =  open('./OpenFAST_models/' + ref_path + '/' + '02_InflowWind' + '/' + 'InflowWind_W0500_Steady.dat', 'r')
file_aero = open('./OpenFAST_models/' + ref_path + '/' + '03_AeroDyn' + '/' + 'WP_AeroDyn.dat', 'r')
file_control =  open('./OpenFAST_models/' + ref_path + '/' + '04_ServoDyn' + '/' + 'WP_ServoDyn.dat', 'r')

onshore_color = "#cc000033"
offshore_color = "#0a75ad33"

checkfile=1

with st.expander("**Hints**",False):

	st.write('''<div style="text-align: justify">
	\nThe suggested procedure is the following:
1. Modify the **OpenFAST input file**:
	- Modify the comment line for a description of your simulation
	</div>''',unsafe_allow_html=True)

	st.divider()

	st.write('''<div style="text-align: justify">	
		\n
2. Modify the **ServoDyn input file**:
	- Modify the pitch control parameters in the <b>SIMPLE VARIABLE-SPEED TORQUE CONTROL</b> section such that:
		- The pitch control is 2x more sensitive to variations in the wind speed (Simulation 1)                    
		- The pitch control is 2x less sensitive to variations in the wind speed (Simulation 2)
	- To make this change, you may modify the gain on TF 1 on the pitch input file
			</div>''',unsafe_allow_html=True)
	
	st.divider()

	st.write('''<div style="text-align: justify">	
		\n
4. Modify the **InflowWind input file** (or use the same from Task 6):
	- Modify the wind type for TurbSim FF (full field) wind in <b>Wind model</b>
	- Include the paths to the corresponding TurbSim generated (.bts) with an high wind speed (for instance, 16 m/s) file in <b>Parameters for Binary TurbSim Full-Field files</b> section
	(you may have to generate this wind field)
			</div>''',unsafe_allow_html=True)

	#checkfile = st.checkbox('**Show InflowWind file details**')
	if checkfile:
		data = []
		for line in file_wind:
			data.append(line)


		tab1,tab2,tab3,tab4 = st.tabs(['**Wind model**',
									   'Steady wind properties',
									   'Uniform wind properties',
									   '**TurbSim full field**'])

		all_idx = range(3,10)
		on_sel_idx = [4]
		off_sel_idx = []
		with tab1:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		all_idx = range(11,14)
		on_sel_idx = []
		off_sel_idx = []
		with tab2:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		all_idx = range(15,18)
		on_sel_idx = []
		off_sel_idx = []
		with tab3:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		all_idx = range(19,20)
		on_sel_idx = [19]
		off_sel_idx = []
		with tab4:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		st.divider()

	st.write('''<div style="text-align: justify">	
		\n
5. Run the OpenFAST simulation and upload the output files.
			</div>''',unsafe_allow_html=True)


with st.expander("**Data analysis**",True):
	file = st.file_uploader("Upload the output files from OpenFAST",accept_multiple_files=False)

	error_check = not(file==None)

	if error_check>0:
		file.seek(0)
		data = pd.read_csv(file , skiprows=[0,1,2,3,4,5,7] , delimiter=r"\s+",header=0)

		cols = st.columns(2)
		t_min = cols[0].number_input('First time instant to plot',0.0,1000.0,400.0)
		t_max = cols[1].number_input('Last time instant to plot',0.0,1000.0,1000.0)
		time_filter = (data.Time>=t_min) & (data.Time<=t_max)
		
		v_mean = 0.2*cols[0].number_input('Reference wind speed',min_value=10,max_value=50,value=50)
		k_weibull = cols[1].number_input('Weibull scale parameter',min_value=0.1,max_value=10.0,value=2.0)

		fs = 20                                 # Hz
		E = 210e6                               # kPa
		r = 5660/2000                           # m
		th = 17.4/1000                          # m
		r -= th/2                               # m
		ConvFactor = pi*r**2*th                 # m^3 

		t = data['Time']
		mfa = data['TwrBsMyt']

		rf = rainflow.count_cycles(-1e-3*np.array(mfa[time_filter])/ConvFactor)

		smax = max(rf)[0]
		smin = np.max((min(rf)[0],smax/100))

		bins = np.logspace(np.log10(smin),np.log10(smax),20)

		D_all = np.zeros(len(bins))
		Count_all = np.zeros(len(bins))

		for j in range(len(bins)):
			if bins[j]>max(rf)[0]:
				D_all[j] = FatigueDamage(rf)
				for k in range(len(rf)):
					Count_all[j] += rf[k][1] 
			else:
				Count = 0
				while (rf[Count][0]<bins[j]) & (Count<len(rf)-1):
					Count += 1
				D_all[j] = FatigueDamage(rf[0:Count])
				for k in range(Count):
					Count_all[j] += rf[k][1] 
					
		fig = plt.figure(figsize = (12,8))
		gs = gridspec.GridSpec(2, 2)

		#gs.update(wspace=2,hspace=0.25)
			
		ax_m = plt.subplot(gs[0,:])
		ax1 = plt.subplot(gs[1,1])
		ax2 = plt.subplot(gs[1,0])

		ax1.plot(bins,D_all,lw=2)
		ax_m.plot(t[time_filter], 1e-3*(mfa[time_filter] - mfa[time_filter].mean()))

		ax1.set_xscale('log')
		ax1.grid()

		ax2.hist(np.array([rf[i][0] for i in range(len(rf))]),bins=bins)
		ax2.set_xscale('log')

		ax1.set_yticks([0,D_all[-1]])
		
		ax1.set_xlabel('$\Delta\sigma$ (MPa)')
		ax1.set_ylabel('Accumulated fatigue damage')

		ax_m.set_xlabel('Time (s)')
		ax_m.set_ylabel('Bending moment (MN.m)')
		ax_m.grid()

		ax2.set_ylabel('Number of cycles')
		ax2.set_xlabel('$\Delta\sigma$ (MPa)')

		ax_s = ax_m.twinx()
		ax_s.set_yticks(ax_m.get_yticks())
		ax_s.set_yticklabels(np.round(ax_m.get_yticks() * 1/ConvFactor,1))
		ax_s.set_ylabel('$\Delta\sigma$ (MPa)')

		st.pyplot(fig)

		figs.append(fig)

		ws = np.linspace(0,25,101)

		v_hub = data['Wind1VelX'][time_filter].mean()
		v_low = np.floor(v_hub)
		v_high = np.ceil(v_hub)

		A = 2*v_mean/np.sqrt(pi)

		rayleigh = 100 * pi/2 * ws/v_mean**2 * np.exp(-pi*(ws/(2*v_mean))**2)
		weibull = 100 * k_weibull/A * (ws/A)**(k_weibull-1) * np.exp(-(ws/A)**k_weibull)
		weibull_mean = np.mean(weibull[(ws>=v_low) & (ws<=v_high)])


		fig = plt.figure(figsize = (12,4))

		ax1 = plt.subplot()

		ax1.plot(ws,rayleigh,label='Rayleigh distribution')
		ax1.plot(ws,weibull,label='Weibull distribution (k=%.1f)'%k_weibull)
		ax1.plot([v_low,v_low],[0,weibull_mean],'r')
		ax1.plot([v_high,v_high],[0,weibull_mean],'r')
		ax1.plot([v_low,v_high],[weibull_mean,weibull_mean],'r')
		ax1.plot([0,v_low],[weibull_mean,weibull_mean],'--r')

		ax1.fill_between(x=[v_low,v_high],y1=[0,0],y2=[weibull_mean,weibull_mean],color='r',alpha=0.5)
		
		ax1.set_xticks([0,v_hub,25])
		ax1.set_yticks([0,np.round(weibull_mean,1)])

		ax1.set_ylim(0)
		ax1.set_xlim(0,25)

		ax1.set_ylabel('Frequency of occurrence (%)')
		ax1.set_xlabel('Hub wind speeds (m/s)')


		leg = ax1.legend(loc='lower center',
						  bbox_to_anchor=(0.5,1),
						  ncol=2,
						  fancybox=False,
						  framealpha=1,
						  frameon=False)
		st.pyplot(fig)

		figs.append(fig)

exp = st.expander('**Export report**',False)

with exp:
	report_text = st.text_input("Name")
	st.write('''<div style="text-align: justify">
		\nBased on the 10 minutes fatigue damage and on the frequency of occurence for the distribution you have chosen, 
		estimate the contribution of this event for the total fatigue consumption during 25 years:
		</div>''',unsafe_allow_html=True)
	Dt = st.number_input("Total damage")

exp_c = exp.columns([0.25,0.25,0.5])
export_as_pdf = exp_c[0].button("Generate Report")
if export_as_pdf:
	try:
		create_pdf_task10(figs,report_text,'Task 10: Fatigue analysis','Task10_report',exp_c[1],exp,Dt)
	except:
		exp.error('Something went wrong. No file available for the analysis.', icon="⚠️")