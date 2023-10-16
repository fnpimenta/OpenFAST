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
apptitle = 'OpenFAST Course'
icon = Image.open("logo.ico")
st.set_page_config(page_title=apptitle, page_icon=icon , layout="wide")

cols = st.columns(4)

#cols[0].markdown("[![Foo](https://sigarra.up.pt/feup/WEB_GESSI_DOCS.download_file?p_name=F-370784536/logo_cores_oficiais.jpg),](https://sigarra.up.pt/feup/pt/web_page.inicial)")
#cols[2].markdown("[![Foo](https://paginas.fe.up.pt/~icfeup/wp-content/themes/ic-wai/img/vendor/instituto-da-construcao-logo.svg)](https://paginas.fe.up.pt/~icfeup)")
cols[0].markdown('''<a href="https://sigarra.up.pt/feup/pt/web_page.inicial">
                    <img height= 80px 
                    src="https://sigarra.up.pt/feup/WEB_GESSI_DOCS.download_file?p_name=F-370784536/logo_cores_oficiais.jpg" 
                    title="FEUP webpage"/>
                    ''',unsafe_allow_html=True)

cols[2].markdown('''<a href="https://paginas.fe.up.pt/~icfeup">
                    <img height= 80px 
                    src="https://paginas.fe.up.pt/~icfeup/wp-content/themes/ic-wai/img/vendor/instituto-da-construcao-logo.svg" 
                    title="IC webpage" />
                    ''',unsafe_allow_html=True)

# Title the app
st.title('Numerical modeling of wind turbines\n2nd Edition, October 2023')


tabs = st.tabs(['**Summary**' , '**Objectives**' , '**Methodology**'])

with tabs[0]:
    st.markdown("""<div style="text-align: justify">
            Wind energy has a crucial role to play in tackling climate change and in boosting Europe's energy independence, which is fundamental to its economic and political stability. 
            These two objectives have forced Europe and Portugal to set very ambitious targets to dramatically increase their wind energy production infrastructure in a short period of time. 
            Although this process requires human resources with adequate specialized training, currently the offer of courses with a specific focus on wind energy in Portugal is practically non-existent. 
            Therefore, it is necessary to convert human resources with training in the various areas of engineering. Many initiatives will be necessary to cover this gap. 
            This course aims to contribute with an introduction to numerical modelling of wind turbines, using an open-source program with widespread use in academia and industry. 
            Considering the relevance of floating wind turbines in the Portuguese present context, this topic will be also covered in the course. 
            </div>""",unsafe_allow_html=True)

with tabs[1]:

    st.write('''<div style="text-align: justify">
                \nThis course will focus on the numerical modelling of wind turbines. 
                However, and in order to provide to the attendees with~the basic concepts regardingof wind energy, 
                fundamental for a correct implementation and interpretation of numerical models, it also includes modules aimed at exposing the basic theoretical concepts that govern the operation of the most common wind turbines. 
                At the end of the course participants should:</div>''',unsafe_allow_html=True)
    st.write("- know the main components of a wind turbine")
    st.write("- understand the mechanism for generating electricity from wind;")
    st.write("- know how to generate realistic wind fields in order to represent the wind loads;")
    st.write("- understand the structural behaviour of foundation, tower and blades and know how to model the main components;")
    st.write("- understand the aerodynamic behaviour of the rotor and know how to model it;")
    st.write("- know the rotor control mechanisms and know how to model them in a simplified way;")
    st.write("- know the operating principles of floating platforms for offshore wind turbines and their basic modelling;")
    st.write("- know wind turbine operating regimes and singular events and know how to model them;")
    st.write("- know how to interpret numerical modelling results.")

with tabs[2]:
    st.markdown("""<div style="text-align: justify">
                The course will last five weeks: four weeks with classes, with an intermediate week for the autonomous development of exercises. 
                The classes will happen in two weekly sessions (Tuesday and Thursday), starting at 17:30, with a duration of 2h. 
                The sessions will combine the presentation of theoretical concepts with their demonstration in physical models (when possible) and the development of practical modelling exercises. 
                The practical applications will be more concentrated in the second sessions of each week.  
                The resolution of the exercises willth be supported by Python routines that simplify the interaction with the OpenFAST software and allow a quicker analysis of the obtained results. 
                The course has 16 contact hours (+6h of estimated autonomous work). 
                The course can be attained online, but the presence at FEUP is highly recommended, since in person it will be possible to provide a much better support in the resolution of the practical exercises. 
                </div>""",unsafe_allow_html=True)
    st.write("")
    st.markdown('''<div style="text-align: justify">
                For the pratical sessions the <a href="https://www.nrel.gov/docs/fy06osti/32495.pdf">WindPact</a> wind turbine
                and the <a href="https://www.nrel.gov/docs/fy09osti/38060.pdf">NREL 5MW</a>
                will be used.
                </div>''',unsafe_allow_html=True
                )