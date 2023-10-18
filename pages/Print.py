import streamlit as st
import matplotlib.pyplot as plt
from fpdf import *
import base64
import numpy as np
from tempfile import NamedTemporaryFile

from sklearn.datasets import load_iris



def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

df = load_iris(as_frame=True)["data"]

figs = []

for col in df.columns:
    fig, ax = plt.subplots()
    ax.plot(df[col])
    st.pyplot(fig)
    figs.append(fig)

export_as_pdf = st.button("Export Report")
report_text = st.text_input("Report Text")


if export_as_pdf:
    pdf = FPDF()
    pdf.set_margins(25,18)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(45, 10, '',align='L',ln=0)
    pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
    pdf.image('ICS.jpg',25,20,40)
    pdf.image('FEUP.jpg',23,30,40)
    
    pdf.set_font('Arial',  '',12)
    pdf.cell(45, 10, '',align='L',ln=0)
    pdf.cell(0, 10, 'Week 1 task: Modal configurations',align='L',ln=1)
    pdf.cell(45, 10, '',align='L',ln=0)
    pdf.cell(0, 10,'Name: %s'%report_text,align='L',ln=1)
    Count = 0
    #for fig in figs:
    #    Count += 1
    #    with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
    #        fig.savefig(tmpfile.name)
    #        pdf.image(tmpfile.name,10,50*Count)
    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "testfile")
    st.markdown(html, unsafe_allow_html=True)