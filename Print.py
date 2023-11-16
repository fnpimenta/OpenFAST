import streamlit as st
import matplotlib.pyplot as plt
from fpdf import *
import base64
import numpy as np
from tempfile import NamedTemporaryFile

def create_download_link(val, filename):
	b64 = base64.b64encode(val)  # val looks like b'...'
	return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

def create_pdf_task1(figs,name,title,FileName,placeholder,placeholder_pdf,s1,s2,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Fitted mode shapes',border=border,align='L',ln=1)
	Count = 0
	for fig in figs:
		Count += 1
		with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
			fig.savefig(tmpfile.name, bbox_inches='tight')
			pdf.cell(45, 10, '',border=border,align='L',ln=1)
			pdf.image(tmpfile.name,25,60,w=0,h=110)
	pdf.cell(0, 110, '',border=border,align='L',ln=1)
	pdf.set_font('Arial', '' , 10)
	pdf.cell(45, 10,r'Scaling factor to apply to Mode 1: %.2f'%s1,border=border,align='L',ln=1)
	pdf.cell(45, 10,r'Scaling factor to apply to Mode 2: %.2f'%s2,border=border,align='L',ln=1)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return

def create_pdf_task2(figs,name,title,FileName,placeholder,placeholder_pdf,file_id,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Preliminary data analysis',border=border,align='L',ln=1)
	Count = 0
	for fig in figs:
		with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
			fig.savefig(tmpfile.name, bbox_inches='tight')

			pdf.image(tmpfile.name,25,60 + 80*Count,w=160,h=0)
			Count += 1
	pdf.cell(0, 75, '',border=border,align='L',ln=1)
	pdf.cell(45, 10,'Free decay analysis of simulation %d'%file_id,border=border,align='L',ln=1)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return

def create_pdf_task3(figs,name,title,FileName,placeholder,placeholder_pdf,yp=0,zp=0,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Full 3D wind field generated with TurbSim',border=border,align='L',ln=1)

	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs[0].savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,25,60 ,w=80,h=0)

	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs[1].savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,105,60 ,w=80,h=0)

	pdf.cell(0, 90, '',border=border,align='L',ln=1)
	pdf.cell(45, 10,'Detailed analysis of the wind speeds in y=%.1f m and z=%.1f m'%(yp,zp),border=border,align='L',ln=1)

	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs[2].savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,25,165 ,w=160,h=0)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return

def create_pdf_task4(figs,name,title,FileName,placeholder,placeholder_pdf,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Free decay analysis with pitch angle set to 90',border=border,align='L',ln=1)
	Count = 0
	for fig in figs:
		with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
			fig.savefig(tmpfile.name, bbox_inches='tight')

			pdf.image(tmpfile.name,25+15,60 + 105*Count,w=160-30,h=0)
			Count += 1
	pdf.cell(0, 95, '',border=border,align='L',ln=1)
	pdf.cell(45, 10,'Free decay analysis with pitch angle set to 0',border=border,align='L',ln=1)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return

def create_pdf_task5(figs,name,title,FileName,placeholder,placeholder_pdf,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Parked simulations with different wind models',border=border,align='L',ln=1)

	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs.savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,25,60 ,w=160,h=0)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return

def create_pdf_task6(figs,name,title,FileName,placeholder,placeholder_pdf,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Normal operation simulations with different wind models',border=border,align='L',ln=1)

	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs.savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,25,60 ,w=160,h=0)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return

def create_pdf_task7(figs,name,title,FileName,placeholder,placeholder_pdf,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Normal operation simulations with control systems',border=border,align='L',ln=1)

	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs.savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,25,60 ,w=160,h=0)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return

def create_pdf_task8(figs,name,title,FileName,placeholder,placeholder_pdf,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Impact of the torque curve parameters on the response',border=border,align='L',ln=1)

	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs.savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,25,60 ,w=160,h=0)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return

def create_pdf_task9(figs,name,title,FileName,placeholder,placeholder_pdf,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Impact of the pitch control on the response',border=border,align='L',ln=1)

	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs.savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,25,60 ,w=160,h=0)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return

def create_pdf_task10(figs,name,title,FileName,placeholder,placeholder_pdf,d,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Fatigue damage for a 10 minutes simulation',border=border,align='L',ln=1)

	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs[0].savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,25,60 ,w=160,h=0)

	pdf.cell(0, 105, '',border=border,align='L',ln=1)
	pdf.cell(45, 10,'Weibull distribution',border=border,align='L',ln=1)
	
	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs[1].savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,25,175 ,w=160,h=0)

	pdf.set_font('Arial', '' , 10)
	pdf.cell(0, 75, '',border=border,align='L',ln=1)
	pdf.cell(45, 10,r'Total fatigue damage for 25 years of operation: %.2f'%d,border=border,align='L',ln=1)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return

def create_pdf_task11(figs,name,title,FileName,placeholder,placeholder_pdf,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Idling wind turbine simulation',border=border,align='L',ln=1)

	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs.savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,25,60 ,w=160,h=0)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return

def create_pdf_task12(figs,name,title,FileName,placeholder,placeholder_pdf,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Wind turbine start up simulation',border=border,align='L',ln=1)

	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs.savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,25,60 ,w=160,h=0)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return

def create_pdf_task13(figs,name,title,FileName,placeholder,placeholder_pdf,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Wind turbine normal shutdown simulation',border=border,align='L',ln=1)

	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs.savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,25,60 ,w=160,h=0)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return

def create_pdf_task14(figs,name,title,FileName,placeholder,placeholder_pdf,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Wind turbine emergency shutdown simulation',border=border,align='L',ln=1)

	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs.savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,25,60 ,w=160,h=0)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return

def create_pdf_task15(figs,name,title,FileName,placeholder,placeholder_pdf,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
	border = 'LRTB'*0
	pdf = FPDF()
	pdf.set_margins(25,18)
	pdf.add_page()
	pdf.set_font('Arial', 'B', 18)
	pdf.cell(45, 10, '',align='L',ln=0)
	pdf.cell(0, 10, 'Numerical modelling of wind turbines',align='L',ln=1)
	pdf.image(logo1,25,20,40)
	pdf.image(logo2,23,30,40)

	pdf.set_font('Arial',  '',12)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10, title,border=border,align='L',ln=1)
	pdf.cell(45, 10, '',border=border,align='L',ln=0)
	pdf.cell(0, 10,'Name: %s'%name,border=border,align='L',ln=1)
	pdf.set_font('Arial', 'B' , 12)
	pdf.cell(45, 10,'Hydrodynamic coefficients for different frequencies',border=border,align='L',ln=1)

	with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
		figs.savefig(tmpfile.name, bbox_inches='tight')
		pdf.image(tmpfile.name,25,60 ,w=160,h=0)

	html = create_download_link(pdf.output(dest="S").encode("latin-1"), FileName)
	placeholder.markdown(html, unsafe_allow_html=True)

	base64_pdf = base64.b64encode(pdf.output(dest="S").encode("latin-1")).decode('utf-8')

	# Embedding PDF in HTML
	pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return