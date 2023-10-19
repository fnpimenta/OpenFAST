import streamlit as st
import matplotlib.pyplot as plt
from fpdf import *
import base64
import numpy as np
from tempfile import NamedTemporaryFile
from pdf2jpg import pdf2jpg

def create_download_link(val, filename):
	b64 = base64.b64encode(val)  # val looks like b'...'
	return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

def create_pdf_week1(figs,name,title,FileName,placeholder,placeholder_pdf,s1,s2,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
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
	#pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	#placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	#image = pdf2image.convert_from_bytes(pdf.output(dest="S").encode("latin-1"))
	#placeholder_pdf.image(image, use_column_width=True,caption='File preview')

	# Create temporary folder for generated image
	tmp_sub_folder_path = create_tmp_sub_folder()

	# Save images in that sub-folder
	result = pdf2jpg.convert_pdf2jpg(pdf.output(dest="S").encode("latin-1"), pages="ALL")[0]["output_jpgfiles"]
	images = []
	for image_path in result[0]["output_jpgfiles"]:
		images.append(np.array(Image.open(image_path)))

	# Create merged image from all images + remove irrelevant whitespace
	merged_arr = np.concatenate(images)
	merged_arr = crop_white_space(merged_arr)
	merged_path = os.path.join(tmp_sub_folder_path, "merged.jpeg")
	Image.fromarray(merged_arr).save(merged_path)

	# Display the image
	st.image(merged_path)
	try_remove(tmp_sub_folder_path)




	return

def create_pdf_week1_2(figs,name,title,FileName,placeholder,placeholder_pdf,file_id,logo1='figures/ICS.jpg',logo2='figures/FEUP.jpg'):
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

	image = pdf2image.convert_from_bytes(pdf.output(dest="S").encode("latin-1"))
	placeholder_pdf.image(image, use_column_width=True,caption='File preview')


	# Embedding PDF in HTML
	#pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="670" height="957" type="application/pdf"></iframe>'

	# Displaying File
	#placeholder_pdf.markdown(pdf_display, unsafe_allow_html=True)

	return
