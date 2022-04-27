#FOR APP DEVELOPMENT


from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from datetime import date
import streamlit as st
from streamlit.components.v1 import iframe
import pandas as pd
from PIL import Image

#FOR IMAGE PROCESSING AND CV
import ipywidgets as widgets
#from matplotlib import pyplot as plt
import numpy as np
import cv2


#--------------- APP DESIGN AND FORMATTING  ------------------------------

st.title("Detect Aruco Markers in an Image")

#Upload the file
uploaded_file = st.file_uploader("Choose an image...")
st.title("    ");
st.title("    ");
st.title("    ");
    
#Open image for app testing if there is an image uploaded
if uploaded_file is not None:
    
    # Insert a slider on the left side to change values
    with st.sidebar:
        st.subheader("Adjust the sliders below to change how the program detects the Aruco markers")
        #Insert a slider in the app to alter the sharpness of the image
        sharpen_val = st.slider('Sharpen Image', 1, 20, 1)
        #Insert a slider in the app to alter the bionary threshold of the image
        thresh_val = st.slider('Binary Threshold Value', 0, 255, 10)
        #Insert a slider in the app to alter the area size of the detection boxes
        area_val = st.slider('Detection Size', 0 , 500, 5)
        
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    img = cv2.imdecode(file_bytes, 1)    
    #st.image(img)

#------------ IMAGE PROCESSING AND ARUCO DETECTION------------------------------------

    sharpen_kernel = np.array([[-1,-1,-1], [-1,sharpen_val,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    gray = cv2.cvtColor(sharpen,cv2.COLOR_BGR2GRAY)


    #Convert image to binary
    _, binary = cv2.threshold(gray,thresh_val,255,cv2.THRESH_BINARY)


    #Finding contours of the arucos 
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
    
    #Initialize lists to save the contours and boxes
    contour = []
    boxes = []

    #Create a new image to hold the boxes
    img_cnt = img.copy()

    #Establish upper and lower bounds of aspect ratio
    AR_low = 0.5
    AR_high = 1.5

    #For loop to find contours
    for c in contours:
        #Extract bounding boz
        (x,y,w,h) = cv2.boundingRect(c)
        #Check if aspect ratio is about = to that of a square
        AR = w/h
        if AR <= AR_high and AR > AR_low:
            #Find the areas of bounding boxes that fit the AR criteria
            area = cv2.contourArea(c)
            #Ignore small speckled noise for bounding boxes
            if area > area_val:
                #Calculate rotated bounding boxes
                rot_rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rot_rect)
                box = np.int0(box)
                #Saving boxes and contour locations 
                boxes = [boxes, box]
                contour = [contour, c]
                #drawing boxes on a new image
                cv2.drawContours(img_cnt,[box],0,(0,0,255),1)


#--------------- DISPLAY OF IMAGES ----------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.subheader("Original")
    col1.image(img, use_column_width=True)
    col2.subheader("Sharpened")
    col2.image(sharpen, use_column_width=True)
    col3.subheader("Binary")
    col3.image(binary, use_column_width=True)
    st.header("Detected Aruco Markers")
    st.image(img_cnt, use_column_width=True, caption=["Detected Aruco Markers"])

    

