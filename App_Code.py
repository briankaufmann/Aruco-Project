#FOR APP DEVELOPMENT
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from datetime import date
import streamlit as st
from streamlit.components.v1 import iframe
import pandas as pd
from PIL import Image

#FOR IMAGE PROCESSING AND CV
import ipywidgets as widgets
from matplotlib import pyplot as plt
import numpy as np
import cv2


#Code for initial app design ------------------------------

st.header("Detect Aruco Markers in an Image")
uploaded_file = st.file_uploader("Choose an image...")

    
#Open image for app testing if there is one

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)    
    st.image(img)

#BELOW CODE IS FOR ARUCO DETECTION

if uploaded_file is not None:
    #Read image and sharpen, then convert to grayscale
    #img = cv2.imread(img)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    gray = cv2.cvtColor(sharpen,cv2.COLOR_BGR2GRAY)

    #Convert image to binary
    _, binary = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)

    #Building kernel

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
            if area > 10:
                #Calculate rotated bounding boxes
                rot_rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rot_rect)
                box = np.int0(box)
                #Saving boxes and contour locations 
                boxes = [boxes, box]
                contour = [contour, c]
                #drawing boxes on a new image
                cv2.drawContours(img_cnt,[box],0,(0,0,255),1)
         
    #Showing the processed image        
    st.image(img_cnt)



