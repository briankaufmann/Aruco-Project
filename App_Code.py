
import streamlit as st
from streamlit.components.v1 import iframe
from PIL import Image
import ipywidgets as widgets
import numpy as np
import cv2

AR_low = 0.8
AR_high = 1.2



#!!!Sections are good, but can be improved by splitting your code out, so have a ui file and also have an open CV file'''
    

def main():
    st.title("Detect Aruco Markers in an Image")
    #Upload the file
    uploaded_file = st.file_uploader("Choose an image...")

    #Add spaces for Aesthetics
    st.title("    ")
    st.title("    ")
    st.title("    ")

    if uploaded_file is None:
        return
 
    #Create the sidebar with sliders and get useful variables from slider values
    sharpen_val, thresh_val, area_val = side_bar()

    #Convert the file to an openCV Image and output the image
    img = convert_image(uploaded_file)

    #Process the image using inputs from sliders in sidebar and output processed images and boxes
    contours, sharpen, binary, img_cnt = image_processing(sharpen_val, thresh_val, area_val, img)

    #Show the detected boxes on the preprocessed image 
    visualize_boxes(contours, area_val, img_cnt)

    #Visualize all the images in the UI
    col1, col2, col3 = st.columns(3)
    add_image_to_column(col=col1, img=img, title='Original')
    add_image_to_column(col=col2, img=sharpen, title='Sharpen')
    add_image_to_column(col=col3, img=binary, title='Binary')

    st.header("Detected Aruco Markers")
    st.image(img_cnt, use_column_width=True, caption=["Detected Aruco Markers"])


def side_bar():
    with st.sidebar:
        st.subheader("Adjust the sliders below to change how the program detects the Aruco markers")
        #Insert a slider in the app to alter the sharpness of the image
        sharpen_val = st.slider('Sharpen Image', 1, 20, step=1, value=8, key=1)
        #Insert a slider in the app to alter the bionary threshold of the image
        thresh_val = st.slider('Binary Threshold Value', 0, 255, step=10, value=65, key=2)
        #Insert a slider in the app to alter the area size of the detection boxes
        area_val = st.slider('Detection Size', 0 , 500, step=5, value=0, key=3)
        
        return sharpen_val, thresh_val, area_val
    
        
def convert_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    return img


def image_processing(sharpen_val, thresh_val, area_val, img):
    sharpen_kernel = np.array([[-1,-1,-1], [-1,sharpen_val,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    gray = cv2.cvtColor(sharpen,cv2.COLOR_BGR2GRAY)


    #Convert image to binary
    _, binary = cv2.threshold(gray,thresh_val,255,cv2.THRESH_BINARY)


    #Finding contours of the arucos 
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

    #Create a new image to hold the boxes
    img_cnt = img.copy()

    return contours, sharpen, binary, img_cnt


def visualize_boxes(contours, area_val, img_cnt):
    
    #Initialize lists to save the contours and boxes
    contour = []
    boxes = []    
    for contour in contours:
        #Extract bounding box
        (x,y,w,h) = cv2.boundingRect(contour)
        #Check if aspect ratio is about = to that of a square
        AR = w/h
        #!!!'''Handle the edge cases first so that you don't end up with lots of indented code e.g.
        
        #!!!if AR > AR_high or AR < AR_low:
            # Aspect ratio is not close to a square
         #!!!   continue

        if AR <= AR_high and AR > AR_low:
            #Find the areas of bounding boxes that fit the AR criteria
            area = cv2.contourArea(contour)
            #Ignore small speckled noise for bounding boxes
            if area > area_val:
                box = find_boxes(contour)
                boxes.append(box)
                contour = [contour, contour]
                #drawing boxes on a new image
                cv2.drawContours(img_cnt,[box],0,(0,0,255),1)

                
 #!!!!make this section a function that returns the box'''
                
def find_boxes(contour):
    #Calculate rotated bounding boxes
    rot_rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rot_rect)
    box = np.int0(box)
    return box


def add_image_to_column(col: any, img: any, title: str = 'Default Image Title', use_column_width: bool = True) -> None:
    col.subheader(title)
    col.image(img, use_column_width)


if __name__ == '__main__':
    ##aruco_marker_ui()
    main()
    


    

