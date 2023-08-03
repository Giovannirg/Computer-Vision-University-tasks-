# import the opencv library
#Giovanni Rodriguez Gutierrez
#S0556233

import cv2
from cv2 import vconcat
from cv2 import erode
import numpy as np
  
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
vid.set(cv2.CAP_PROP_FRAME_WIDTH,640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

if not vid.isOpened():
        raise IOError("Cannot open webcam")
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
   
    

    GB3 = cv2.GaussianBlur(frame,(3,3),cv2.BORDER_DEFAULT)
    cv2.putText(GB3, 
                'GaussianBlur 3x3', 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
    GB55 = cv2.GaussianBlur(frame,(55,3),cv2.BORDER_DEFAULT)
    cv2.putText(GB55, 
                'GaussianBlur 55x3', 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
    median = cv2.medianBlur(frame,3,3)
    cv2.putText(median, 
                'MedianBlur 3x3', 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)

    
    
   
    # display input and output image
    cv2.imshow("Gaussian Smoothing",np.hstack((frame, GB3,GB55,median)))
    
    #### High Pass filters.
    # converting to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    # convolute with proper kernels
    laplacian = cv2.Laplacian(gray,cv2.CV_16S,ksize=3)
    # Normalized Capture
    laplacian_norm = cv2.normalize( laplacian, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U);
    cv2.putText(laplacian_norm, 
                'Laplacian', 
                (50, 50), 
                font, 1, 
                (0, 0, 0), 
                3, 
                cv2.LINE_4)
    sobelx = cv2.Sobel(gray,cv2.CV_16S,1,0,ksize=3)  # x
    sobelx_norm = cv2.normalize(sobelx, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U);
    cv2.putText(sobelx_norm, 
                'Sobel X Axis', 
                (50, 50), 
                font, 1, 
                (255, 255, 255), 
                3, 
                cv2.LINE_4)
    sobely = cv2.Sobel(gray,cv2.CV_16S,0,1,ksize=3)  # y
    sobely_norm = cv2.normalize(sobely, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U);
    cv2.putText(sobely_norm, 
                'Sobel y Axis', 
                (50, 50), 
                font, 1, 
                (255, 255, 125), 
                3, 
                cv2.LINE_4)
    cv2.imshow("Hochpass",np.hstack((laplacian_norm, sobelx_norm,sobely_norm)))

    #### Edges and Thresholding.
    # Setting parameter values
    t_lower = 50  # Lower Threshold
    t_upper = 150  # Upper threshold   
    # Applying the Canny Edge filter
    Canny_edge = cv2.Canny(frame, t_lower, t_upper)     
    cv2.putText(Canny_edge, 
                'Canny Edges', 
                (50, 50), 
                font, 1, 
                (255, 255, 255), 
                2, 
                cv2.LINE_4)
    
    # Otsu's thresholding
    ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.putText(th2, 
                'Otsu thresholding', 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
    equ = cv2.equalizeHist(gray)
    cv2.putText(equ, 
                'equalized Histogram', 
                (50, 50), 
                font, 1, 
                (0, 0, 0), 
                2, 
                cv2.LINE_4)
    cv2.imshow("Edges, threshold and EQ",np.hstack((Canny_edge, th2,equ)))

    #### Erosion and dilation of images.
   
   
    kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    img_erosion = cv2.erode(frame, kernel_e, iterations=1)
    cv2.putText(img_erosion, 
                'Erosion', 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17,17))
    img_dilation = cv2.dilate(frame, kernel_d, iterations=1)
    cv2.putText(img_dilation, 
                'Dilation', 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
    cv2.imshow("Erosion and dilation",np.hstack((img_erosion,img_dilation)))



    # the 'q' button is set as the quit button
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()


# Destroy all the windows
cv2.destroyAllWindows()