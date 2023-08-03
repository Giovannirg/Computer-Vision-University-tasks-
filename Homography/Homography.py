import cv2
import numpy as np
 
# reading image in grayscale
# img = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)
 
# initializing web cam
vid = cv2.VideoCapture(0)

vid.set(cv2.CAP_PROP_FRAME_WIDTH,640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

boardW = 8
boardH = 6

if not vid.isOpened():
        raise IOError("Cannot open webcam")
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
      
 # Capture the video frame
 # by frame
 ret, frame = vid.read()
  
 img = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
 cv2.imshow('frame', img)

 found, corners = cv2.findChessboardCorners(img, (boardW,boardH ), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
 img = 255 - img
 corners = np.squeeze(corners) 
 
 cv2.circle(img, np.int32((corners)))

 
    
    
    # ret, corners = cv2.findChessboardCorners(img, corners, None)   

    # srPoints = np.array([corners[0, 0], corners[8, 0], corners[45, 0], corners[corners.shape[0] - 1, 0]]) 
    
    # H, _ = cv2.findHomography(srcPoints, ...)

    # the 'q' button is set as the quit button
   
 if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()


# Destroy all the windows
cv2.destroyAllWindows()