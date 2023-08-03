# Labor Feature Tracking - M8
#Giovanni Rodriguez Gutierrez
#S0556233

import cv2
from pathlib import Path
from cv2 import vconcat
from cv2 import erode
import numpy as np
  
  
# define a video capture object

#data_folder = Path("Documents/SoSE2022/M8/3Labor/")

#video = data_folder / "A10bridge2.mp4"

cap = cv2.VideoCapture('/Users/korvo/Documents/SoSE2022/M8/4Labor/A10bridge2.mp4') #change with the location of the video


# Check if the video is opened correctly

if not cap.isOpened():
        raise IOError("Cannot open Video")

#cap.set(1,150); # from which frame do you want to start
ret, frame = cap.read()

feature_params = dict(
maxCorners = 100,
qualityLevel = 0.3,
minDistance = 7,
blockSize = 7
)

#Overlay and Algorithm Parameters

overlay = np.zeros_like(frame)

color = np.random.randint (0, 255, (feature_params['maxCorners'], 3))

lk_params = dict(
winSize = (31, 31),
maxLevel = 4,
criteria = (cv2.TERM_CRITERIA_COUNT+cv2.TERM_CRITERIA_EPS, 100, 0.01)
)

#while Loop:
while cap.isOpened():
# Capture frame-by-frame
  ret,frame = cap.read()
  
  curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  prev_feature_points = cv2.goodFeaturesToTrack(prev_frame_gray, mask=None, **feature_params) 

  
  curr_feature_points, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, curr_frame_gray, prev_feature_points,None, **lk_params)
  #cv2.imshow('frame', frame)
  curr_good_feature_points = curr_feature_points[status == 1]
  prev_good_feature_points = prev_feature_points[status == 1]

  for i, (new,old) in enumerate(zip(curr_good_feature_points,prev_good_feature_points)):
   a, b = new.ravel()
   c, d = old.ravel()
   overlay = cv2.line(overlay, np.int32((a,b)), np.int32((c,d)), color[i].tolist(), 2)

   frame = cv2.circle(frame, np.int32((a, b)), 5, color[i].tolist(), -1)


   img = cv2.add(frame, overlay)
   cv2.imshow('frame', img)

   prev_frame_gray = curr_frame_gray.copy()
   prev_feature_points = curr_good_feature_points.reshape(-1, 1, 2)

   Exit
 
  if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()