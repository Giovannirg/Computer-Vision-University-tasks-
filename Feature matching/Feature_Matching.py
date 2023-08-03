# Labor Feature Matching - M8
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

class Mouse:

 def __init__(self):
  self.curr = None # current position of the mouse
  self.down = None # where left button was pressed down
  self.is_down = False
  self.goes_up = False
 
 def callback(self, event, x, y, flags, env):
  self.curr = (x, y)
  if event == cv2.EVENT_LBUTTONDOWN:
     self.down = (x, y)
     self.is_down = True
  if event == cv2.EVENT_LBUTTONUP:
     self.is_down = False
     self.goes_up = True

mouse = Mouse()





cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('img', mouse.callback)

cap = cv2.VideoCapture(0) #change with the location of Picture

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

# Check if the picture is opened correctly

if not cap.isOpened():
        raise IOError("Cannot open the File")

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('img', mouse.callback)

# Red color in BGR
color = (187, 0, 0)
   

## Line thickness of 2 px
thickness = 2

detector = cv2.SIFT_create(2000)


#while Loop:
while cap.isOpened():
  sucess, img = cap.read()
# Copy Image
  out_img = img.copy()
  
  cv2.imshow('img',img) #shows read image

  
  if mouse.is_down:
    
   out_img = cv2.rectangle(out_img, mouse.down, mouse.curr, color, thickness)
   cv2.imshow('img',out_img)

  if mouse.goes_up:
   obj_img = img[mouse.down[1]:mouse.curr[1], mouse.down[0]:mouse.curr[0]].copy()
   mouse.goes_up = False
   #this works only when dragging the mouse down, dragging upwards crashes 
  # concatenate image Vertically
   h1, w1 = obj_img.shape[:2]
   h2, w2 = out_img.shape[:2]

   con_img = np.zeros((max(h1, h2), w1+w2,3), dtype=np.uint8)
   con_img[:,:] = (255,255,255)

   con_img[:h1, :w1,:3] = obj_img
   con_img[:h2, w1:w1+w2,:3] = out_img

   
   #cv2.imshow('con_img',con_img) #shows concatenaded image
  
   obj_kp, des1 = detector.detectAndCompute(obj_img,None)
   out_img_kp, des2 = detector.detectAndCompute(out_img,None)
   out_image_sift = cv2.drawKeypoints(out_img, out_img_kp, None)
   #cv2.imshow('keypoints2',out_image_sift)    #to make the screenshot for report
  
   if obj_img is not None:
         #obj_img_gray= cv2.cvtColor(obj_img,cv2.COLOR_BGR2GRAY) 
         #out_img_gray= cv2.cvtColor(out_img,cv2.COLOR_BGR2GRAY)   
         matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
         matches = matcher.match(des1, des2)

         #find the keypoints and descriptors with SIFT
         matches = sorted(matches, key = lambda x:x.distance)

           # Homography
         src_pts = np.float32([obj_kp[m.queryIdx].pt for m in matches])
         dst_pts = np.float32([out_img_kp[m.trainIdx].pt for m in matches])      

         H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
  

         h, w = obj_img.shape[:-1]
         src_box = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
         dst_box = cv2.perspectiveTransform(src_box, H)
         out_img = cv2.polylines(out_img, [np.int32(dst_box)],True,255,3, cv2.LINE_AA)

         kp_img = cv2.drawMatches(obj_img, obj_kp, out_img, out_img_kp, matches[:50], con_img, flags=2)
         cv2.imshow('Brute Force (BF) Matcher',kp_img)
         
         # Apply ratio test
         bf_matcher = cv2.BFMatcher() #worked only with the defalut parameters, otherwise will crash
         # find best and second best matching descriptor in img_des list
         match_candidates = bf_matcher.knnMatch(des1, des2, k=2)
         # ratio test
         bf_matches = []
         for first, second in match_candidates:
               
            if first.distance < 0.75*second.distance:
               bf_matches.append(first)
                 # Homography
               src_pts = np.float32([obj_kp[m.queryIdx].pt for m in matches])
               dst_pts = np.float32([out_img_kp[m.trainIdx].pt for m in matches])      

               H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
  

               h, w = obj_img.shape[:-1]
               src_box = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
               dst_box = cv2.perspectiveTransform(src_box, H)
               out_img = cv2.polylines(out_img, [np.int32(dst_box)],True,255,3, cv2.LINE_AA)
         ratio_img = cv2.drawMatches(obj_img, obj_kp, out_img, out_img_kp, bf_matches[:50], None, flags=2)
         cv2.imshow('Ratio Test for BF Matcher',ratio_img)

         # FLANN 
         
         index_params = dict(algorithm = 1, trees = 5)
         search_params = dict(checks=50)  

         flann = cv2.FlannBasedMatcher(index_params,search_params)
         flann_matches = flann.knnMatch(des1,des2,k=2)
         fmatches = []
        
         for first, second in match_candidates:
               
            if first.distance < 0.75*second.distance:
               fmatches.append(first)
                 # Homography
               src_pts = np.float32([obj_kp[m.queryIdx].pt for m in matches])
               dst_pts = np.float32([out_img_kp[m.trainIdx].pt for m in matches])      

               H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
  

               h, w = obj_img.shape[:-1]
               src_box = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
               dst_box = cv2.perspectiveTransform(src_box, H)
               out_img = cv2.polylines(out_img, [np.int32(dst_box)],True,255,3, cv2.LINE_AA)
               flann_img = cv2.drawMatches(obj_img, obj_kp, out_img, out_img_kp, fmatches[:50], None, flags=2)
         
         cv2.imshow('FLANN',flann_img) 
  
    


  if cv2.waitKey(1) & 0xFF == ord('q'):
     break
img.release()
cv2.destroyAllWindows()

