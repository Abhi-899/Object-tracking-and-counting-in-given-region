# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 01:07:06 2021

@author: Param
"""

import cv2
import time

# initialize opencv background subtractor
BS_KNN=cv2.createBackgroundSubtractorKNN()
BS_MOG2=cv2.createBackgroundSubtractorMOG2()
cap=cv2.VideoCapture('highway.mp4')
while cap.isOpened:
  success,frame=cap.read()
  if not success:
      break
  KNN_mask=BS_KNN.apply(frame)
  cv2.imshow('KNN mask',KNN_mask)
  MOG_mask=BS_MOG2.apply(frame)
  cv2.imshow('MOG2 mask',MOG_mask)
  if cv2.waitKey(1) & 0xff==ord('q'):
      break
  
cv2.destroyAllWindows()
cap.release()

# With absdiff

cap=cv2.VideoCapture('highway.mp4')
vehicle=0
success,frame1=cap.read()
h,w,c=frame1.shape
out = cv2.VideoWriter('counter.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, (w,h))
while cap.isOpened:
  success,frame2=cap.read()
  frame=frame2.copy()
  if not success:
      break
  
  fgMask=cv2.absdiff(frame1,frame2)
  thresh_val,thresh_img=cv2.threshold(fgMask,50,255,cv2.THRESH_BINARY)
  thresh_img = cv2.cvtColor(thresh_img, cv2.COLOR_BGR2GRAY )
  frame1=frame2
  # drawing ref line
  cv2.line(frame,(580,360),(800,360),(255,0,0),2)
  cv2.line(frame,(580,400),(800,400),(0,0,255),1)# offset above
  cv2.line(frame,(580,320),(800,320),(0,0,255),1)# offset below  
  # extracting contours
  conts,_=cv2.findContours(thresh_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
  for c in conts:
      # ignore all small contours
      if cv2.contourArea(c)<100:
          continue
      x,y,w,h=cv2.boundingRect(c)
      if ( y>250 and y<450) and (x>500 and x<850):
       cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
       x_center=int((x+(x+w))/2)
       y_center=int((y+(y+h))/2)
       cv2.circle(frame,(x_center,y_center),5,(0,0,255),5)
       if ( y_center>390 and y_center<400) and (x_center>580 and x_center<800):
          vehicle += 1
  cv2.imshow('manual_mask',thresh_img)        
  cv2.putText(frame,"No. of vehicles in the given area:{}".format(vehicle),(360,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
  out.write(frame)
  cv2.imshow('original',frame)      
  if cv2.waitKey(1) & 0xff==ord('q'):
      break

cv2.destroyAllWindows()
cap.release()
out.release()

    

 
  
    
  
    