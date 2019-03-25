import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('teste7.jpg',0) # Load in image here - Ensure 8-bit grayscale

#
# final_circles = [] # Stores the final circles that don't go out of bounds
# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0) # Your code
# rows = img.shape[0] # Obtain rows and columns
# cols = img.shape[1]
# circles = np.round(circles[0, :]).astype("int") # Convert to integer
# for (x, y, r) in circles: # For each circle we have detected...
#     if (r <= x <= cols-1-r) and (r <= y <= rows-1-r): # Check if circle is within boundary
#         if r > 20:
#             final_circles.append([x, y, r]) # If it is, add this to our final list
#
# final_circles = np.asarray(final_circles).astype("int")
# print(final_circles)


_,  img = cv2.threshold(img, 110, 255, cv2.AGAST_FEATURE_DETECTOR_THRESHOLD)
img = cv2.medianBlur(img, 5)
cv2.imshow('detected circles', img)
cv2.waitKey(0)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=10, minRadius=5, maxRadius=30) # Your code


for i in circles[0,:]:
# draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
# draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
cv2.imshow('detected circles',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
