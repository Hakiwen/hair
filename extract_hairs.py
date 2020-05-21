import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt


refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(img_color, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", img_color)



img_color = cv2.imread('IMG_0048.JPG')
clone = img_color.copy()
cv2.namedWindow("image")
cv2.resizeWindow("image", 1920,1080)
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", img_color)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		img_color = clone.copy()
 
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break
# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)
 
# close all open windows
cv2.destroyAllWindows()


# img = cv2.medianBlur(img,5)

# ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)

# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]

# for i in xrange(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()


# sharpen
blurred = cv2.blur(roi, (3,3))
roi = cv2.addWeighted(roi, 1.5, blurred, -0.5, 0)


img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)



laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=31)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=31)

direction = cv2.divide(sobely, sobelx)
prodX = cv2.multiply(sobelx, sobelx)
prodY = cv2.multiply(sobely, sobely)
sumXY = prodX + prodY
sumXY = cv2.sqrt(sumXY)

# plt.imshow(sumXY,cmap='gray')
sumXY = sumXY.astype(np.uint8)




edges = cv2.Canny(img, 240,240,apertureSize=5)

img_lines = roi.copy()

# thx = ithx
# thy = ithy


# for i in range(40,400,40):
	# for j in range(40,400,40):
		# edgesx = cv2.Canny(img,i,j,apertureSize = 5)
		# plt.subplot(9,9,9*(i/40 - 1) + j/40), plt.imshow(edgesx,cmap='gray')
		# plt.title(str(i) + ',' + str(j))

lines = cv2.HoughLinesP(edges,1,np.pi/360, threshold=100, minLineLength=5, maxLineGap=0)


if lines is not None:
	for line in lines:
		x1,y1,x2,y2 = line[0]
		cv2.line(img_lines,(x1,y1),(x2,y2),(0,255,0),2)



plt.subplot(3,1,1), plt.imshow(roi)
plt.subplot(3,1,2), plt.imshow(edges, cmap='gray')
plt.subplot(3,1,3), plt.imshow(img_lines)

# plt.imshow(img_lines)

plt.show()
