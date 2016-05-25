import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import math
from tesserwrap import Tesseract
from PIL import Image

tr = Tesseract("/usr/local/share")

def auto_canny(image, sigma=0.33):	
	v = np.median(image)	
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)	
	return edged

img = cv2.imread("image.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
threshold = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
wide = cv2.Canny(threshold, 10, 200)
tight = cv2.Canny(threshold, 225, 250)
auto = auto_canny(threshold)
#cv2.imshow('my_image', img)
#cv2.imshow("Edges", np.hstack([wide, tight, auto]))
#cv2.imshow("Wide",wide)
#cv2.imshow("Tight",tight)
#cv2.imshow("Auto",auto)

bin, contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#contours = sorted(contours, key=cv2.contourArea,reverse=True) 
#print(len(contours))

#perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
#listindex=[i for i in range(15) if perimeters[i]>perimeters[0]/2]
#numcards=len(listindex)
listindex = [];

for index, c in enumerate(contours):
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) == 4 and index != 0 and peri < 800:
		screenCnt = approx
		listindex.append(index)
		#print index
		#break	


# Show image
imgcont = img.copy()
[cv2.drawContours(imgcont, [contours[i]], 0, (0,255,0), 5) for i in listindex]
#plt.imshow(imgcont)
#plt.show()
#cv2.waitKey(5000)
#plt.rcParams['figure.figsize'] = (3.0, 3.0)
warp = []
for i in listindex:	
	#print(i)
	# approximate the contour
	#print(i)
	card = contours[i]
	#print(card)
	peri = cv2.arcLength(card, True)
	#print(peri)
	approx = cv2.approxPolyDP(card, 0.02 * peri, True)
	#print(approx)
	rect = cv2.minAreaRect(contours[i])
	r = cv2.boxPoints(rect)
	h = np.array([ [0,0],[99,0],[99,99],[0,99] ],np.float32)
	approx = np.array([item for sublist in approx for item in sublist],np.float32)
	#print(h)
	#print(approx)
	transform = cv2.getPerspectiveTransform(approx,h)
	warp.append(cv2.warpPerspective(img,transform,(100,100)))

warp =  warp[::-1]

#print(warp)
fig = plt.figure(1, (10,10))
grid = ImageGrid(fig, 111,
                nrows_ncols = (17, 10), 
                axes_pad=0.1, 
                aspect=True, 
                )

tsOutput = []
for i in range(0,162):
	#print warp[i]
	image = warp[i]
	(h, w) = image.shape[:2]
	center = (w / 2, h / 2)
	M = cv2.getRotationMatrix2D(center, 270, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h))
	letter=cv2.flip(rotated,1)
	grid[i].imshow(letter)
	im = Image.fromarray(letter)
	text = tr.ocr_image(im).replace("\n","")	
	if text == "":
		text = " "
	tsOutput.append(text)
	#print tr.ocr_image(im),
	#print(tr.ocr_image(im),end = "")

print "".join(tsOutput)
plt.show()                    
cv2.destroyAllWindows()