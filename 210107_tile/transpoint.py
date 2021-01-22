import numpy as np
import cv2

dx,dy = 400,400
centre = dx//2,dy//2
img = np.zeros((dy,dx),np.uint8)

# construct a long thin triangle with the apex at the centre of the image
polygon = np.array([(0,0),(100,10),(100,-10)],np.int32)
polygon += np.int32(centre)

# draw the filled-in polygon and then rotate the image
cv2.fillConvexPoly(img,polygon,(255))
M = cv2.getRotationMatrix2D(centre,20,1) # M.shape =  (2, 3)
rotatedimage = cv2.warpAffine(img,M,img.shape)

# as an alternative, rotate the polygon first and then draw it

# these are alternative ways of coding the working example
# polygon.shape is 3,2

# magic that makes sense if one understands numpy arrays
poly1 = np.reshape(polygon,(3,1,2))
# slightly more accurate code that doesn't assumy the polygon is a triangle
poly2 = np.reshape(polygon,(polygon.shape[0],1,2))
# turn each point into an array of points
poly3 = np.array([[p] for p in polygon])
# use an array of array of points
poly4 = np.array([polygon])
# more magic
poly5 = np.reshape(polygon,(1,3,2))

for poly in (poly1,poly2,poly3,poly4,poly5):
    newimg = np.zeros((dy,dx),np.uint8)
    rotatedpolygon = cv2.transform(poly,M)
    cv2.fillConvexPoly(newimg,rotatedpolygon,(127))

    cv2.imshow("win",newimg)
    cv2.waitKey(0)
cv2.destroyWindow("win")