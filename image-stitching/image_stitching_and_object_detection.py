#importing the libraries
import cv2
import numpy as np

#reading the 2 images to be stitched and converting them to gray
img_ = cv2.imread('E:/assignment/images/1/2.jpg')
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
img = cv2.imread('E:/assignment/images/1/1.jpg')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# finding the keypoints and descriptors with SIFT feature detection algorithm
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

#BFMatcher to match the features common to both images
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Applying ratio test
good = []
for m in matches:
    if m[0].distance < 0.5*m[1].distance:
        good.append(m)
matches = np.asarray(good)

if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

else:
    raise AssertionError("Can't find enough keypoints.")
#performing feature based alignment
dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))

dst[0:img.shape[0], 0:img.shape[1]] = img
cv2.imwrite('E:/assignment/images/1/output.jpg',dst)

###################### Object Detection ##############################

# Initializing the HOG person 
# detector 
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
   
# Reading the Image 
image = cv2.imread('E:/assignment/images/1/output.jpg')  
   
# Detecting all the regions in the  
# Image that has a pedestrians inside it 
(regions, _) = hog.detectMultiScale(image,  
                                    winStride=(4, 4), 
                                    padding=(7, 7), 
                                    scale=1.05) 
   
# Drawing the regions in the Image 
for (x, y, w, h) in regions: 
    cv2.rectangle(image, (x, y),  
                  (x + w, y + h),  
                  (0, 0, 255), 2) 
  
# Showing the output Image 
cv2.imshow("Image", image) 
cv2.imwrite('E:/assignment/images/1/object_detection.jpg', image)
cv2.waitKey(0) 
   
cv2.destroyAllWindows() 
