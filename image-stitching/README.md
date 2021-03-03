## Table of Content
 * [Overview](#overview)
 * [Libraries Used](#libraries-used)
 * [Techniques Used](#techniques-used)


## Overview
<b> This project combines various computer vision techniques to stitch two images and perform object detection on them.</b>

## Libraries Used
  * cv2 
  * numpy 

## Techniques Used
  **Image Stitching**
  * SIFT algorithm - To perform feature detection in both the images.
  * BFMatcher - To match the features common to both the images
  * warpPerspective - To perform feature based alignment

  **Object Detection**
  * HOG person detection - To detect the pedestrians on the stitched image
  * cv2.rectangle - To draw bounding boxes
  
  
