import sys
import argparse
import numpy as np
import cv2
import vpi
from math import sin, cos, pi, atan2, radians
from contextlib import contextmanager
from os import path
from jetson_utils import (cudaDrawCircle, cudaDrawLine, cudaResize, cudaDrawRect, videoSource, videoOutput, loadImage, cudaAllocMapped, cudaConvertColor, 
                          cudaDeviceSynchronize, cudaToNumpy, cudaFromNumpy, cudaAllocMapped)
def update_mask(mask, trackColors, prevFeatures, curFeatures, status = None):
    '''Draw keypoint path from previous frame to current one'''

    numTrackedKeypoints = 0

    def none_context(a=None): return contextmanager(lambda: (x for x in [a]))()

    with curFeatures.rlock_cpu(), \
        (status.rlock_cpu() if status else none_context()), \
        (prevFeatures.rlock_cpu() if prevFeatures else none_context()):

        for i in range(curFeatures.size):
            # keypoint is being tracked?
            if not status or status.cpu()[i] == 0:
                color = tuple(trackColors[i,0].tolist())

                # OpenCV 4.5+ wants integers in the tuple arguments below
                cf = tuple(np.round(curFeatures.cpu()[i]).astype(int))

                # draw the tracks
                if prevFeatures:
                    pf = tuple(np.round(prevFeatures.cpu()[i]).astype(int))
                    cv2.line(mask, pf, cf, color, 2)

                cv2.circle(mask, cf, 5, color, -1)

                numTrackedKeypoints += 1

    return numTrackedKeypoints
def save_file_to_disk(frame, mask, baseFileName, frameCounter):
    '''Apply mask on frame and save it to disk'''
  
    frame = frame.convert(vpi.Format.BGR8, backend=vpi.Backend.CUDA)
    with frame.rlock_cpu() as frameData:
        frame = cv2.add(frameData, mask)
  
    name, ext = path.splitext(baseFileName)
    fname = "{}_{:04d}{}".format(name, frameCounter, ext)
  
     #cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    cv2.imshow("flow",frame)
    cv2.waitKey(1)
MAX_KEYPOINTS = 100
#################### USER VARS ######################################
# Decreases stabilization latency at the expense of accuracy. Set to 1 if no downsamping is desired. 
# Example: downSample = 0.5 is half resolution and runs faster but gets jittery
downSample = 0.8

#Zoom in so you don't see the frame bouncing around. zoomFactor = 1 for no zoom
zoomFactor = 0.9

# pV and mV can be increased for more smoothing #### start with pV = 0.01 and mV = 2 
processVar=0.03
measVar=2

# set to 1 to display full screen -- doesn't actually go full screen if your monitor rez is higher than stream rez which it probably is. TODO: monitor resolution detection
showFullScreen = 1

# If test video plays too fast then increase this until it looks close enough. Varies with hardware. 
# LEAVE AT 1 if streaming live video from WFB (unless you like a delay in your stream for some weird reason)
delay_time = 1 


######################## Region of Interest (ROI) ###############################
# This is the portion of the frame actually being processed. Smaller ROI = faster processing = less latency
#
# roiDiv = ROI size divisor. Minimum functional divisor is about 3.0 at 720p input. 4.0 is best for solid stabilization.
# Higher FPS and lower resolution can go higher in ROI (and probably should)
# Set showrectROI and/or showUnstabilized vars to = 1 to see the area being processed. On slower PC's 3 might be required if 720p input
roiDiv = 3.0

# set to 1 to show the ROI rectangle 
showrectROI = 0

#showTrackingPoints # show tracking points found in frame. Useful to turn this on for troubleshooting or just for funzies. 
showTrackingPoints = 0

# set to 1 to show unstabilized B&W ROI in a window
showUnstabilized = 1

# maskFrame # Wide angle camera with stabilization warps things at extreme edges of frame. This helps mask them without zoom. 
# Feels more like a windshield. Set to 0 to disable or find the variable down in the code to adjust size of mask
maskFrame = 0

# parse command line
parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
args = parser.parse_known_args()[0]

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)    # default:  options={'width': 1280, 'height': 720, 'framerate': 30}
output = videoOutput(args.output, argv=sys.argv)  # default:  options={'codec': 'h264', 'bitrate': 4000000}

count = 0
a = 0
x = 0
y = 0
Q = np.array([[processVar]*3])
R = np.array([[measVar]*3])
K_collect = []
P_collect = []
prevFrame = None

x_prev = 0
y_prev = 0
x_curr = 0
y_curr = 0

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return (np.rad2deg((ang1 - ang2) % (2 * np.pi)))
currFrame = 0

image = input.Capture(format='rgb8', timeout=1000)
# Retrieve features to be tracked from first frame using
# Harris Corners Detector
with vpi.Backend.CPU:
    frame = vpi.asimage(np.uint8(cudaToNumpy(image)), vpi.Format.BGR8).convert(vpi.Format.U8)
    curFeatures, scores = frame.harriscorners(strength=0.1, sensitivity=0.01)
    curFeatures1 = curFeatures
# Limit the number of features we'll track and calculate their colors on the
# output image
with curFeatures.lock_cpu() as featData, scores.rlock_cpu() as scoresData:
    # Sort features in descending scores order and keep the first MAX_KEYPOINTS
    ind = np.argsort(scoresData, kind='mergesort')[::-1]
    featData[:] = np.take(featData, ind, axis=0)
    curFeatures.size = min(curFeatures.size, MAX_KEYPOINTS)

    # Keypoints' have different hues, calculated from their position in the first frame
    trackColors = np.array([[(int(p[0]) ^ int(p[1])) % 180,255,255] for p in featData], np.uint8).reshape(-1,1,3)
    prevPts = np.array([(int(p[0]), int(p[1])) for p in featData]).reshape(-1,1,2)
    # Convert colors from HSV to RGB
    #trackColors = cv2.cvtColor(trackColors, cv2.COLOR_HSV2BGR).astype(int)

# Counter for the frames
idFrame = 0
# Create mask with features' tracks over time
mask = np.zeros((frame.height, frame.width, 3), np.uint8)
numTrackedKeypoints = update_mask(mask, trackColors, None, curFeatures)


with vpi.Backend.CUDA:
     optflow = vpi.OpticalFlowPyrLK(frame, curFeatures, 2)
     #optflow1 = vpi.OpticalFlowPyrLK(frame, curFeatures1, 2)
# capture frames until end-of-stream (or the user exits)
while True:
    # Apply mask to frame and save it to disk
    save_file_to_disk(frame, mask, "image.png", idFrame)
  
    print("Frame id={}: {} points tracked.".format(idFrame, numTrackedKeypoints))
    # format can be:   rgb8, rgba8, rgb32f, rgba32f (rgb8 is the default)
    # timeout can be:  -1 for infinite timeout (blocking), 0 to return immediately, >0 in milliseconds (default is 1000ms)
    image = input.Capture(format='rgb8', timeout=1000)
    currFrame = currFrame +1
    res_w_orig = image.shape[1]
    res_h_orig = image.shape[0]
    res_w = int(res_w_orig * downSample)
    res_h = int(res_h_orig * downSample)
    top_left= [int(res_h/roiDiv),int(res_w/roiDiv)]
    bottom_right = [int(res_h - (res_h/roiDiv)),int(res_w - (res_w/roiDiv))]
    #cudaDrawRect(image, (top_left[0],top_left[1],bottom_right[0],bottom_right[1]), (255,127,0,100))
    #cudaDrawLine(image, (top_left[1],top_left[0]), (bottom_right[1],bottom_right[0]), (255,0,200,200), 10)
    
    frameSize = (res_w,res_h)
    
    if downSample != 1:
        imgOutput = cudaAllocMapped(width=image.width * downSample, 
                                         height=image.height * downSample, 
                                         format=image.format)
        cudaResize(image, imgOutput)
    converted_img = cudaAllocMapped(width=imgOutput.width, height=imgOutput.height, format="gray8")
    cudaConvertColor(imgOutput, converted_img)
    prevFeatures = curFeatures
    with vpi.Backend.CUDA:
        input2 = vpi.asimage(np.uint8(cudaToNumpy(image)))
        input1 = vpi.asimage(np.uint8(cudaToNumpy(image)), vpi.Format.BGR8).convert(vpi.Format.S16)
        corners, scores2 = input1.harriscorners(strength=0.1, sensitivity=0.01)
        frame = vpi.asimage(np.uint8(cudaToNumpy(image)), vpi.Format.BGR8).convert(vpi.Format.U8)
        curFeatures1, scores1 = frame.harriscorners(strength=0.1, sensitivity=0.01)
        optflow1 = vpi.OpticalFlowPyrLK(frame, curFeatures1, 1)
        curFeatures1, status1 = optflow1(frame)
        #optflow = vpi.OpticalFlowPyrLK(frame, corners, 2)
        if corners.size > 1:
            with corners.lock_cpu() as corners_data:
                x_sum = 0
                y_sum = 0
                
                for i in range(corners.size):
                    kpt = tuple(corners_data[i].astype(np.int16))
                    #print(kpt)
                    x_sum = x_sum + kpt[0]
                    y_sum = y_sum + kpt[1]
                    cudaDrawCircle(image, kpt, 5, (0,255,127,200))
                x_curr = int(x_sum/corners.size)
                y_curr = int(y_sum/corners.size)
                cudaDrawCircle(image, (x_curr,y_curr), 30, (0,255,127,200))
                cudaDrawCircle(image, (int(x_prev),int(y_prev)), 30, (200,255,127,200))   
                cudaDrawLine(image, (int(x_prev),int(y_prev)), (x_curr,y_curr), (255,0,200,200), 10)
                
                #print(atan2(y_curr-y_prev,x_curr-x_prev))
                
                #print(int(x_prev), int(y_prev))
                # Move image's center to origin of coordinate system
                T1 = np.array([[1, 0, -image.width/2.0],
                            [0, 1, -image.height/2.0],
                            [0, 0, 1]])

                # Apply some time-dependent perspective transform
                v1 = 0#sin(atan2(y_curr-y_prev,x_curr-x_prev))#/30.0*2*pi/2)*0.0005
                v2 = cos(1/30.0*2*pi/3)*0.0005
                P = np.array([[0.99, -0.0, 0],
                            [0.0, 0.99, 0],
                            [v1, v2, 1]])

                # Move image's center back to where it was
                T2 = np.array([[1, 0, image.width/2.0],
                            [0, 1, image.height/2.0],
                            [0, 0, 1]])
                input2 = input2.perspwarp(np.matmul(T2, np.matmul(P, T1)))
                image = cudaFromNumpy(input2.cpu())
                bgr_img = cudaAllocMapped(width=image.width,
                          height=image.height,
						  format='bgr8')
                cudaConvertColor(image, bgr_img)
                cv_img = cudaToNumpy(bgr_img)
                cv2.imshow("asd",cv_img)
                cv2.waitKey(1)
                #print(v1)
                x_prev, y_prev = (x_curr,y_curr)
        # Limit the number of features we'll track and calculate their colors on the
        # output image
        with curFeatures1.lock_cpu() as featData1, scores1.rlock_cpu() as scoresData1, prevFeatures.lock_cpu():
            # Sort features in descending scores order and keep the first MAX_KEYPOINTS
            ind1 = np.argsort(scoresData1, kind='mergesort')[::-1]
            featData1[:] = np.take(featData1, ind1, axis=0)

            curFeatures1.size = min(curFeatures1.size, prevFeatures.size)
            if curFeatures1.size < prevFeatures.size:
                prevFeatures.size = curFeatures1.size
            #prevFeatures.size = min(curFeatures1.size, prevFeatures.size)
            # Keypoints' have different hues, calculated from their position in the first frame
            trackColors1 = np.array([[(int(p[0]) ^ int(p[1])) % 180,255,255] for p in featData1], np.uint8).reshape(-1,1,3)
            currPts = np.array([[p[0], p[1]] for p in featData1]).reshape(-1,1,2)
            print(currPts.shape,prevPts.shape)
            # Convert colors from HSV to RGB
            #trackColors1 = cv2.cvtColor(trackColors1, cv2.COLOR_HSV2BGR).astype(int)
            #currPts = featData1
            #prevPts = prevFeatures
            #assert prevPts.shape == currPts.shape
            #idx = np.where(status == 1)[0]
            # Add orig video resolution pts to roi pts
            #prevPts = prevPts[idx] + np.array([int(res_w_orig/roiDiv),int(res_h_orig/roiDiv)]) 
            #currPts = currPts[idx] + np.array([int(res_w_orig/roiDiv),int(res_h_orig/roiDiv)])
            #print(prevPts, currPts)
            #if showTrackingPoints == 1:
            #    for pT in prevPts:
            #        cv2.circle(cv_img, (int(pT[0][0]),int(pT[0][1])) ,5,(211,211,211))
            if prevPts.shape == currPts.shape:
                if prevPts.size & currPts.size:
                    m, inliers = cv2.estimateAffinePartial2D(prevPts, currPts)
                    print(m)
            #if m is None:
            #    m = lastRigidTransform
            idFrame += 1
            prevPts = currPts
            prevFeatures = curFeatures1
            # Update the mask with the current keypoints' position
            #numTrackedKeypoints = update_mask(mask, trackColors1, prevFeatures, curFeatures1, status1)
            #print("numTrackedKeypoints", numTrackedKeypoints)
            # No more keypoints to track?
            #if numTrackedKeypoints == 0:
            #    print("No keypoints to track.")
                #curFeatures, scores = frame.harriscorners(strength=0.1, sensitivity=0.01)#break # nothing else to do
                #optflow = vpi.OpticalFlowPyrLK(frame, curFeatures, 2)
            
        
    if image is None:  # if a timeout occurred
        continue
	
    output.Render(image)

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
