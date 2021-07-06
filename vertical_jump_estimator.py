import numpy as np
import tensorflow as tf
import cv2
import time
import matplotlib.pyplot as plt
import argparse

# Extract time series data from video clip #################################################################

# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

def rotate(img, scaleFactor, degreesCCW):
    (oldY,oldX) = img.shape[:2] #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.

    #choose a new image size.
    newX,newY = oldX*scaleFactor,oldY*scaleFactor
    #include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

    #the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    #So I will find the translation that moves the result to the center of that region.
    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
    M[1,2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
    return rotatedImg

def extractData(mp4file):
    """
    Extract the feet and head time series data from a video
    
    Arguments:
    mp4file: string containing the name of the video file
    
    Returns:
    feet: array containing the feet y coordinate time series data
    head: array containing the head y coordinate time series data
    """
    
    # neural net parameters
    model_path = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.3 # confidence threshold
    
    # capture each frame of the video
    cap = cv2.VideoCapture(mp4file)

    feet = [] # store y coordinates in pixels of feet of each frame
    head = [] # store y coordinate in pixels of head of each frame
    frame = 0 # frame counter

    while True:
        r, img = cap.read() # get the next frame
        if r == False: # no more frames in the video
            break
        else:
            img = cv2.resize(img, (1200, 700)) # resize to fit on screen
            img = rotate(img, 0.9, 270) # rotate image
            boxes, scores, classes, num = odapi.processFrame(img) # run the neural network

            boxScores = [] # confidence scores for all human classifications
            
            # Find all the classifications of humans above the confidence threshold.  Class 1 represents human
            for i in range(len(boxes)):
                if classes[i] == 1 and scores[i] > threshold:
                    boxScores += [scores[i]]

            # if we found at least one human, use the classification with the highest confidence score, and extract the
            # positions.  Print info about the detection
            if len(boxScores) > 0:        
                ind = boxScores.index(max(boxScores))
                box = boxes[ind]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                print("Score: ", scores[ind])
                print("frame: ", frame)
                print("")
                
                feet += [box[2]] # lower y coordinate of box
                head += [box[0]] # upper y coordinate of box
            
            # if there were no humans found, insert NONE for this frame
            elif len(boxScores) == 0:
                feet += [None]
                head += [None]
                
            # increment frame
            frame += 1
            
            # Visualize the results of the detection in real-time
            cv2.imshow("preview", img)
            
            # user can press 'q' to stop the process
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
    
    # stop showing the visualization
    cap.release()
    cv2.destroyAllWindows()
    
    return feet, head

# Plotting functions #######################################################################################

def plotData(feet, head):
    """
    Plot the time series feet and head data
    
    Arguments:
    feet: time series feet data
    head: time series head data
    
    Returns:
    none
    """

    # Mask the NaN values for frames without boxes
    xs = np.arange(len(feet))
    feetNP = np.array(feet).astype(np.double)
    feetMask = np.isfinite(feetNP)
    headNP = np.array(head).astype(np.double)
    headMask = np.isfinite(headNP)

    fig = plt.figure(figsize=(14, 6), dpi= 80) # size the plot
    plt.plot(xs[feetMask], feetNP[feetMask], linestyle = 'none', marker='o', label='feet')
    plt.plot(xs[headMask], headNP[headMask], linestyle = 'none',marker='o', label='head')
    plt.legend(loc='upper right')
    plt.ylabel('Distance From Top Of Frame (Pixels)')
    plt.xlabel('Frame Number')
    plt.show()

def plotEstimate(feet, f0, mu0, h, k, f1, mu1, length):
    """
    Plot the original data with the estimate overlayed
    
    Arguments:
    feet:  original data
    f0:  x coordinate of takeoff
    mu0:  y coordinate of takeoff
    h:  x coordinate of apex
    k:  y coordinate of apex
    f1:  x coordinate of landing
    mu1:  y coordinate of landing
    
    Returns:
    None
    """
    
    # Original data
    # Mask the NaN values for frames without boxes
    xs = np.arange(len(feet))
    feetNP = np.array(feet).astype(np.double)
    feetMask = np.isfinite(feetNP)
    
    # estimate
    xest = np.arange(length)
    yest = np.zeros(length)
    yest[0:f0] = mu0
    yp = createParabola(f0, mu0, h, k, f1, mu1)
    yest[f0:f1+1] = yp
    yest[f1:] = mu1

    fig = plt.figure(figsize=(14, 6), dpi= 80) # size the plot
    plt.plot(xs[feetMask], feetNP[feetMask], linestyle = 'none', marker='o', label='feet')
    plt.plot(xest, yest, marker='_', color='black', label='estimate')
    plt.legend(loc='upper right')
    plt.ylabel('Distance From Top Of Frame (Pixels)')
    plt.xlabel('Frame Number')
    plt.show()

# Helper functions #########################################################################################

def createParabola(f0, mu0, h, k, f1, mu1):
    """
    A parabola can be uniquely defined by 3 unique points.  Create that parabola.
    Adapted from http://chris35wills.github.io/parabola_python/
    
    Arguments:
    f0:  x coordinate of first point
    mu0:  y coordinate of first point
    h:  x coordinate of second point
    k:  y coordinate of second point
    f1:  x coordinate of third point
    mu1:  y coordinate of third point
    
    Returns:
    y:  The unique parabola defined by (f0, mu0), (h, k), (f1, mu1)
    """
    
    # use the given three points to find A, B, C that satisfies y=ax^2+bx+c
    denom = (f0-h) * (f0-f1) * (h-f1)
    A     = (f1 * (k-mu0) + h * (mu0-mu1) + f0 * (mu1-k)) / denom
    B     = (f1*f1 * (mu0-k) + h*h * (mu1-mu0) + f0*f0 * (k-mu1)) / denom
    C     = (h * f1 * (h-f1) * mu0+f1 * f0 * (f1-f0) * k+f0 * h * (f0-h) * mu1) / denom
    
    # now apply the equation to calculate values along the parabola between f0 and f1
    x = np.arange(f0, f1+1)
    y = np.zeros(f1-f0+1)
    y = (A*(x**2))+(B*x)+C
    
    return y

def createYhat(f0, mu0, h, k, f1, mu1, length):
    """
    create the piecewise estimate function for the jump given 6 parameters
    
    Arguments:
    f0:  x coordinate of takeoff point
    mu0:  pre-jump y coordinate mean
    h:  x coordinate of apex of jump
    k:  y coordinate of apex of jump
    f1:  x coordinate of landing point
    mu1:  post-jump y coordinate mean
    length:  total number of points
    
    Returns:
    yhat: the piecewise estimate function
    """
    
    yhat = np.zeros(length)
    
    yhat[0:f0] = mu0
    yp = createParabola(f0, mu0, h, k, f1, mu1)
    yhat[f0:f1+1] = yp
    yhat[f1:] = mu1
    
    return yhat

def calcError(f0, f1, h, k, length, feet):
    """
    Calculate the error between the model given by f0, f1 and the data.
    
    Arguments:
    f0: frame position of takeoff
    f1: frame position of landing
    h: frame of apex of jump
    k: vertical coordinate of apex of jump
    length: length of feet array
    feet: feet time series data
    
    Returns:
    error: the sum of squared residuals between the model and the data
    """
    
    # create model parameters
    mu0 = np.average(feet[0:f0+1])
    mu1 = np.average(feet[f1:])
    yhat = createYhat(f0, mu0, h, k, f1, mu1, length);
    # compute error
    error = np.sum((yhat-feet)**2)
    
    return error

# Maximum Likelihood Estimation (MLE) using gradient descent ###############################################

def mle_gd1(feet, plot):
    """
    Uses the gradient descent to try to improve runtime from the brute force optimization while maintaining the same error
    rate.  We still assume that (h,k) is the minimum value
    
    Arguments:
    feet:  time series feet data
    plot:  whether or not to generate a plot
    
    Returns:
    f0:  frame number at takeoff
    f1:  frame number at landing
    """
    
    length = len(feet) # number of frames
    minJump = 10 # 10 frames in the air is the minimum jump we allow (1.34 inches)
    
    # gradient descent approach to optimization
    minidx = np.argmin(feet)
    h = minidx
    k = feet[minidx]
    errors = np.ones((length, length)) # store all errors in a 2d matrix (rows, cols)
    errors = errors*1e10 # initialize to a high value so that unused elements in the array are not selected
    
    # initialize f0 and f1 close to where we expect f0 and f1 to be
    f0 = h - 50
    f1 = h + 50
    minInd = (0,0) # initialize so we enter the while loop
    while (minInd != (1,1)): # while the current point is not the most optimal
        # create a grid of error values around the current point.  [1,1] is the point we are currently at
        # in the grid, col is f0 frame, row is f1 frame
        # save everything in a larger grid (errors) so we don't need to re-calculate anything
        grid = np.zeros((3,3))
        
        if (errors[f1,f0] == 1e10):
            grid[1,1] = calcError(f0, f1, h, k, length, feet)
            errors[f1,f0] = grid[1,1]
        else:
            grid[1,1] = errors[f1,f0]
            
        if (errors[f1,f0+1] == 1e10):
            grid[1,2] = calcError(f0+1, f1, h, k, length, feet)
            errors[f1,f0+1] = grid[1,2]
        else:
            grid[1,2] = errors[f1,f0+1]
            
        if (errors[f1-1,f0+1] == 1e10):
            grid[2,2] = calcError(f0+1, f1-1, h, k, length, feet)
            errors[f1-1,f0+1] = grid[2,2]
        else:
            grid[2,2] = errors[f1-1,f0+1]
            
        if (errors[f1-1,f0] == 1e10):
            grid[2,1] = calcError(f0, f1-1, h, k, length, feet)
            errors[f1-1,f0] = grid[2,1]
        else:
            grid[2,1] = errors[f1-1,f0]
            
        if (errors[f1-1,f0-1] == 1e10):
            grid[2,0] = calcError(f0-1, f1-1, h, k, length, feet)
            errors[f1-1,f0-1] = grid[2,0]
        else:
            grid[2,0] = errors[f1-1,f0-1]
            
        if (errors[f1,f0-1] == 1e10):
            grid[1,0] = calcError(f0-1, f1, h, k, length, feet)
            errors[f1,f0-1] = grid[1,0]
        else:
            grid[1,0] = errors[f1,f0-1]
            
        if (errors[f1+1,f0-1] == 1e10):
            grid[0,0] = calcError(f0-1, f1+1, h, k, length, feet)
            errors[f1+1,f0-1] = grid[0,0]
        else:
            grid[0,0] = errors[f1+1,f0-1]
            
        if (errors[f1+1,f0] == 1e10):
            grid[0,1] = calcError(f0, f1+1, h, k, length, feet)
            errors[f1+1,f0] = grid[0,1]
        else:
            grid[0,1] = errors[f1+1,f0]
            
        if (errors[f1+1,f0+1] == 1e10):
            grid[0,2] = calcError(f0+1, f1+1, h, k, length, feet)
            errors[f1+1,f0+1] = grid[0,2]
        else:
            grid[0,2] = errors[f1+1,f0+1]
        
        # move in the direction of least error
        minInd = np.unravel_index(np.argmin(grid, axis=None), grid.shape)
        f1 = f1 + (1-minInd[0]) # because a higher index in grid means decreasing f1 frame number
        f0 = f0 + (minInd[1]-1) # because a higher index in grid means an increasing f0 frame number
    
    # plot the results
    if (plot==True):
        mu0 = np.average(feet[0:f0+1])
        mu1 = np.average(feet[f1:])
        plotEstimate(feet, f0, mu0, h, k, f1, mu1, length)
#         plt.imshow(errors)
#         plt.xlabel("f0 frame number")
#         plt.ylabel("f1 frame number")
#         plt.ylim(100,190)
    
    return f0, f1

def gd3_hk(feet, f0, f1):
    """
    helper function for mle_gd3 that finds the optimal h,k position given an f0 and f1 position
    
    Argruments:
    feet: time series observations
    f0: take-off frame
    f1: landing frame
    
    Returns:
    h, k: the optimal vertex point
    """
    
    length = len(feet) # number of frames
    kmax = np.average(feet[0:f0+1]).astype(int) # the maximum value for k, but k should be much less
    
    # gradient descent approach to optimization
    errors = np.ones((kmax, length)) # store all errors in a 2d matrix (rows, cols).  Rows are for k, cols are for h
    errors = errors*1e10 # initialize to a high value so that unused elements in the array are not selected
    
    # initialize h and k with the minimum value assumption
    minidx = np.argmin(feet)
    h = minidx
    k = feet[minidx]
    
    minInd = (0,0) # so we enter the while loop
    while (minInd != (1,1)): # while the current point is not the most optimal
        # create a grid of error values around the current point.  [1,1] is the point we are currently at
        # in the grid, col is f0 frame, row is f1 frame
        # save everything in a larger grid (errors) so we don't need to re-calculate anything
        grid = np.zeros((3,3))
        
        if (errors[k,h] == 1e10):
            grid[1,1] = calcError(f0, f1, h, k, length, feet)
            errors[k,h] = grid[1,1]
        else:
            grid[1,1] = errors[k,h]
            
        if (errors[k,h+1] == 1e10):
            grid[1,2] = calcError(f0, f1, h+1, k, length, feet)
            errors[k,h+1] = grid[1,2]
        else:
            grid[1,2] = errors[k,h+1]
            
        if (errors[k-1,h+1] == 1e10):
            grid[2,2] = calcError(f0, f1, h+1, k-1, length, feet)
            errors[k-1,h+1] = grid[2,2]
        else:
            grid[2,2] = errors[k-1,h+1]
            
        if (errors[k-1,h] == 1e10):
            grid[2,1] = calcError(f0, f1, h, k-1, length, feet)
            errors[k-1,h] = grid[2,1]
        else:
            grid[2,1] = errors[k-1,h]
            
        if (errors[k-1,h-1] == 1e10):
            grid[2,0] = calcError(f0, f1, h-1, k-1, length, feet)
            errors[k-1,h-1] = grid[2,0]
        else:
            grid[2,0] = errors[k-1,h-1]
            
        if (errors[k,h-1] == 1e10):
            grid[1,0] = calcError(f0, f1, h-1, k, length, feet)
            errors[k,h-1] = grid[1,0]
        else:
            grid[1,0] = errors[k,h-1]
            
        if (errors[k+1,h-1] == 1e10):
            grid[0,0] = calcError(f0, f1, h-1, k+1, length, feet)
            errors[k+1,h-1] = grid[0,0]
        else:
            grid[0,0] = errors[k+1,h-1]
            
        if (errors[k+1,h] == 1e10):
            grid[0,1] = calcError(f0, f1, h, k+1, length, feet)
            errors[k+1,h] = grid[0,1]
        else:
            grid[0,1] = errors[k+1,h]
            
        if (errors[k+1,h+1] == 1e10):
            grid[0,2] = calcError(f0, f1, h+1, k+1, length, feet)
            errors[k+1,h+1] = grid[0,2]
        else:
            grid[0,2] = errors[k+1,h+1]
        
        # move in the direction of least error
        minInd = np.unravel_index(np.argmin(grid, axis=None), grid.shape)
        k = k + (1-minInd[0]) # because a higher index in grid means decreasing k
        h = h + (minInd[1]-1) # because a higher index in grid means an increasing h frame number
    
    
    return h, k

def mle_gd3(feet, plot):
    """
    Uses gradient descent similar to mle_gd1, but finds the optimal h,k position using another gradient descent.  Does not
    assume that (h,k) is the minimum value.  It calle mle_gd1 to get a good initial guess of f0 and f1, then refines the
    estimate using the double gradient descent approach
    
    Arguments:
    feet:  time series feet data
    plot:  whether or not to generate a plot
    
    Returns:
    f0:  frame number at takeoff
    f1:  frame number at landing
    """
    
    length = len(feet) # number of frames
    minJump = 10 # 10 frames in the air is the minimum jump we allow (1.34 inches)
    
    # gradient descent approach to optimization
    errors = np.ones((length, length)) # store all errors in a 2d matrix (rows, cols)
    errors = errors*1e10 # initialize to a high value so that unused elements in the array are not selected
    
    # initialize f0 and f1 by calling the version of gradient descent that makes the assumption
    f0, f1 = mle_gd1(feet, plot=False)
    
    # now refine the estimate
    minInd = (0,0) # so we enter the while loop
    while (minInd != (1,1)): # while the current point is not the most optimal
        # create a grid of error values around the current point.  [1,1] is the point we are currently at
        # in the grid, col is f0 frame, row is f1 frame
        # save everything in a larger grid (errors) so we don't need to re-calculate anything
        grid = np.zeros((3,3))
        
        if (errors[f1,f0] == 1e10):
            h, k = gd3_hk(feet, f0, f1)
            grid[1,1] = calcError(f0, f1, h, k, length, feet)
            errors[f1,f0] = grid[1,1]
        else:
            grid[1,1] = errors[f1,f0]
            
        if (errors[f1,f0+1] == 1e10):
            h, k = gd3_hk(feet, f0+1, f1)
            grid[1,2] = calcError(f0+1, f1, h, k, length, feet)
            errors[f1,f0+1] = grid[1,2]
        else:
            grid[1,2] = errors[f1,f0+1]
            
        if (errors[f1-1,f0+1] == 1e10):
            h, k = gd3_hk(feet, f0+1, f1-1)
            grid[2,2] = calcError(f0+1, f1-1, h, k, length, feet)
            errors[f1-1,f0+1] = grid[2,2]
        else:
            grid[2,2] = errors[f1-1,f0+1]
            
        if (errors[f1-1,f0] == 1e10):
            h, k = gd3_hk(feet, f0, f1-1)
            grid[2,1] = calcError(f0, f1-1, h, k, length, feet)
            errors[f1-1,f0] = grid[2,1]
        else:
            grid[2,1] = errors[f1-1,f0]
            
        if (errors[f1-1,f0-1] == 1e10):
            h, k = gd3_hk(feet, f0-1, f1-1)
            grid[2,0] = calcError(f0-1, f1-1, h, k, length, feet)
            errors[f1-1,f0-1] = grid[2,0]
        else:
            grid[2,0] = errors[f1-1,f0-1]
            
        if (errors[f1,f0-1] == 1e10):
            h, k = gd3_hk(feet, f0-1, f1)
            grid[1,0] = calcError(f0-1, f1, h, k, length, feet)
            errors[f1,f0-1] = grid[1,0]
        else:
            grid[1,0] = errors[f1,f0-1]
            
        if (errors[f1+1,f0-1] == 1e10):
            h, k = gd3_hk(feet, f0-1, f1+1)
            grid[0,0] = calcError(f0-1, f1+1, h, k, length, feet)
            errors[f1+1,f0-1] = grid[0,0]
        else:
            grid[0,0] = errors[f1+1,f0-1]
            
        if (errors[f1+1,f0] == 1e10):
            h, k = gd3_hk(feet, f0, f1+1)
            grid[0,1] = calcError(f0, f1+1, h, k, length, feet)
            errors[f1+1,f0] = grid[0,1]
        else:
            grid[0,1] = errors[f1+1,f0]
            
        if (errors[f1+1,f0+1] == 1e10):
            h, k = gd3_hk(feet, f0+1, f1+1)
            grid[0,2] = calcError(f0+1, f1+1, h, k, length, feet)
            errors[f1+1,f0+1] = grid[0,2]
        else:
            grid[0,2] = errors[f1+1,f0+1]
        
        # move in the direction of least error
        minInd = np.unravel_index(np.argmin(grid, axis=None), grid.shape)
        f1 = f1 + (1-minInd[0]) # because a higher index in grid means decreasing f1 frame number
        f0 = f0 + (minInd[1]-1) # because a higher index in grid means an increasing f0 frame number
    
    # plot the results
    if (plot==True):
        mu0 = np.average(feet[0:f0+1])
        mu1 = np.average(feet[f1:])
        plotEstimate(feet, f0, mu0, h, k, f1, mu1, length)
    
    return f0, f1

# Putting it all together #################################################################################

def calcHeight(mp4file, frameRate=30, plot=False):
    """
    Computes the vertical jump height for a given video file
    
    Arguments:
    mp4file:  String of the mp4 video file
    plot:  whether or not to generate a plot
    
    Returns:
    height:  height of the jump in inches
    """
    
    g = 386.09 # acceleration of gravity in inches/second**2
    
    feet, head = extractData(mp4file)
    f0, f1 = mle_gd3(feet, plot) # perform mle with enhanced gradient descent method
    t0 = f0/frameRate # convert from frame to time
    t1 = f1/frameRate
    time = t1 - t0 # time in the air
    height = 0.5*g*(time/2)**2 # kinematics equation to find height
    
    return height

# Main function ###########################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute jump height from an mp4 video file")
    parser.add_argument("mp4file", help="relative path to mp4 video file")
    parser.add_argument("frameRate", type=int, help="frame rate of the video file in frames/sec")
    args = parser.parse_args()

    height = calcHeight(args.mp4file, args.frameRate, plot=True)
    print("Height: %f inches" % height)
