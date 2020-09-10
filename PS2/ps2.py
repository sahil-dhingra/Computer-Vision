"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np

import math

def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
        
    
    img = img_in
    
    #Circles
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=0)
    edges = cv2.Canny(img_blur, 20, 50)
    
    circles = cv2.HoughCircles(
        edges, method=cv2.HOUGH_GRADIENT, 
        dp=1, param1=50, param2=20, 
        minDist=20, minRadius=min(radii_range)-2, maxRadius=max(radii_range)+2)
    
    center = None
    if circles is not None:
        a = np.sort(circles, axis=1)[0]
        
        centers = None
        if len(circles[0][:,0])>2:
            for i in range(len(circles[0][:,0])-2):
                if abs(a[i,0] - a[i+1,0])<5 and abs(a[i,0] - a[i+2,0])<5 and abs(a[i,2] - a[i+1,2])<2 and \
                abs(a[i,2] - a[i+2,2])<2 and abs(a[i+1,1] - a[i,1] - a[i+2,1] + a[i+1,1])<10:
                    centers = [[a[i,0], a[i,1]], [a[i+1,0], a[i+1,1]], [a[i+2,0], a[i+2,1]]]
                    center = (int(a[i+1,0]), int(a[i+1,1]))
                    colors = [img[int(centers[0][1]),int(centers[0][0])], img[int(centers[1][1]), \
                                  int(centers[1][0])], img[int(centers[2][1]),int(centers[2][0])]]
                    if colors[0][2] == 255:
                        state = "red"
                    elif colors[2][1] == 255:
                        state = "green"
                    else:
                        state = "yellow"
                else:
                    None
        else:
            None
    else:
        None
    
    
    if center is not None:
        out = (center, state)
    else:
        out = None

    return out
    raise NotImplementedError


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    img = img_in
    img_blur = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=3, sigmaY=3)
    edges = cv2.Canny(img_blur, 300, 500)

    #Lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=30,
                            maxLineGap=4)
    lines = lines.reshape(lines.shape[0], -1)[:6]
        
    center = (int(np.mean(lines[:,[0,2]])), 
              int(np.mean(lines[:,[1,3]])))
    
    return center
    raise NotImplementedError


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    img = img_in
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, ksize=(9,9), sigmaX=1, sigmaY=1)
    edges = cv2.Canny(img_blur, 300, 500)

    #Lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=5, minLineLength=5,
                            maxLineGap=2)
    lines = lines.reshape(lines.shape[0], -1)[:8]
        
    center = (int(np.mean(lines[:,[0,2]])), 
              int(np.mean(lines[:,[1,3]])))

    return center
    raise NotImplementedError


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    img = img_in

    img_blur = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0)
    edges = cv2.Canny(img_blur, 50, 100)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/360, threshold=0)
    
    lines[:,0,1] = lines[:,0,1]*180/np.pi
    
    lines45  = lines[np.argwhere(np.logical_and(lines[:,0,1]>= 44.5,lines[:,0,1]<= 45.5)),0,:][0:6]
    lines135 = lines[np.argwhere(np.logical_and(lines[:,0,1]>= 134.5,lines[:,0,1]<= 135.5)),0,:][0:6]
    
    lines45 = lines45.reshape(lines45.shape[0], -1)
    lines135 = lines135.reshape(lines135.shape[0], -1)
    
    lines45 = cv2.kmeans(lines45[:,0], 2, (1,2), \
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),\
                     attempts = 100, flags=cv2.KMEANS_PP_CENTERS)[2]
    
    lines135 = cv2.kmeans(lines135[:,0], 2, (1,2), \
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),\
                     attempts = 100, flags=cv2.KMEANS_PP_CENTERS)[2]
    
    lines45 = np.array([[lines45[0][0],45],[lines45[1][0],45]])
    lines135 = np.array([[lines135[0][0],135],[lines135[1][0],135]])
    
    lines45[:,1] = lines45[:,1]*np.pi/180
    lines135[:,1] = lines135[:,1]*np.pi/180
    
    a = np.array([[np.cos(lines45[0,1]), np.sin(lines45[0,1])], [np.cos(lines135[0,1]), np.sin(lines135[0,1])],\
                  [np.cos(lines45[1,1]), np.sin(lines45[1,1])], [np.cos(lines135[1,1]), np.sin(lines135[1,1])]])
    b = np.array([lines45[0,0], lines135[0,0], lines45[1,0], lines135[1,0]])
    points = np.array([np.linalg.solve(a[0:2], b[0:2]), np.linalg.solve(a[1:3], b[1:3]), \
                       np.linalg.solve(a[2:4], b[2:4]), np.linalg.solve(a[(0,3),:], [b[0],b[3]])])
    
    center = (int(np.mean(points[:,0])), int(np.mean(points[:,1])))

    return center
    raise NotImplementedError


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    img = img_in

    img_blur = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0)
    edges = cv2.Canny(img_blur, 50, 100)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/360, threshold=0)
    
    lines[:,0,1] = lines[:,0,1]*180/np.pi
    
    lines45  = lines[np.argwhere(np.logical_and(lines[:,0,1]>= 44.5,lines[:,0,1]<= 45.5)),0,:][0:6]
    lines135 = lines[np.argwhere(np.logical_and(lines[:,0,1]>= 134.5,lines[:,0,1]<= 135.5)),0,:][0:6]
    
    lines45 = lines45.reshape(lines45.shape[0], -1)
    lines135 = lines135.reshape(lines135.shape[0], -1)
    
    lines45 = cv2.kmeans(lines45[:,0], 2, (1,2), \
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),\
                     attempts = 100, flags=cv2.KMEANS_PP_CENTERS)[2]
    
    lines135 = cv2.kmeans(lines135[:,0], 2, (1,2), \
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),\
                     attempts = 100, flags=cv2.KMEANS_PP_CENTERS)[2]
    
    lines45 = np.array([[lines45[0][0],45],[lines45[1][0],45]])
    lines135 = np.array([[lines135[0][0],135],[lines135[1][0],135]])
    
    lines45[:,1] = lines45[:,1]*np.pi/180
    lines135[:,1] = lines135[:,1]*np.pi/180
    
    a = np.array([[np.cos(lines45[0,1]), np.sin(lines45[0,1])], [np.cos(lines135[0,1]), np.sin(lines135[0,1])],\
                  [np.cos(lines45[1,1]), np.sin(lines45[1,1])], [np.cos(lines135[1,1]), np.sin(lines135[1,1])]])
    b = np.array([lines45[0,0], lines135[0,0], lines45[1,0], lines135[1,0]])
    points = np.array([np.linalg.solve(a[0:2], b[0:2]), np.linalg.solve(a[1:3], b[1:3]), \
                       np.linalg.solve(a[2:4], b[2:4]), np.linalg.solve(a[(0,3),:], [b[0],b[3]])])
    
    center = (int(np.mean(points[:,0])), int(np.mean(points[:,1])))

    return center
    raise NotImplementedError


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    img = img_in
    img_blur = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=3, sigmaY=3)
    edges = cv2.Canny(img_blur, 300, 500)

    #Lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=30,
                            maxLineGap=2)
    lines = lines.reshape(lines.shape[0], -1)[:4]
    
    circles = cv2.HoughCircles(edges, method=cv2.HOUGH_GRADIENT, dp=1, param1=200, param2=30,
                               minDist=20)
        
    center_lines = np.array((int(np.mean(lines[:,[0,2]])), 
                            int(np.mean(lines[:,[1,3]]))))
    
    center_circles = circles[0][0][0:2]
    
    center = tuple((center_lines+center_circles)/2)
    
    return center
    raise NotImplementedError


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """

    signs = {}
    img_blur = cv2.medianBlur(img_in, 11)
    
    hsv = np.array(cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV), dtype = "float32")
    mask_yellow = cv2.inRange(hsv, (21,0,0), (40, 255, 255))
    mask_orange = cv2.inRange(hsv, (10,50,50), (20, 255, 255))
    mask_red1 = cv2.inRange(hsv, (0,50,50), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (170,50,50), (180, 255, 255))
    mask_black = cv2.inRange(hsv, (170,50,50), (180, 255, 255))

    ## Yellow mask
    mask = cv2.bitwise_or(mask_yellow, mask_yellow)
    target = cv2.bitwise_and(img_blur, img_blur, mask=mask)
    target_blur = cv2.medianBlur(target, 11)
    edges = cv2.Canny(target_blur, 200, 300)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=15, minLineLength=20,
                            maxLineGap=10)
    
    if lines is not None and lines.shape[0]>3:
        lines = lines[:4]
        lines = lines.reshape(lines.shape[0], -1)
        slope = np.array([math.degrees(math.atan((i[3]-i[1])/(i[2]-i[0]))) for i in lines[:4]],
                      dtype="float32")
        center_wrng = np.array((int(np.mean(lines[:,[0,2]])), 
                                int(np.mean(lines[:,[1,3]]))))
        signs['warning'] = (center_wrng[0], center_wrng[1])
    
    edges = cv2.Canny(target_blur, 200, 400)
    circles = cv2.HoughCircles(edges, method=cv2.HOUGH_GRADIENT, dp=1, param1=400, param2=19,
                               minDist=80)
    
    if circles is not None:
        circles = circles[0]
        index = np.where(np.logical_and(circles[:,2]>8, circles[:,2]<20))[0]
        if index.size>0:
            center_ts = circles[index[0]][:2]
            signs['traffic_light'] = (center_ts[0], center_ts[1])

     
    ## Orange mask
    mask = cv2.bitwise_or(mask_orange, mask_orange)
    target = cv2.bitwise_and(img_blur, img_blur, mask=mask)
    target_blur = cv2.medianBlur(target, 11)
    edges = cv2.Canny(target_blur, 300, 600)
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=15, minLineLength=15,
                            maxLineGap=10)
    
    if lines is not None and lines.shape[0]>3:
        lines = lines[:4]
        lines = lines.reshape(lines.shape[0], -1)
        slope = np.array([math.degrees(math.atan((i[3]-i[1])/(i[2]-i[0]))) for i in lines[:4]],
                          dtype="float32")
        center_constr = np.array((int(np.mean(lines[:,[0,2]])), 
                                  int(np.mean(lines[:,[1,3]]))))
        signs['construction'] = (center_constr[0], center_constr[1])

    ## Red mask
    mask = cv2.bitwise_or(mask_red1, mask_red2)
    white_pixels_mask = np.all(img_blur > [100, 100, 100], axis=-1)
    mask[white_pixels_mask] = 0
    target = cv2.bitwise_and(img_blur, img_blur, mask=mask)
    target_blur = cv2.medianBlur(target, 11)
    
    # YIELD Sign   
    edges = cv2.Canny(target_blur, 700,1500)    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/90, threshold=40, minLineLength=30,
                            maxLineGap=2)
    
    if lines is not None and lines.shape[0]>2:
        lines = lines.reshape(lines.shape[0], -1)[:6]
        center_yield = np.array((int(np.mean(lines[:,[0,2]])), 
                                 int(np.mean(lines[:,[1,3]]))))
        signs['yield'] = (center_yield[0], center_yield[1])
        target_blur[center_yield[1]-40:center_yield[1]+70, center_yield[0]-60:center_yield[0]+60] = \
    target_blur[center_yield[1]-40:center_yield[1]+70, center_yield[0]-60:center_yield[0]+60]*0
    
    # STOP and DNE
    edges = cv2.Canny(target_blur, 300,600)
    circles = cv2.HoughCircles(edges, method=cv2.HOUGH_GRADIENT, dp=1, param1=600, param2=20,
                               minDist=80)

    if circles is not None:
        circles = circles[0]
        index_stp = np.where(np.logical_and(circles[:,2]>45, circles[:,2]<60))[0]
        index_dne = np.where(np.logical_and(circles[:,2]>25, circles[:,2]<40))[0]
        if index_stp.size>0:
            center_stp = circles[index_stp[0]][:2]
            signs['stop'] = (center_stp[0], center_stp[1])
        if index_dne.size>0:
            center_dne = circles[index_dne[0]][:2]
            signs['no_entry'] = (center_dne[0], center_dne[1])

    return signs
    raise NotImplementedError


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    signs = {}
    img_blur = cv2.medianBlur(img_in, 11)
    
    hsv = np.array(cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV), dtype = "float32")
    mask_yellow = cv2.inRange(hsv, (21,0,0), (40, 255, 255))
    mask_orange = cv2.inRange(hsv, (10,50,50), (20, 255, 255))
    mask_red1 = cv2.inRange(hsv, (0,50,50), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (170,50,50), (180, 255, 255))
    mask_black = cv2.inRange(hsv, (170,50,50), (180, 255, 255))

    ## Yellow mask
    mask = cv2.bitwise_or(mask_yellow, mask_yellow)
    target = cv2.bitwise_and(img_blur, img_blur, mask=mask)
    target_blur = cv2.medianBlur(target, 11)
    edges = cv2.Canny(target_blur, 200, 300)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=15, minLineLength=20,
                            maxLineGap=10)
    
    if lines is not None:
        lines = lines[:4]
        lines = lines.reshape(lines.shape[0], -1)
        slope = np.array([math.degrees(math.atan((i[3]-i[1])/(i[2]-i[0]))) for i in lines[:4]],
                      dtype="float32")
        center_wrng = np.array((int(np.mean(lines[:,[0,2]])), 
                                int(np.mean(lines[:,[1,3]]))))
        signs['warning'] = (center_wrng[0], center_wrng[1])
    
    edges = cv2.Canny(target_blur, 200, 400)
    circles = cv2.HoughCircles(edges, method=cv2.HOUGH_GRADIENT, dp=1, param1=400, param2=19,
                               minDist=80)
    
    if circles is not None:
        circles = circles[0]
        index = np.where(np.logical_and(circles[:,2]>8, circles[:,2]<20))[0]
        if index.size>0:
            center_ts = circles[index[0]][:2]
            signs['traffic_light'] = (center_ts[0], center_ts[1])

     
    ## Orange mask
    mask = cv2.bitwise_or(mask_orange, mask_orange)
    target = cv2.bitwise_and(img_blur, img_blur, mask=mask)
    target_blur = cv2.medianBlur(target, 11)
    edges = cv2.Canny(target_blur, 300, 600)
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=15, minLineLength=15,
                            maxLineGap=10)
    
    if lines is not None:
        lines = lines[:4]
        lines = lines.reshape(lines.shape[0], -1)
        slope = np.array([math.degrees(math.atan((i[3]-i[1])/(i[2]-i[0]))) for i in lines[:4]],
                          dtype="float32")
        center_constr = np.array((int(np.mean(lines[:,[0,2]])), 
                                  int(np.mean(lines[:,[1,3]]))))
        signs['construction'] = (center_constr[0], center_constr[1])

    ## Red mask
    mask = cv2.bitwise_or(mask_red1, mask_red2)
    white_pixels_mask = np.all(img_blur > [100, 100, 100], axis=-1)
    mask[white_pixels_mask] = 0
    target = cv2.bitwise_and(img_blur,img_blur, mask=mask)
    target_blur = cv2.medianBlur(target, 11)
    
    # YIELD Sign   
    edges = cv2.Canny(target_blur, 700,1500)    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/90, threshold=40, minLineLength=30,
                            maxLineGap=15)
    
    if lines is not None:
        lines = lines.reshape(lines.shape[0], -1)[:3]
        center_yield = np.array((int(np.mean(lines[:,[0,2]])), 
                                 int(np.mean(lines[:,[1,3]]))))
        signs['yield'] = (center_yield[0], center_yield[1])
        target_blur[center_yield[1]-40:center_yield[1]+70, center_yield[0]-60:center_yield[0]+60] = \
        target_blur[center_yield[1]-40:center_yield[1]+70, center_yield[0]-60:center_yield[0]+60]*0
    
    # STOP and DNE
    edges = cv2.Canny(target_blur, 300,600)
    circles = cv2.HoughCircles(edges, method=cv2.HOUGH_GRADIENT, dp=1, param1=600, param2=20,
                               minDist=80)

    if circles is not None:
        circles = circles[0]
        index_stp = np.where(np.logical_and(circles[:,2]>45, circles[:,2]<60))[0]
        index_dne = np.where(np.logical_and(circles[:,2]>25, circles[:,2]<40))[0]
        if index_stp.size>0:
            center_stp = circles[index_stp[0]][:2]
            signs['stop'] = (center_stp[0], center_stp[1])
        if index_dne.size>0:
            center_dne = circles[index_dne[0]][:2]
            signs['no_entry'] = (center_dne[0], center_dne[1])

    return signs
    raise NotImplementedError


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    return {'0':(0,0)}
    raise NotImplementedError
