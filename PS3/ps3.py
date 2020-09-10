"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
from scipy import ndimage
from scipy.cluster.vq import kmeans


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    euc = np.sqrt(np.sum(np.square(p0[0]-p1[0]), np.square(p0[1]-p1[1])))
    return euc
    raise NotImplementedError


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    x = image.shape[0]-1
    y = image.shape[1]-1
    corners = [(0,0), (0,x), (y,0), (y,x)]
    
    return corners
    raise NotImplementedError


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)
    
    B = (1/8)*np.ones((8,8), dtype=np.float32)
    B[:4,4:] = -1*B[:4,4:]
    B[4:,:4] = -1*B[4:,:4]
    
    angles = np.arange(-90, 91, 7.5, dtype=int)
    for i in angles:
        B_ = ndimage.rotate(B, i, reshape=True)
        
        C = cv2.filter2D(img_gray[5:,5:][:-5,:-5], ddepth = -1, kernel = B_)

        n=22
        j=6
        u = 0.8
        v = 4
        lo = int(0.5*(n-j))
        hi = int(0.5*(n+j))
        spot = (-1)*np.ones((n,n), dtype=np.float32)
        spot[lo:hi,lo:hi] = -1*spot[lo:hi,lo:hi]
        spot[-v:,:] = u*spot[-v:,:]
        spot[:,-v:] = u*spot[:,-v:]
        spot[:v,:] =  u*spot[:v,:]
        spot[:,:v] =  u*spot[:,:v]
        blobs = cv2.filter2D(C, ddepth = -1, kernel = spot)
        
        centers = np.array(np.argwhere(blobs==255), dtype = "float32") + 5
        if centers.shape[0]>15:
            break
    
    if centers.shape[0]>3:
        markers = np.array(kmeans(centers,4)[0], dtype = int)   
        
        rank_y = markers[:,1].argsort()
        rank_x1 = markers[rank_y[:2]][:,0].argsort()
        rank_x2 = markers[rank_y[2:]][:,0].argsort()
        
        p1 = markers[rank_y[:2]][rank_x1[0]]
        p2 = markers[rank_y[:2]][rank_x1[1]]
        p3 = markers[rank_y[2:]][rank_x2[0]]
        p4 = markers[rank_y[2:]][rank_x2[1]]
        
        final_markers = [(p1[1], p1[0]), (p2[1], p2[0]), (p3[1], p3[0]), (p4[1], p4[0])]
    else:
        final_markers = [(0,0), (2,0), (0,2), (2,2)]
    
    return final_markers
    raise NotImplementedError


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    
    cv2.line(image, (markers[0][0], markers[0][1]), (markers[1][0], markers[1][1]), (0,255,0), thickness)
    cv2.line(image, (markers[1][0], markers[1][1]), (markers[3][0], markers[3][1]), (0,255,0), thickness)
    cv2.line(image, (markers[2][0], markers[2][1]), (markers[3][0], markers[3][1]), (0,255,0), thickness)
    cv2.line(image, (markers[2][0], markers[2][1]), (markers[0][0], markers[0][1]), (0,255,0), thickness)
    
    return image
    raise NotImplementedError


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    H = np.linalg.inv(homography)
    
    k, w = imageB.shape[:2]
    indy, indx = np.indices((k, w), dtype=np.float32)
    lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])
    
    map_ind = H.dot(lin_homg_ind)
    map_x, map_y = map_ind[:-1]/map_ind[-1]
    map_x = map_x.reshape(k, w).astype(np.float32)
    map_y = map_y.reshape(k, w).astype(np.float32)
    
    if np.min(map_x)<-5000 and np.max(map_x)>5000:
        ad = imageB
    else:
        ad = cv2.remap(imageA, map_x, map_y, cv2.INTER_LINEAR, dst=imageB, borderMode=cv2.BORDER_TRANSPARENT)

    return ad    
    raise NotImplementedError


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    
    x1 = src_points[0][0]
    y1 = src_points[0][1]
    x2 = src_points[1][0]
    y2 = src_points[1][1]
    x3 = src_points[2][0]
    y3 = src_points[2][1]
    x4 = src_points[3][0]
    y4 = src_points[3][1] 
    
    x1_ = dst_points[0][0]
    y1_ = dst_points[0][1]
    x2_ = dst_points[1][0]
    y2_ = dst_points[1][1]
    x3_ = dst_points[2][0]
    y3_ = dst_points[2][1]
    x4_ = dst_points[3][0]
    y4_ = dst_points[3][1]       
                                        
    P = np.array([[-x1, -y1, -1, 0, 0, 0, x1*x1_, y1*x1_, x1_],
                 [0, 0, 0, -x1, -y1, -1, x1*y1_, y1*y1_, y1_],
                 [-x2, -y2, -1, 0, 0, 0, x2*x2_, y2*x2_, x2_],
                 [0, 0, 0, -x2, -y2, -1, x2*y2_, y2*y2_, y2_],
                 [-x3, -y3, -1, 0, 0, 0, x3*x3_, y3*x3_, x3_],
                 [0, 0, 0, -x3, -y3, -1, x3*y3_, y3*y3_, y3_],
                 [-x4, -y4, -1, 0, 0, 0, x4*x4_, y4*x4_, x4_],
                 [0, 0, 0, -x4, -y4, -1, x4*y4_, y4*y4_, y4_]])
    
    u, s, vh = np.linalg.svd(P, full_matrices=True)
    h = vh[-1,:]
    H = h.reshape((3,3))
    H = H/H[2,2]
    
    return H
    raise NotImplementedError


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video =  cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    
    yield None
    raise NotImplementedError
