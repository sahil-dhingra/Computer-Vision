"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os
from scipy import signal


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    image_out = cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale = 1/8, borderType = cv2.BORDER_DEFAULT)
    
    return grad_x
    raise NotImplementedError


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale = 1/8, borderType = cv2.BORDER_DEFAULT)
    
    return grad_y
    raise NotImplementedError


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    
    h,w = img_a.shape
    sobel = (1/8)*np.array([[-1 , 0 , 1], [-2 , 0 , 2], [-1 , 0 , 1]])
    grad_x = cv2.filter2D(img_a, ddepth = -1, kernel = sobel)
    grad_y = cv2.filter2D(img_b, ddepth = -1, kernel = sobel.T)
    grad_t = img_b - img_a
    
    i_xx = np.zeros((h,w))
    i_xy = np.zeros((h,w))
    i_yy = np.zeros((h,w))
    i_xt = np.zeros((h,w))
    i_yt = np.zeros((h,w))
    
    i_xx = cv2.boxFilter(np.multiply(grad_x, grad_x),  cv2.CV_64F, (k_size, k_size), i_xx, (-1,-1), False,  cv2.BORDER_DEFAULT)
    i_xy = cv2.boxFilter(np.multiply(grad_x, grad_y),  cv2.CV_64F, (k_size, k_size), i_xy, (-1,-1), False,  cv2.BORDER_DEFAULT)
    i_yy = cv2.boxFilter(np.multiply(grad_y, grad_y),  cv2.CV_64F, (k_size, k_size), i_yy, (-1,-1), False,  cv2.BORDER_DEFAULT)
    
    i_xt = cv2.boxFilter(np.multiply(grad_x, grad_t),  cv2.CV_64F, (k_size, k_size), i_xt, (-1,-1), False,  cv2.BORDER_DEFAULT)
    i_yt = cv2.boxFilter(np.multiply(grad_y, grad_t),  cv2.CV_64F, (k_size, k_size), i_yt, (-1,-1), False,  cv2.BORDER_DEFAULT)

    
    A = np.array([[i_xx.T, i_xy.T], [i_xy.T, i_yy.T]]).T
    b = np.array([-i_xt.T, -i_yt.T]).T
    
    dets = np.linalg.det(A)
    
    A_inv = A.copy()
    A_inv[np.where(dets != 0)] = np.linalg.inv(A[np.where(dets != 0)])
    A_inv[np.where(dets == 0)] = A_inv[np.where(dets == 0)]*0
    
    A_u = A_inv.T[0].T
    A_v = A_inv.T[1].T
    
    U = np.sum(A_u*b, axis=2)
    V = np.sum(A_v*b, axis=2)
    
    return (U,V)
    raise NotImplementedError


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    image_norm = image.copy()
    five_tap = (1/16)*np.array([1, 4, 6, 4, 1])
    reduce = cv2.sepFilter2D(src = image_norm, ddepth = cv2.CV_64F, kernelX = five_tap, kernelY = five_tap)
    reduce = reduce[::2,::2]
    h,w = reduce.shape

    return reduce
    raise NotImplementedError


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    
    gp = [image]
    for i in range(levels-1):
        gp.append(reduce_image(gp[i]))
    
    return gp
    raise NotImplementedError


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    
    image = img_list[0]
    h,w = image.shape
    for i in range(len(img_list)-1):
        img = img_list[i+1]
        h_, w_ = img.shape
        new_image = np.zeros((h,w_))
        new_image[:h_,:w_] = img
        image = np.concatenate((image, new_image), axis=1)
    
    image_ = image.copy()
    image_ = cv2.normalize(image, image_, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX)
    
    return image_
    raise NotImplementedError


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    image_norm = image.copy()
    
    h,w = image.shape
    expand_ = np.zeros((2*h,2*w))
    expand_[::2,::2] = image_norm
    
    five_tap = (1/8)*np.array([1, 4, 6, 4, 1])
    expand = cv2.sepFilter2D(expand_, ddepth = cv2.CV_64F, kernelX = five_tap, kernelY = five_tap)
    
    return expand
    raise NotImplementedError


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    l_pyr = g_pyr.copy()
    for i in range(len(g_pyr)-1):
        expand = expand_image(g_pyr[-1-i]) 
        reduce = g_pyr[-2-i].copy()
        h, w = reduce.shape
        if h%2 != 0:
            reduce = np.vstack((reduce, reduce[-1,:]))
        if w%2 != 0:
            reduce = np.vstack((reduce.T, reduce[:,-1])).T
        l_pyr[-2-i] = reduce - expand
        
    return l_pyr
    raise NotImplementedError


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    k, w = image.shape
    indy, indx = np.indices((k, w), dtype=np.float32)

    map_x, map_y = indx + U, indy + V 
    map_x = map_x.reshape(k, w).astype(np.float32)
    map_y = map_y.reshape(k, w).astype(np.float32)

    warp = np.zeros((k,w))
    warp = cv2.remap(image, map_x, map_y, interpolation, dst=warp, borderMode=border_mode)
    
    return warp
    raise NotImplementedError


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    
    reduce_a = [img_a]
    reduce_b = [img_b]
    for i in range(levels-1):
        reduce_a.append(reduce_image(reduce_a[i]))
        reduce_b.append(reduce_image(reduce_b[i]))
    
    for k in range(levels):
        a = reduce_a[-1-k]
        b = reduce_b[-1-k]
        if k == 0:
            h, w = a.shape
            U = np.zeros((h,w))
            V = np.zeros((h,w))
        else:
            U = 2*expand_image(U)
            V = 2*expand_image(V)
        
        warped_b = warp(b, U, V, interpolation, border_mode)
        dx, dy = optic_flow_lk(a, warped_b, k_size, k_type, sigma)
        U = U + dx
        V = V + dy
        
    return (U, V)
    raise NotImplementedError
