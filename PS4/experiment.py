"""Problem Set 4: Motion Detection"""

import cv2
import os
import numpy as np

import ps4

# I/O directories
input_dir = "input_images"
output_dir = "output"


# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in range(0, v.shape[0], stride):

        for x in range(0, u.shape[1], stride):

            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out


# Functions you need to complete:

def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """
    
    i = level
    while i!=0:
        h, w = pyr[i-1].shape
        u = 2*ps4.expand_image(u)
        v = 2*ps4.expand_image(v)
        h_, w_ = u.shape
        if h != h_:
            u = u[:-1,:]
            v = v[:-1,:]
        if w%2 != 0:
            u = u[:,:-1]
            v = v[:,:-1]
        i = i - 1

    # TODO: Your code here
    return (u,v)
    raise NotImplementedError


def part_1a():

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                       'ShiftR2.png'), 0) / 255.
    shift_r5_u5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                          'ShiftR5U5.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    k = 7
    sigma = 5
    shift_0g = cv2.GaussianBlur(shift_0, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)
    shift_r2g = cv2.GaussianBlur(shift_r2, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)

    k_size = 30  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.
    k = 17
    sigma = 7
    shift_0g = cv2.GaussianBlur(shift_0, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)
    shift_r5_u5g = cv2.GaussianBlur(shift_r5_u5, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)

    k_size = 35 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1 # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0g, shift_r5_u5g, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=1, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.
    
    # Optional: smooth the images if LK doesn't work well on raw images
    k = 31
    sigma = 9
    shift_0g = cv2.GaussianBlur(shift_0, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)
    shift_r10g = cv2.GaussianBlur(shift_r10, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)

    k_size = 65  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0g, shift_r10g, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=0.6, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    k = 31
    sigma = 15
    shift_0g = cv2.GaussianBlur(shift_0, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)
    shift_r20g = cv2.GaussianBlur(shift_r20, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)

    k_size = 90 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1 # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0g, shift_r20g, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=0.5, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-2.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    k = 61
    sigma = 15
    shift_0g = cv2.GaussianBlur(shift_0, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)
    shift_r40g = cv2.GaussianBlur(shift_r40, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)

    k_size = 120 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1 # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0g, shift_r40g, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=0.25, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-3.png"), u_v)
    


def part_2():

    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1',
                                         'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    
    k = 11
    sigma = 7
    yos_img_01g = cv2.GaussianBlur(yos_img_01, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)
    yos_img_02g = cv2.GaussianBlur(yos_img_02, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)


    levels = 5  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01g, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02g, levels)

    level_id = 0 # TODO: Select the level number (or id) you wish to use
    k_size = 20 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 10  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    k = 11
    sigma = 7
    yos_img_02g = cv2.GaussianBlur(yos_img_02, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)
    yos_img_03g = cv2.GaussianBlur(yos_img_03, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)

    levels = 5  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02g, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03g, levels)

    level_id = 0 # TODO: Select the level number (or id) you wish to use
    k_size = 15 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0 # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    levels = 5  # TODO: Define the number of levels
    k_size = 15  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=0.5, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    # You may want to try different parameters for the remaining function
    # calls.
    k = 15
    sigma = 7
    shift_0g = cv2.GaussianBlur(shift_0, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)
    shift_r20g = cv2.GaussianBlur(shift_r20, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)

    k_size = 10
    u20, v20 = ps4.hierarchical_lk(shift_0g, shift_r20g, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u20, v20, scale=0.25, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)


    k = 61
    sigma = 7
    shift_0g = cv2.GaussianBlur(shift_0, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)
    shift_r40g = cv2.GaussianBlur(shift_r40, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)

    k_size = 11
    
    u40, v40 = ps4.hierarchical_lk(shift_0g, shift_r40g, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=0.15, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban01.png'), 0) / 255.
    urban_img_02 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban02.png'), 0) / 255.

    levels = 6  # TODO: Define the number of levels
    k_size = 20  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 7  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    
    urban_img_01g = cv2.GaussianBlur(urban_img_01, ksize=(9,9), sigmaX=sigma, sigmaY=sigma)
    urban_img_02g = cv2.GaussianBlur(urban_img_02, ksize=(9,9), sigmaX=sigma, sigmaY=sigma)
    
    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=0.2, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))


def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
                                        

    # Optional: smooth the images if LK doesn't work well on raw images
    levels = 5  # TODO: Define the number of levels
    k_size = 15  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    # Flow image
    shift_02 = ps4.warp(shift_r10, 0.8*u, 0.8*v, interpolation, border_mode)
    shift_04 = ps4.warp(shift_r10, 0.6*u, 0.6*v, interpolation, border_mode)
    shift_06 = ps4.warp(shift_r10, 0.4*u, 0.4*v, interpolation, border_mode)
    shift_08 = ps4.warp(shift_r10, 0.2*u, 0.2*v, interpolation, border_mode)
    
    u_v = np.vstack((np.hstack((shift_0, shift_02, shift_04)), np.hstack((shift_06, shift_08, shift_r10))))
    cv2.imwrite(os.path.join(output_dir, "ps4-5-a-1.png"), ps4.normalize_and_scale(u_v))


def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    mc01 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                   'mc01.png'), 0) / 255.
    mc02 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                   'mc02.png'), 0) / 255.
    mc03 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                   'mc03.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    levels = 5  # TODO: Define the number of levels
    k_size = 20  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(mc01, mc02, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    # Flow image
    shift_02 = ps4.warp(mc02, 0.8*u, 0.8*v, interpolation, border_mode)
    shift_04 = ps4.warp(mc02, 0.6*u, 0.6*v, interpolation, border_mode)
    shift_06 = ps4.warp(mc02, 0.4*u, 0.4*v, interpolation, border_mode)
    shift_08 = ps4.warp(mc02, 0.2*u, 0.2*v, interpolation, border_mode)
    
    u_v = np.vstack((np.hstack((mc01, shift_02, shift_04)), np.hstack((shift_06, shift_08, mc02))))
    cv2.imwrite(os.path.join(output_dir, "ps4-5-b-1.png"), ps4.normalize_and_scale(u_v))
    

    # Optional: smooth the images if LK doesn't work well on raw images
    levels = 6  # TODO: Define the number of levels
    k_size = 60  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 9  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    
    mc02g = cv2.GaussianBlur(mc02, ksize=(15,15), sigmaX=sigma, sigmaY=sigma)
    mc03g = cv2.GaussianBlur(mc03, ksize=(15,15), sigmaX=sigma, sigmaY=sigma)
    

    u, v = ps4.hierarchical_lk(mc02g, mc03g, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    # Flow image
    shift_02 = ps4.warp(mc03, 0.8*u, 0.8*v, interpolation, border_mode)
    shift_04 = ps4.warp(mc03, 0.6*u, 0.6*v, interpolation, border_mode)
    shift_06 = ps4.warp(mc03, 0.4*u, 0.4*v, interpolation, border_mode)
    shift_08 = ps4.warp(mc03, 0.2*u, 0.2*v, interpolation, border_mode)
    
    u_v = np.vstack((np.hstack((mc02, shift_02, shift_04)), np.hstack((shift_06, shift_08, mc03))))
    cv2.imwrite(os.path.join(output_dir, "ps4-5-b-2.png"), ps4.normalize_and_scale(u_v))
    
def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    
    IMG_DIR = "input_images"
    VID_DIR = "input_videos"
    OUT_DIR = "output"
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    
    video_file = "ps4-my-video.mp4"
    fps = 40
    
    counter_init = 1
    output_prefix = 'part_6'
    
    # Todo: Complete this part on your own.'
    video = os.path.join(VID_DIR, video_file)
    image_gen = video_frame_generator(video)
    image1 = image_gen.__next__()
    image2 = image_gen.__next__()
    h, w, d = image1.shape
    
    out_path = "output/ar_{}-{}".format('part_6', 'quiver.mp4')
    video_out = mp4_video_writer(out_path, (w, h), fps)

    output_counter = counter_init

    frame_num = 1
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    while image2 is not None:
    
        
        print("Processing fame {}".format(frame_num))
        k = 15
        sigma = 7
        image1g = cv2.GaussianBlur(image1, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)
        image2g = cv2.GaussianBlur(image2, ksize=(k,k), sigmaX=sigma, sigmaY=sigma)
        
        levels = 4
        k_size = 30
        u20, v20 = ps4.hierarchical_lk(image1g[:,:,0]/255., image2g[:,:,0]/255., 
                                       4, k_size, k_type,
                                       sigma, interpolation, border_mode)
    
        u_v = quiver_(u20, v20, image1, scale=3, stride=10)

        frame_id = frame_ids[(output_counter - 1) % 3]

        if frame_num == frame_id:
            out_str = output_prefix + "-{}.png".format(output_counter)
            save_image(out_str, u_v)
            output_counter += 1

        video_out.write(u_v)
        
        image1 = image2.copy()
        image2 = image_gen.__next__()

        frame_num += 1

    video_out.release()


def quiver_(u, v, image, scale, stride, color=(0, 255, 0)):

    img_out = image.copy()

    for y in range(0, v.shape[0], stride):

        for x in range(0, u.shape[1], stride):

            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out

def mp4_video_writer(filename, frame_size, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

def save_image(filename, image):
    cv2.imwrite(os.path.join(OUT_DIR, filename), image)
    
    
if __name__ == '__main__':
    part_1a()
    part_1b()
    part_2()
    part_3a_1()
    part_3a_2()
    part_4a()
    part_4b()
    part_5a()
    part_5b()
    part_6()
