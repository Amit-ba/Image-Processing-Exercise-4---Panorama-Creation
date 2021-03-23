from scipy.signal import convolve2d
import numpy as np
from scipy.signal import convolve as signal_convolve
from scipy.ndimage import convolve
from imageio import imread
from skimage import color as sk


def read_image(filename, representation):
    """
    Reads image
    :param filename: name of file
    :param representation: 1 for bw, 2 for color
    :return: image, bw or color as wanted, in 0-1 range
    """
    image = imread(filename)
    image = image.astype('float64')
    image = image/255
    if(representation == 1 and len(image.shape) == 3):
        if(image.shape[2] == 3):
            image = sk.rgb2gray(image)
        else:
            # I downloaded some RGBA images for testing so I added this:
            image = sk.rgb2gray(sk.rgba2rgb(image))
    return image

def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img

# ======================== functions from sol3.py for building gaussian pyramid: =======
import numpy as np
from scipy.ndimage import convolve
from scipy.signal import convolve as signal_convolve
from matplotlib import pyplot as plt
from skimage import color as sk
import imageio
import os

def read_image(filename, representation):
    """
    Reads image
    :param filename: name of file
    :param representation: 1 for bw, 2 for color
    :return: image, bw or color as wanted, in 0-1 range
    """
    image = imageio.imread(filename)
    image = image.astype('float64')
    image = image/255
    if(representation == 1 and len(image.shape) == 3):
        if(image.shape[2] == 3):
            image = sk.rgb2gray(image)
        else:
            # I downloaded some RGBA images for testing so I added this:
            image = sk.rgb2gray(sk.rgba2rgb(image))
    return image

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    :param im: a grayscale image with double values in [0,1]
    :param max_levels:  the maximal number of levels1in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter

    output pyr is a python array (not numpy!!)
    filter row vec is of shape (1, filter_size) used for construction of pyr.
    """
    filter_vec = get_filter_vec(filter_size)
    pyr = [im]
    for i in range(max_levels-1):
        pyr.append(build_gaussian_level(pyr[i],filter_vec))
        if(pyr[i+1].shape[0]//2 < 16): break
    return pyr, filter_vec

def get_filter_vec(filter_size):
    if(filter_size == 1):
        return np.atleast_2d(np.array([1]))
    convolution_vec = filter_vec = np.array([1,1])
    for i in range(filter_size-2):
        filter_vec = signal_convolve(filter_vec, convolution_vec, mode='full')
    # return filter_vec * (1/np.sum(filter_vec))
    return np.atleast_2d(filter_vec) * (1/np.sum(filter_vec))


def build_gaussian_level(prev_im, filter_vec):
    # blur:
    #used to be mode 'constant'
    blurred_image = convolve(prev_im, filter_vec)
    blurred_image = convolve(blurred_image, filter_vec.T)
    # subsample:
    blurred_image = np.delete(blurred_image, np.s_[::2], axis=0)
    blurred_image = np.delete(blurred_image, np.s_[::2], axis=1)
    return np.array(blurred_image)


def expand_gaussian_level(im, filter_vec):
    # pad with zeros:
    expanded_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    expanded_im[0::2, 0::2] = im

    # blur:
    expanded_im = convolve(expanded_im, filter_vec*2)
    expanded_im = convolve(expanded_im, filter_vec.T * 2)
    return np.array(expanded_im)



def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    :param im: a grayscale image with double values in [0,1]
    :param max_levels:  the maximal number of levels1in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter

    output pyr is a python array (not numpy!!)
    filter row vec is of shape (1, filter_size) used for construction of pyr.
    """
    gaussian_pyr, filter = build_gaussian_pyramid(im, max_levels, filter_size)
    max_levels = len(gaussian_pyr)
    expanded = [expand_gaussian_level(gaussian_pyr[i], filter) for i in range(1, max_levels)]
    laplacian_pyr = [gaussian_pyr[i] - expanded[i] for i in range(0, max_levels-1)]
    laplacian_pyr.append(gaussian_pyr[max_levels-1])
    return laplacian_pyr, filter


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    :param lpyr: like above
    :param filter_vec:  like above
    :param coeff: python list, length same as level number is lpyr.
    :return:
    """
    sum = lpyr[0]*coeff[0]
    for i in range(1, len(lpyr)):
        cur_level = lpyr[i]
        for j in range(i):
            cur_level = expand_gaussian_level(cur_level, filter_vec)
        sum += cur_level * coeff[i]
    return sum


def render_pyramid(pyr, levels):
    """
    :param pyr:  as above
    :param levels: number of levels to present
    :return: image
    THE NUMBER OF LEVELS INCLUDE THE OG IMAGE
    STRECH THE VALUES TO [0,1]
    """
    res = [pyr[0]]
    original_row_num = pyr[0].shape[0]
    for i in range(1, levels):
        current_im = pyr[i]
        pad_size = original_row_num - current_im.shape[0]
        current_im = np.append(current_im,np.zeros((pad_size, current_im.shape[1])),axis=0)
        res.append(current_im)
    res = np.concatenate(res, axis=1)

    # linear strech:
    res = (res - np.min(res))/(np.max(res) - np.min(res))
    return res

def display_pyramid(pyr, levels):
    rendered = render_pyramid(pyr, levels)
    plt.imshow(rendered, cmap='gray')
    plt.show()

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """

    :param im1: greyscale image to be blended
    :param im2: greyscale image to be blended
    :param mask: dtype bool representing the parts. True = im1, False = im2 (I THINK??)
    :param max_levels: used in pyr generation
    :param filter_size_im:
    :param filter_size_mask:
    :return:
    """
    mask = mask.astype(np.float64)
    laplacian_1, filter_1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    laplacian_2, filter_2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    mask_gaussian, mask_filer = build_gaussian_pyramid(mask, max_levels, filter_size_mask)
    neg_mask_guassian = [(np.ones(level.shape)-level) for level in mask_gaussian]

    new_pyramid = []
    # for i in range(max_levels):
    for i in range(len(laplacian_1)):
        temp = np.multiply(laplacian_1[i], mask_gaussian[i]) + np.multiply(laplacian_2[i], neg_mask_guassian[i])
        new_pyramid.append(temp)
    blended_image = laplacian_to_image(new_pyramid, filter_1, [1 for i in range(max_levels)])
    return np.clip(blended_image, 0, 1)


def rgb_blending(im1, im2, mask, max_levels, filter_size_im,filter_size_mask):
    blended = []
    for i in range(len(im1.shape)):
        blended.append(pyramid_blending(im1[:,:,i], im2[:,:,i], mask, max_levels, filter_size_im, filter_size_mask))
    blended = np.dstack(blended)
    return blended