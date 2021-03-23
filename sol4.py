# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged

import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates
import shutil
from imageio import imwrite

import math
from scipy.ndimage import convolve
from scipy import signal

import sol4_utils


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    # derivate image:
    filter = np.atleast_2d(np.array([1, 0, -1]))
    x_der = signal.convolve2d(im, filter, 'same', boundary="symm")
    y_der = signal.convolve2d(im, filter.T, 'same', boundary="symm")

    x_der_squared = np.multiply(x_der, x_der)
    y_der_squared = np.multiply(y_der, y_der)
    xy_der = np.multiply(x_der, y_der)

    # blur image:
    kernel_size = 3  # as stated in PDF
    x_der_squared = sol4_utils.blur_spatial(x_der_squared, kernel_size)
    y_der_squared = sol4_utils.blur_spatial(y_der_squared, kernel_size)
    xy_der = sol4_utils.blur_spatial(xy_der, kernel_size)

    # compute R:
    det_M = (x_der_squared * y_der_squared) - (xy_der ** 2)
    trace_M = x_der_squared + y_der_squared
    k = 0.04  # as told in ex4 pdf

    r = det_M - k * (trace_M ** 2)

    maximum = non_maximum_suppression(r)

    temp = np.argwhere(maximum == True)
    temp = np.flip(temp, axis=1)  # flipping x,y axes
    return temp


def sample_descriptor(im, pos, desc_rad=3):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image. - ALREADY EXPECTS IM TO BE THE 3RD LEVEL OF PYR
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desciptors_list[i,:,:].
    1. prepare 3-level Gaussian Pyramid of the image
    2. We will extract a 7x7 matrix of **normalized** image intensities. These should be sampled in the third pyramid level image,
    in a 7x7 patch centred around p_l_3 (location of p in 3rd level).
    3. sample at sub-pixel corrdinates Sby interploation (map_coordinates)
    4, normalize
    desc_rad should be set to 3
    Use map_coordinates with order=1 and prefilter=False for linear interpolation
    1. linspace, meshgrid -> build a 7x7 window aroudn each pooints of indicies of Y THEN X
    2. take the windows and give them to map_coordinates which gets y,x and image
    3. do reshape with output
    4. if the windows[i] - windows[i].mean() == 0: do windows[i] = 0
    5. else windowss[i] = windows[i] - windows[i].mean() / blah blah blah like before
    6. return windows as ndarray
    """
    desciptors_list = []
    for i in range(pos.shape[0]):
        x = pos[i, :][0]
        y = pos[i, :][1]
        x_linspace = np.linspace(x - desc_rad, x + desc_rad, num=(1 + 2 * desc_rad))
        y_linspace = np.linspace(y - desc_rad, y + desc_rad, num=(1 + 2 * desc_rad))
        x_mesh, y_mesh = np.meshgrid(x_linspace, y_linspace)
        descriptor = map_coordinates(im, [y_mesh, x_mesh], order=1, prefilter=False)
        normalization_factor = np.linalg.norm(descriptor - np.mean(descriptor))
        if (normalization_factor != 0):
            descriptor = (descriptor - np.mean(descriptor)) * (1 / normalization_factor)
        else:
            descriptor = np.zeros(descriptor.shape)
        desciptors_list.append(descriptor)
    return np.array(desciptors_list)


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    im = pyr[0]
    harris_corners = spread_out_corners(im, 7, 7, 13)
    # moving up two levels is equivalent for divison by 2^2=4:
    samples = sample_descriptor(pyr[2], harris_corners / 4)
    return harris_corners, samples

def match_features(desc1, desc2, min_score):
    """
      Return indices of matching descriptors.
      :param desc1: A feature descriptor array with shape (N1,K,K).
      :param desc2: A feature descriptor array with shape (N2,K,K).
      :param min_score: Minimal match score.
      :return: A list containing:
                  1) An array with shape (M,) and dtype int of matching indices in desc1.
                  2) An array with shape (M,) and dtype int of matching indices in desc2.
      """
    k = desc1.shape[1]
    scores=np.sum(desc1.reshape(desc1.shape[0],1,k,k) * desc2.reshape(1, desc2.shape[0], k, k), axis=(2, 3))
    array_max_indicator = np.zeros(scores.shape)

    # adding one where above min:
    array_max_indicator[scores >min_score] += 1
    scores[scores <= min_score] = 0

    # adds 1 when col max:
    cols_sec = np.partition(scores, kth=-2, axis=0)[-2, :]
    array_max_indicator[scores >= cols_sec] += 1

    # adds 1 when row max:
    rows_sec = np.partition(scores, kth=-2, axis=1)[:, -2]
    array_max_indicator[np.transpose(scores.T >= rows_sec)] += 1
    result = np.where(array_max_indicator == 3)
    return result


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    # convert_to_homography_coordinates: lambda x,y: x,y

    # standart2homography:
    homography_coordinates = np.concatenate((pos1, np.ones((pos1.shape[0], 1))), axis=1)

    # multiply coordinate with H12:
    homography_coordinates = np.einsum('ij,kj->ki', H12, homography_coordinates)

    # homography2standart:
    divisor = np.dstack((homography_coordinates[:, 2], homography_coordinates[:, 2], homography_coordinates[:, 2]))
    homography_coordinates = np.divide(homography_coordinates, divisor)[0]
    return homography_coordinates[:, :-1]


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.

    I may find the following funcions useful: np.random.permutationornp.random.chioce
    """
    biggest_inlier_indexes = np.empty((0))
    N = points1.shape[0]
    for i in range(num_iter):
        random_indicies = np.random.choice(N, 2)  # this rolls the dice twice
        H12 = estimate_rigid_transform(points1[random_indicies], points2[random_indicies], translation_only)
        H12 = H12 / (H12[2, 2])  # normalization
        estimated_points2 = apply_homography(points1, H12)
        dist = np.linalg.norm(estimated_points2 - points2, axis=1) ** 2
        inliers_indexes = np.argwhere(dist < inlier_tol)
        if (len(inliers_indexes) > len(biggest_inlier_indexes)):
            biggest_inlier_indexes = inliers_indexes.reshape(inliers_indexes.shape[0])
    H = estimate_rigid_transform(points1[biggest_inlier_indexes], points2[biggest_inlier_indexes], translation_only)
    H = H / H[2, 2]
    return H, biggest_inlier_indexes


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """

    im = np.hstack((im1, im2))
    x_shift = im1.shape[1]
    plt.imshow(im, cmap='gray')
    points2[:, 0] += x_shift

    inliers_arr = np.array([points1[inliers], points2[inliers]])
    mask = np.ones(points1.shape[0])
    mask[inliers] = False
    mask = mask.astype(bool)
    outliers_arr = np.array([points1[mask], points2[mask]])

    plt.plot(outliers_arr[:, :, 0], outliers_arr[:, :, 1], c="b", marker='o',mfc='r', ms=1, lw=.4, zorder=1)
    plt.plot(inliers_arr[:, :, 0], inliers_arr[:, :, 1], c="y", marker='o',mfc='r', ms=1, lw=.4, zorder=1)

    # the PDF said:
    # This function should display a horizontally concatenated image
    # (usenp.hstackof an image pairim1andim2, with the matched points provided in pos1 andpos 2
    #  ***overlayed correspondingly** as red dots
    # thus I added the red dots on top of plot.
    plt.scatter(outliers_arr[:, :, 0], outliers_arr[:, :, 1], marker='.', color='r', zorder=2)
    plt.scatter(inliers_arr[:, :, 0], inliers_arr[:, :, 1],marker='.',color='r', zorder=2)
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    accumulated = [np.empty((3, 3)) for i in range(len(H_succesive) + 1)]
    accumulated[m] = np.eye(3)  # the 1 matrix
    for i in range(m - 1, -1, -1):
        matrix = accumulated[i + 1] @ H_succesive[i]
        matrix = matrix / matrix[2, 2]
        accumulated[i] = matrix
    for i in range(m + 1, len(H_succesive) + 1):
        current_inv = np.linalg.inv(H_succesive[i - 1])
        matrix = accumulated[i - 1] @ current_inv
        matrix = matrix / matrix[2, 2]
        accumulated[i] = matrix
    return accumulated


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner   """

    points = apply_homography(np.array([[w - 1, h - 1], [0, 0], [w - 1, 0], [0, h - 1]]), homography)
    points = points.astype(np.int).T

    top_left_x = points[0].min()
    top_left_y = points[1].min()
    bottom_right_x = points[0].max()
    bottom_right_y = points[1].max()
    return np.array([[top_left_x, top_left_y], [bottom_right_x, bottom_right_y]]).astype(np.int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    inv = np.linalg.inv(homography)
    x, y = compute_bounding_box(homography, image.shape[1], image.shape[0])

    x_linspace = np.linspace(x[0], y[0], num=y[0] - x[0] + 1)
    y_linspace = np.linspace(x[1], y[1], num=y[1] - x[1] + 1)
    x_mesh, y_mesh = np.meshgrid(x_linspace, y_linspace)
    xy_mesh = np.vstack((x_mesh.flatten(), y_mesh.flatten())).T

    warped_coordinates = apply_homography(xy_mesh, inv).reshape((x_mesh.shape[0], x_mesh.shape[1], 2))
    warped_x, warped_y = warped_coordinates[:, :, 0], warped_coordinates[:, :, 1]

    warped = map_coordinates(image, [warped_y, warped_x], order=1, prefilter=False)
    return warped


# ============== [ Supplied Code + Bonus Adjustments] =================


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.num_images = num_images
        self.bonus = bonus
        self.data_dir = data_dir
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        if(self.bonus == True):
            self.generate_panoramic_images_bonus(number_of_panoramas)
        else:
            self.generate_panoramic_images_regular(number_of_panoramas)

    def generate_panoramic_images_regular(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image

        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()

# ========================== [ADDED CODE] ======================================

    def approximate_power_of_2(self, num):
        """
        ================ [ADDED CODE ] ================
        :param num: a number
        :return: the closet x, s.t. x < num and \exists y s.t. 2^y=x
        """
        x = math.log(num, 2)
        x = math.floor(x)
        x = 2 ** x
        return int(x)

    def save_align(self):
        """
        a function for debugging, which saves homorgraphies
        :return: nothing
        """
        np.savez("align.npz", a=self.homographies, b=self.frames_for_panoramas)

    def load_align(self):
        """
        a function for debugging, which uses saves homoraphies and frames for panoramas.
        this way it's take way less time to run the new panorama generator and helps me debug quickly.
        :return: nothings
        """
        data = np.load('align.npz')
        self.homographies = data['a']
        self.frames_for_panoramas = data['b']
        # self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape



    def generate_panoramic_images_bonus(self, number_of_panoramas):
        """
        combine slices from input images to panoramas and blends them with barcode blending.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        # ===========[ ADDED CODE ] =============
        # we want to crop and get a new size in powers of 2:
        crop_width = self.approximate_power_of_2(panorama_size[1])
        crop_height = self.approximate_power_of_2(panorama_size[0])

        # let's define an even and odd panorama + a binary mask:
        # first I define them at usual, only in panorama_crop_and_blend I actually do resize things
        even_strips_panorama = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        odd_strips_panorama = even_strips_panorama.copy()  # same dimensions
        barcode_mask = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0]), dtype=np.bool)

        self.panoramas = np.zeros((number_of_panoramas, crop_width, crop_height, 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)

            # ===========[ ADDED CODE ] =============
            # cropped image to wanted size:
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]
            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[-1] + x_offset]
                x_end = boundaries[0] + image_strip.shape[1]

                # ===========[ ADDED CODE ] =============
                # we will define equivalent variables for the mask:
                mask_boundaries = x_strip_boundary[panorama_index, i:i + 2]
                mask_strip = warped_image[:, boundaries[0] - x_offset: boundaries[-1] - x_offset]
                mask_x_end = mask_boundaries[0] + mask_strip.shape[1]
                if (i % 2 == 0):
                    barcode_mask[panorama_index, :, boundaries[0]:mask_x_end] = True
                    even_strips_panorama[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip
                else:
                    odd_strips_panorama[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # =========== [ ADDED CODE ] ============
        # pyramid_blending:
        for panorama_index in range(number_of_panoramas):
            self.panoramas[panorama_index] = self.panorama_crop_and_blend(even_strips_panorama[panorama_index],
                                                                          odd_strips_panorama[panorama_index],
                                                                          barcode_mask[panorama_index],
                                                                          crop_height, crop_width, panorama_size)
        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        self.final_crop_panorama(crop_width, panorama_size)

    def panorama_crop_and_blend(self, even, odd, mask, crop_height, crop_width, panorama_size):
        """
        ================ [ADDED CODE ] ================
        :param even: even panorama image
        :param odd: odd panorama image
        :param mask: barcode mask
        :param height_top_crop: how much to crop from top to crop from the middle of the image
        :param height_bottom_crop: how much to crop from bottom
        :param width_crop_left: how much to crop from left
        :param width_crop_right: how much to crop from right
        :return: cropped, pyramid blended panorama
        """
        width_crop_left = int((panorama_size[1] - crop_width) * 0.5)
        width_crop_right = int(width_crop_left + crop_width)
        height_top_crop = int((panorama_size[0] - crop_height) * 0.5)
        height_bottom_crop = int(crop_height + height_top_crop)

        cropped_even = even[width_crop_left:width_crop_right, height_top_crop:height_bottom_crop, :]
        cropped_odd = odd[width_crop_left:width_crop_right, height_top_crop:height_bottom_crop, :]
        cropped_mask = mask[width_crop_left:width_crop_right, height_top_crop:height_bottom_crop]
        # after some tries I think this parameters gives be a nice result:
        blended = sol4_utils.rgb_blending(cropped_odd, cropped_even, cropped_mask, 7, 3, 7)
        return blended

    def final_crop_panorama(self, crop_width, panorama_size):
        """
        =========================== [ ADJUSTED CODE ] ============================
        :param width_crop_left: how much I cropped from left to get powers of 2
        :return: nothing
        """
        # I have already cropped width_crop_left from the panorama, but bounding_boxes doens't know that
        # so I need to make some adjustments:
        width_crop_left = int((panorama_size[1] - crop_width) * 0.5)

        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_left = max(0, crop_left - width_crop_left)

        # same for the right
        crop_right = int(self.bounding_boxes[-1][0, 0])
        crop_right = min(crop_right - width_crop_left, crop_right)
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

