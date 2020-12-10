import numpy as np
from tqdm import tqdm
from utils import *


def transfer_color(in_img, src_img, sample_rate=0.05, filter_colors=False):
    """
    Transfer colors from src_img to in_img

    :param in_img: greyscale image to transfer colors to
    :param src_img: colored image to transfer colors from
    :param sample_rate: proportion of pixels in in_img to be colored
    :param filter_colors: indicator of whether or not colors should be filtered with bilateral
    :return: RGB image with colors transferred from src_img to in_img at rate sample_rate
    """
    # Separate luminance and chrominance of images
    in_yuv = rgb_to_yuv(in_img)
    src_yuv = rgb_to_yuv(src_img)

    # Get coordinates in src_img to get colors from and coordinates in in_img to color
    sampled_coordinates = sample_coordinates(src_yuv)
    color_coordinates = sample_coordinates(in_yuv, sample_rate=sample_rate)

    # Compute statistics of coordinates in in_img and src_img
    in_means, in_vars = compute_statistics(in_yuv[:, :, 0], color_coordinates)
    src_means, src_vars = compute_statistics(src_yuv[:, :, 0], sampled_coordinates)

    out_yuv = in_yuv.copy()
    for i, j in tqdm(color_coordinates, desc="Finding color matches in source"):
        # Find sampled coordinate in src_img with most similar stats and transfer chrominance
        similar_coord = get_similar_coordinate((i, j), sampled_coordinates, in_means, in_vars, src_means, src_vars)
        out_yuv[i, j, 1] = src_yuv[similar_coord[0], similar_coord[1], 1]
        out_yuv[i, j, 2] = src_yuv[similar_coord[0], similar_coord[1], 2]
    
    # Perform bilateral filter on U and V domains if necessary
    if filter_colors:
        out_yuv[:, :, 1] = bilateral_filter(out_yuv[:, :, 1])
        out_yuv[:, :, 2] = bilateral_filter(out_yuv[:, :, 2])

    # Convert back to RGB
    return yuv_to_rgb(out_yuv)


def get_similar_coordinate(coord, sampled_coordinates, in_means, in_vars, src_means, src_vars):
    """
    Get coordinate from sampled_coordinates that has the most similar stats in in_img

    :param coord: (row, col) of pixel in in_img to find similar coordinate to
    :param sampled_coordinates: set of (row, col) in src_img to check similarity with
    :param in_means: dictionary of (row, col): <neighborhood mean> in in_img
    :param in_vars: dictionary of (row, col): <neighborhood variance> in in_img
    :param src_means: dictionary of (row, col): <neighborhood mean> in src_img
    :param src_vars: dictionary of (row, col): <neighborhood variance> in src_img
    :return: (row, col) of pixel in src_img with most similar statistics to coord in in_img
    """
    min_diff = float("inf")
    min_coord = None
    in_mean = in_means[(coord[0], coord[1])]
    in_var = in_vars[(coord[0], coord[1])]

    for i, j in sampled_coordinates:
        # Compute difference of two pixels as the sum of neighborhood mean and variance with equal weight
        diff = abs(in_mean - src_means[(i, j)]) + \
                abs(in_var**0.5 - src_vars[(i, j)]**0.5)
        # Find coordinate with minimum difference
        if diff < min_diff:
            min_diff = diff
            min_coord = (i, j)
    return min_coord


def compute_statistics(channel_data, coords):
    """
    Compute statistics of channel data at given coords

    :param channel_data: 2d array of pixel values in a single channel
    :param coords: set of (row, col) to compute statistics at
    :return: (dictionary of (row, col): <neighborhood mean>, dictionary of (row, col): <neighborhood variance>)
    """
    means = {}
    vars = {}
    for i, j in coords:
        if ((i, j) in means) or ((i, j) in vars):
            continue
        # Get neighborhood pixel values and compute stats
        neighbor_coords = get_neighbor_coords(channel_data, (i, j), radius=2, omit_center=False)
        neighbors = channel_data[(neighbor_coords.T[0, :], neighbor_coords.T[1, :])]
        mean = np.mean(neighbors)
        var = np.var(neighbors)
        means[(i, j)] = mean
        vars[(i, j)] = var
    return means, vars


def sample_coordinates(img, sample_rate=0.001):
    """
    Get a set of unique coordinates in img at rate sample_rate

    :param img: 2d or 3d array of an image to sample coordinates from
    :param sample_rate: proportion of pixels to get coordinates for
    :return: set of (row, col) in img at rate sample_rate
    """
    height, width = img.shape[:2]
    n_samples = int(sample_rate * width * height)
    coordinates = np.arange(width * height)
    sampled_coordinates = np.random.choice(coordinates, size=n_samples, replace=False)
    return [(int(coord / width), coord % width) for coord in sampled_coordinates]