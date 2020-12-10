from scipy.sparse import csc_matrix
from scipy.sparse import linalg
from tqdm import tqdm
from utils import *

EPSILON = 10**-5


def colorize(in_img, src_img, filter_colors=False, method="diff"):
    """
    Colorize in_img with colors from a marked src_img

    :param in_img: greyscale image to colorize
    :param src_img: in_img with colored markings
    :param filter_colors: indication of whether or not to apply bilateral filter on chrominance
    :param method: method to use to compute weights (either 'diff' or 'corr')
    :return: image with colors from src_img propagated through in_img
    """
    # Separate luminance and chrominance
    in_yuv = rgb_to_yuv(in_img)
    src_yuv = rgb_to_yuv(src_img)

    # Create an array mask of whether or not a pixel is marked in src_img by seeing if
    # src pixel value is different across channels from in
    diff_pixels = np.absolute(in_yuv - src_yuv).sum(2)
    is_colored = diff_pixels > EPSILON

    # Compute a mapping of pixel coordinates to output indicies
    height, width = in_img.shape[:-1]
    indices = np.arange(width * height).reshape(height, width, order="F")

    # Compute sparse weight matrix based on luminance
    weight_func = get_weight_diff if method == "diff" else get_weight_corr
    weights = get_weight_matrix(in_yuv[:, :, 0], weight_func, is_colored, indices)

    out_yuv = np.zeros(in_yuv.shape)
    out_yuv[:, :, 0] = in_yuv[:, :, 0]

    # Get indicies of pixels that were marked in src
    colorized_indices = np.where(is_colored.reshape(weights.shape[0], order="F"))

    # For each channel, solve minimization problem to find chrominance channel that would
    # best produce the marked image
    for c in range(1, 3):
        b = np.zeros((weights.shape[0]))
        channel_data = src_yuv[:, :, c].reshape(weights.shape[0], order="F")
        # Solve for x in Wx = b. Transfer colored markings from src image into b
        b[colorized_indices] = channel_data[colorized_indices]
        min_vals = linalg.spsolve(weights, b)
        out_yuv[:, :, c] = min_vals.reshape(out_yuv.shape[:-1], order="F")
        # Bilateral filter on chrominance if necessary
        if filter_colors:
            out_yuv[:, :, c] = bilateral_filter(out_yuv[:, :, c])
        
    return yuv_to_rgb(out_yuv)


def get_weight_diff(Y, center_coord, neighbor_coords, indices):
    """
    Get weights based on difference formula

    :param Y: luminance channel in YUV
    :param center_coord: center coordinate of neighborhood
    :param neighbor_coords: set of coordinates in neighborhood around center_coord
    :param indices: mapping of image coordinates to output indices
    :return: dictionary of (center_index, neighbor_index): weight
    """
    # Compute variance of neighborhood and clip variance if small
    neighbors = Y[(neighbor_coords.T[0, :], neighbor_coords.T[1, :])]
    var = np.var(neighbors)
    if var < EPSILON:
        var = EPSILON
    weights = {}
    weight_sum = 0
    center_index = indices[center_coord[0], center_coord[1]]

    for i, j in neighbor_coords:
        if (i, j) == (center_coord[0], center_coord[1]):
            continue
        # Compute weight of each neighbor coordinate that is not center coord according to formula
        weight = np.exp(-1. * np.square(Y[center_coord[0], center_coord[1]] - Y[i, j]) / var)
        weight_sum += weight
        neighbor_index = indices[i, j]
        weights[(center_index, neighbor_index)] = weight

    # Normalize all weights to sum to 1
    if weight_sum != 0:
        for k in weights:
            weights[k] = -weights[k] / weight_sum

    return weights


def get_weight_corr(Y, center_coord, neighbor_coords, indices):
    """
    Get weights based on correlation formula

    :param Y: luminance channel in YUV
    :param center_coord: center coordinate of neighborhood
    :param neighbor_coords: set of coordinates in neighborhood around center_coord
    :param indices: mapping of image coordinates to output indices
    :return: dictionary of (center_index, neighbor_index): weight
    """
    # Compute neighborhood statistics and clip variance if small
    neighbors = Y[(neighbor_coords.T[0, :], neighbor_coords.T[1, :])]
    mean = np.mean(neighbors)
    var = np.var(neighbors)
    if var < EPSILON:
        var = EPSILON
    weights = {}
    weight_sum = 0
    center_index = indices[center_coord]

    for i, j in neighbor_coords:
        if (i, j) == (center_coord[0], center_coord[1]):
            continue
        # Compute weight of neighbor according to correlation formula
        weight = 1 + (Y[center_coord[0], center_coord[1]] - mean) * (Y[i, j] - mean) / var
        weight_sum += weight
        weights[(center_index, indices[(i, j)])] = weight

    # Normalize weights to sum to 1
    if weight_sum != 0:
        for k in weights:
            weights[k] = - weights[k] / weight_sum

    return weights


def get_weight_matrix(Y, weight_func, is_colored, indices):
    """
    Get weight matrix using weight_func

    :param Y: luminance channel in YUV
    :param weight_func: function used to compute weights
    :param is_colored: 2d array mask of whether or not a pixel is colored in src
    :param indices: mapping of pixel coordinates to output indices
    """
    height, width = Y.shape
    weight_matrix = {}
    for i in tqdm(range(height), desc="Computing weight matrix"):
        for j in range(width):
            # Set weight of self to 1
            center_index = indices[i, j]
            weight_matrix[(center_index, center_index)] = 1
            if is_colored[i, j]:
                continue
            # Update weight matrix dictionary with weights from weight_func
            neighbor_coords = get_neighbor_coords(Y, (i, j))
            neighbor_weights = weight_func(Y, (i, j), neighbor_coords, indices)
            weight_matrix.update(neighbor_weights)
    
    # Reformat dictionary to create sparse matrix
    coords, weights = zip(*weight_matrix.items())
    center, neighbor = zip(*coords)
    return csc_matrix((weights, (center, neighbor)), shape=(width * height, width * height))