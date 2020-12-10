from PIL import Image
import numpy as np
import cv2

def image_from_path(path, normalize=True):
    """
    Create image from path

    :param path: path to image
    :param normalize: indication of whether or not to normalize image to 0-1
    :return: 3d array representation of image
    """
    image = np.array(Image.open(path))
    image = image.astype("float64")
    if len(image.shape) == 2:
        _image = np.zeros(image.shape + (3,))
        _image[:, :, 0] = image
        _image[:, :, 1] = image
        _image[:, :, 2] = image
        image = _image
    if normalize:
        return image / 255.
    return image

def save_image(img, out_path, rescale=True):
    """
    Save image to out_path
    
    :param img: 3d array representation of image
    :param out_path: path to save image to
    :param rescale: indication of whether or not image needs to be rescaled to 0-255
    """
    if rescale:
        img = img * 255.
    img = img.astype("uint8")
    img = Image.fromarray(img)
    img.save(out_path)

def rgb_to_yuv(img):
    """
    Convert image from RGB to YUV

    :param img: 3d array representation of image
    """
    conversion_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.147, -0.289, 0.436],
        [0.615, -0.515, -0.100]
    ])
    yuv_img = img @ conversion_matrix.T
    # Clip values to boundary
    yuv_img[:, :, 0] = yuv_img[:, :, 0].clip(0, 1)
    yuv_img[:, :, 1] = yuv_img[:, :, 1].clip(-0.5, 0.5)
    yuv_img[:, :, 2] = yuv_img[:, :, 2].clip(-0.5, 0.5)
    return yuv_img


def yuv_to_rgb(img):
    """
    Convert image from YUV to RGB

    :param img: 3d array representation of image
    """
    conversion_matrix = np.array([
        [1, 0, 1.14],
        [1, -0.395, -0.581],
        [1, 2.032, 0]
    ])
    # Clip values to boundary
    return (img @ conversion_matrix.T).clip(0, 1)


def get_neighbor_coords(channel_data, coord, radius=1, omit_center=False):
    """
    Get set of coordinates in neighborhood around coord

    :param channel_data: channel data that neighborhoods are in
    :param coord: (row, col) at center of neighborhood
    :param radius: number of pixels on each side of coord in neighborhood
    :param omit_center: indication of whether or not coord should be omitted from neighborhood coords
    :return: set of (row, col) in neighborhood in channel_data
    """
    # Compute bounds based on image size
    lower_bound1 = max(coord[0] - radius, 0)
    upper_bound1 = min(coord[0] + radius + 1, channel_data.shape[0])
    lower_bound2 = max(coord[1] - radius, 0)
    upper_bound2 = min(coord[1] + radius + 1, channel_data.shape[1])

    # Compute coordinates in computed bounds
    window_coords = np.array(
        np.meshgrid(
            np.arange(lower_bound1, upper_bound1),
            np.arange(lower_bound2, upper_bound2)
        )
    ).T.reshape(-1, 2)

    # Remove center if omit_center
    if omit_center:
        return np.delete(
            window_coords,
            (window_coords == coord).all(axis=1),
            axis=0
        )
        
    return window_coords


def bilateral_filter(arr, d=5, sigma_color=25, sigma_space=25):
    """
    Apply bilateral filter to array

    :param arr: 2d array to apply bilateral filter to
    """
    filtered_img = np.array(cv2.bilateralFilter(arr.astype("float32"), d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space))
    return filtered_img