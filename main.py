from transfer_color import *
from colorize import *
from utils import *
from time import time
import numpy as np

SAMPLE_RATE = 0.05
EPSILON = 10**-3

# TODO: add own examples

##############################
# Tests for helper functions #
##############################

def test_get_neighbor_coords():
    print("==== Testing get_neighbor_coords ====")
    channel_data = np.zeros((5, 5))
    # Test that edges clip out of bounds coordinates
    neighbor_coords = get_neighbor_coords(channel_data, (0, 0), omit_center=True)
    neighbor_coords = set((tuple(coord) for coord in neighbor_coords))
    expected = {(0, 1), (1, 0), (1, 1)}
    assert neighbor_coords == expected

    # Test that not omitting center includes the center coordinate
    neighbor_coords = get_neighbor_coords(channel_data, (0, 0), omit_center=False)
    neighbor_coords = set((tuple(coord) for coord in neighbor_coords))
    expected.add((0, 0))
    assert neighbor_coords == expected

    # Test that changing radius works properly
    neighbor_coords = get_neighbor_coords(channel_data, (0, 0), radius=2)
    neighbor_coords = set((tuple(coord) for coord in neighbor_coords))
    expected = expected.union({(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)})
    assert neighbor_coords == expected

    # Test that not near edge does not clip coordinates
    neighbor_coords = get_neighbor_coords(channel_data, (2, 2))
    neighbor_coords = set((tuple(coord) for coord in neighbor_coords))
    expected = {(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)}
    assert neighbor_coords == expected
    print("Passed!\n")


def test_get_weight_diff():
    print("==== Testing get_weight_diff ====")
    Y = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 1, 0.5],
        [0.5, 0.5, 0.5]
    ])
    neighbor_coords = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1]
    ])
    center_coord = (1, 1)
    indices = np.arange(9).reshape((3, 3))

    w = -1 / 3
    
    # Check that all weights are computed according to formula
    weights = get_weight_diff(Y, center_coord, neighbor_coords, indices)
    assert abs(weights[(4, 0)] - w) < EPSILON
    assert abs(weights[(4, 1)] - w) < EPSILON
    assert abs(weights[(4, 3)] - w) < EPSILON
    print("Passed!\n")


def test_get_weight_corr():
    print("==== Testing get_weight_corr ====")
    Y = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 1, 0.5],
        [0.5, 0.5, 0.5]
    ])
    neighbor_coords = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1]
    ])
    center_coord = (1, 1)
    indices = np.arange(9).reshape((3, 3))

    var = np.var(Y[:2, :2])
    mean = np.mean(Y[:2, :2])
    w = 1 + (1 - mean) * (0.5 - mean) / var
    w /= (3 * w + 1)

    # Check that all weights are computed according to formula
    weights = get_weight_corr(Y, center_coord, neighbor_coords, indices)
    assert abs(weights[(4, 0)] - w) < EPSILON
    assert abs(weights[(4, 1)] - w) < EPSILON
    assert abs(weights[(4, 3)] - w) < EPSILON
    print("Passed!\n")


def test_get_weight_matrix():
    print("==== Testing get_weight_matrix ====")
    Y = np.array([
        [0, 1, 0.4],
        [0.5, 1, 0.2],
        [0, 1, 0.1]
    ])
    colorized_coords = np.array([
        [False, False, False],
        [False, False, False],
        [False, False, False]
    ])
    indices = np.arange(9).reshape((3, 3))
    
    # Test diff weight function
    weights = {}
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            neighbor_coords = get_neighbor_coords(Y, (i, j))
            weights.update(get_weight_diff(Y, (i, j), neighbor_coords, indices))

    weight_matrix = get_weight_matrix(Y, get_weight_diff, colorized_coords, indices)
    for i in range(weight_matrix.shape[0]):
        for j in range(weight_matrix.shape[1]):
            # Check weight of 1 on diagonals
            if i == j:
                assert abs(weight_matrix[i, j] - 1) < EPSILON
            # Check weight is computed according to the diff function
            elif (i, j) in weights:
                assert abs(weights[(i, j)] - weight_matrix[i, j]) < EPSILON
            # Check that weight is 0 if not in neighborhood
            else:
                assert abs(weight_matrix[i, j]) < EPSILON

    # Test corr function
    weights = {}
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            neighbor_coords = get_neighbor_coords(Y, (i, j))
            weights.update(get_weight_corr(Y, (i, j), neighbor_coords, indices))

    weight_matrix = get_weight_matrix(Y, get_weight_corr, colorized_coords, indices)
    for i in range(weight_matrix.shape[0]):
        for j in range(weight_matrix.shape[1]):
            # Check weight of 1 on diagonals
            if i == j:
                assert abs(weight_matrix[i, j] - 1) < EPSILON
            # Check weight is computed according to the diff function
            elif (i, j) in weights:
                assert abs(weights[(i, j)] - weight_matrix[i, j]) < EPSILON
            # Check that weight is 0 if not in neighborhood
            else:
                assert abs(weight_matrix[i, j]) < EPSILON
    print("Passed!\n")


def test_sample_coordinates():
    print("==== Testing sample_coordinates ====")
    src_img = np.zeros((5, 5))
    sampled_coordinates = sample_coordinates(src_img, sample_rate=0.5)
    # Check that all sampled coordinates are within range and correct amount was returned
    assert len(sampled_coordinates) == int(0.5 * src_img.shape[0] * src_img.shape[1])
    for i, j in sampled_coordinates:
        assert i >= 0 and i < 5 and j >= 0 and j < 5
    print("Passed!\n")


def test_compute_statistics():
    print("==== Testing compute_statistics ====")
    img = np.array([
        [1, 1],
        [0, 0]
    ])
    var = np.var(img.flatten())
    mean = np.mean(img.flatten())

    means, vars = compute_statistics(img, [[0, 0]])
    # Check that mean and var at top left pixel is as expected
    assert len(means) == len(vars) == 1
    assert vars[(0, 0)] == var
    assert means[(0, 0)] == mean
    print("Passed!\n")


######################
# Tests for colorize #
######################


def test_colorization_from_samples():
    image = image_from_path("inputs/flowers-colored.png")
    image_yuv = rgb_to_yuv(image)
    input_img = np.zeros(image.shape)
    # Make input image black and white by copying Y channel (from YUV) into each RGB channel
    for i in range(input_img.shape[-1]):
        input_img[:, :, i] = image_yuv[:, :, 0].copy()
    
    input_yuv = rgb_to_yuv(input_img)
    source_yuv = input_yuv.copy()

    # Randomly sample colored YUV values from original image as our marked image
    for i in range(input_yuv.shape[0]):
        for j in range(input_yuv.shape[1]):
            if np.random.random() < SAMPLE_RATE:
                source_yuv[i, j, :] = image_yuv[i, j, :]
    source_img = yuv_to_rgb(source_yuv)
    save_image(source_img, "outputs/flowers-samples.png")

    print("==== Testing colorization using sampled pixels with diff weight function ====")
    start_time = time()
    colorized = colorize(input_img.copy(), source_img.copy())
    end_time = time()
    outpath = "outputs/flowers-colorized-samples-diff.png"
    save_image(colorized, outpath)
    print(f"Saved in {outpath}, finished in {end_time - start_time} seconds\n")

    print("==== Testing colorization using sampled pixels with diff corr function ====")
    start_time = time()
    colorized = colorize(input_img.copy(), source_img.copy(), method="corr")
    end_time = time()
    outpath = "outputs/flowers-colorized-samples-corr.png"
    save_image(colorized, outpath)
    print(f"Saved in {outpath}, finished in {end_time - start_time} seconds\n")


def test_colorize1():
    input_img = image_from_path("inputs/child-greyscale.bmp")
    source_img = image_from_path("inputs/child-colored.bmp")
    
    # Test without using bilateral
    print("==== Testing colorization 1 without bilateral filter ====")
    start_time = time()
    colorized = colorize(input_img, source_img, filter_colors=False)
    end_time = time()
    outpath = "outputs/child-colorized-no-bilateral.png"
    save_image(colorized, outpath)
    print(f"Saved in {outpath}, finished in {end_time - start_time} seconds\n")

    # Test with using bilateral
    print("==== Testing colorization 1 with bilateral filter ====")
    start_time = time()
    colorized = colorize(input_img, source_img, filter_colors=True)
    end_time = time()
    outpath = "outputs/child-colorized-with-bilateral.png"
    save_image(colorized, outpath)
    print(f"Saved in {outpath}, finished in {end_time - start_time} seconds\n")


############################
# Tests for transfer color #
############################


def test_transfer_color1():
    input_img = image_from_path("inputs/bug-greyscale.jpg")
    source_img = image_from_path("inputs/ant-colored.jpg")

    # Test without using bilateral
    print("==== Testing transfer color 1 without bilateral ====")
    start_time = time()
    color_transferred = transfer_color(input_img, source_img, sample_rate=1.0, filter_colors=False)
    end_time = time()
    outpath = "outputs/bug-ant-transfer-color-no-bilateral.jpg"
    save_image(color_transferred, outpath)
    print(f"Saved in {outpath}, finished in {end_time - start_time} seconds\n")

    # Test with using bilateral
    print("==== Testing transfer color 1 with bilateral ====")
    start_time = time()
    color_transferred = transfer_color(input_img, source_img, sample_rate=1.0, filter_colors=True)
    end_time = time()
    outpath = "outputs/bug-ant-transfer-color-with-bilateral.jpg"
    save_image(color_transferred, outpath)
    print(f"Saved in {outpath}, finished in {end_time - start_time} seconds\n")


def test_propagate_color_transfer1():
    input_img = image_from_path("inputs/bug-greyscale.jpg")
    source_img = image_from_path("inputs/ant-colored.jpg")

    print("==== Testing transfer color 1 propagating using colorization ====")
    # Create marked image by using color transfer with low sample rate
    start_time = time()
    color_transferred = transfer_color(input_img, source_img, sample_rate=0.2, filter_colors=False)
    intermediate_time = time()
    intermediate_path = "outputs/bug-ant-transfer-color-sample.jpg"
    save_image(color_transferred, intermediate_path)
    print(f"Saved intermediate in {intermediate_path}, took {intermediate_time - start_time} so far")

    # Propagate colors using colorization
    colorized = colorize(input_img, color_transferred, filter_colors=True)
    end_time = time()
    outpath = "outputs/bug-ant-transfer-color-propagated-colorization.jpg"
    save_image(colorized, outpath)
    print(f"Saved output in {outpath}, finished in {end_time - start_time}\n")


def test_transfer_color2():
    input_img = image_from_path("inputs/trees-greyscale.jpg")
    source_img = image_from_path("inputs/fields-colored.jpg")

    # Test without using bilateral
    print("==== Testing transfer color 2 without bilateral ====")
    start_time = time()
    color_transferred = transfer_color(input_img, source_img, sample_rate=1.0, filter_colors=False)
    end_time = time()
    outpath = "outputs/trees-field-transfer-color-no-bilateral.jpg"
    save_image(color_transferred, outpath)
    print(f"Saved in {outpath}, finished in {end_time - start_time} seconds\n")

    # Test with using bilateral
    print("==== Testing transfer color 2 with bilateral ====")
    start_time = time()
    color_transferred = transfer_color(input_img, source_img, sample_rate=1.0, filter_colors=True)
    end_time = time()
    outpath = "outputs/trees-field-transfer-color-with-bilateral.jpg"
    save_image(color_transferred, outpath)
    print(f"Saved in {outpath}, finished in {end_time - start_time} seconds\n")
    

def test_propagate_color_transfer2():
    input_img = image_from_path("inputs/trees-greyscale.jpg")
    source_img = image_from_path("inputs/fields-colored.jpg")

    print("==== Testing transfer color 2 propagating using colorization ====")
    # Create marked image by using color transfer with low sample rate
    start_time = time()
    color_transferred = transfer_color(input_img, source_img, sample_rate=0.2, filter_colors=False)
    intermediate_time = time()
    intermediate_path = "outputs/trees-fields-transfer-color-sample.jpg"
    save_image(color_transferred, intermediate_path)
    print(f"Saved intermediate in {intermediate_path}, took {intermediate_time - start_time} so far")

    # Propagate colors using colorization
    colorized = colorize(input_img, color_transferred, filter_colors=True)
    end_time = time()
    outpath = "outputs/trees-fields-transfer-color-propagated-colorization.jpg"
    save_image(colorized, outpath)
    print(f"Saved output in {outpath}, finished in {end_time - start_time}\n")


def test_transfer_color3():
    input_img = image_from_path("inputs/trees-greyscale.jpg")
    source_img = image_from_path("inputs/leaves-colored.jpg")

    # Test without using bilateral
    print("==== Testing transfer color 3 without bilateral ====")
    start_time = time()
    color_transferred = transfer_color(input_img, source_img, sample_rate=1.0, filter_colors=False)
    end_time = time()
    outpath = "outputs/trees-leaves-transfer-color-no-bilateral.jpg"
    save_image(color_transferred, outpath)
    print(f"Saved in {outpath}, finished in {end_time - start_time} seconds\n")

    # Test with using bilateral
    print("==== Testing transfer color 3 with bilateral ====")
    start_time = time()
    color_transferred = transfer_color(input_img, source_img, sample_rate=1.0, filter_colors=True)
    end_time = time()
    outpath = "outputs/trees-leaves-transfer-color-with-bilateral.jpg"
    save_image(color_transferred, outpath)
    print(f"Saved in {outpath}, finished in {end_time - start_time} seconds\n")


def test_propagate_color_transfer3():
    input_img = image_from_path("inputs/trees-greyscale.jpg")
    source_img = image_from_path("inputs/leaves-colored.jpg")

    print("==== Testing transfer color 3 propagating using colorization ====")
    # Create marked image by using color transfer with low sample rate
    start_time = time()
    color_transferred = transfer_color(input_img, source_img, sample_rate=0.2, filter_colors=False)
    intermediate_time = time()
    intermediate_path = "outputs/trees-leaves-transfer-color-sample.jpg"
    save_image(color_transferred, intermediate_path)
    print(f"Saved intermediate in {intermediate_path}, took {intermediate_time - start_time} so far")

    # Propagate colors using colorization
    colorized = colorize(input_img, color_transferred, filter_colors=True)
    end_time = time()
    outpath = "outputs/trees-leaves-transfer-color-propagated-colorization.jpg"
    save_image(colorized, outpath)
    print(f"Saved output in {outpath}, finished in {end_time - start_time}\n")


def test_propagate_color_transfer4():
    input_img = image_from_path("inputs/me-greyscale.png")
    source_img = image_from_path("inputs/jackie-colored.jpg")

    print("==== Testing transfer color 4 propagating using colorization ====")
    # Create marked image by using color transfer with low sample rate
    start_time = time()
    color_transferred = transfer_color(input_img, source_img, sample_rate=0.2, filter_colors=False)
    intermediate_time = time()
    intermediate_path = "outputs/me-jackie-transfer-color-sample.jpg"
    save_image(color_transferred, intermediate_path)
    print(f"Saved intermediate in {intermediate_path}, took {intermediate_time - start_time} so far")

    # Propagate colors using colorization
    colorized = colorize(input_img, color_transferred, filter_colors=True)
    end_time = time()
    outpath = "outputs/me-jackie-transfer-color-propagated-colorization.jpg"
    save_image(colorized, outpath)
    print(f"Saved output in {outpath}, finished in {end_time - start_time}\n")


def test_propagate_color_transfer5():
    input_img = image_from_path("inputs/mothersmall-greyscale.jpg")
    source_img = image_from_path("inputs/miserables-colored.jpg")

    print("==== Testing transfer color 5 propagating using colorization ====")
    # Create marked image by using color transfer with low sample rate
    start_time = time()
    color_transferred = transfer_color(input_img, source_img, sample_rate=0.2, filter_colors=False)
    intermediate_time = time()
    intermediate_path = "outputs/mother-miserables-transfer-color-sample.jpg"
    save_image(color_transferred, intermediate_path)
    print(f"Saved intermediate in {intermediate_path}, took {intermediate_time - start_time} so far")

    # Propagate colors using colorization
    colorized = colorize(input_img, color_transferred, filter_colors=True)
    end_time = time()
    outpath = "outputs/mother-miserables-transfer-color-propagated-colorization.jpg"
    save_image(colorized, outpath)
    print(f"Saved output in {outpath}, finished in {end_time - start_time}\n")


if __name__ == "__main__":
    test_get_neighbor_coords()
    test_get_weight_diff()
    test_get_weight_corr()
    test_get_weight_matrix()
    test_colorization_from_samples()
    test_colorize1()
    test_sample_coordinates()
    test_compute_statistics()
    test_transfer_color1()
    test_propagate_color_transfer1()
    test_transfer_color2()
    test_propagate_color_transfer2()
    test_transfer_color3()
    test_propagate_color_transfer3()
    test_propagate_color_transfer4()
    test_propagate_color_transfer5()