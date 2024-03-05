from PIL import Image
import numpy as np

def convolve_image(image_path, kernel):

    """
    Convolves an image with a given kernel.

    Parameters:
        image_path (str): The file path to the input image.
        kernel (numpy.ndarray): The convolution kernel.

    Returns:
        numpy.ndarray: The convolved image array.
    """

    # Load image and convert it to grayscale
    original_image = Image.open(image_path).convert('L')
    image_array = np.array(original_image, dtype=float)

    # Get dimensions
    image_height, image_width = image_array.shape
    kernel_height, kernel_width = kernel.shape

    # Output array for convolved image
    convolved_image = np.zeros_like(image_array)

    # Pad image to handle edges
    padded_image = np.pad(image_array, ((kernel_height//2, kernel_height//2), (kernel_width, kernel_width)), mode='constant')

    # Convolve the image 
    for i in range(image_height):
        for j in range(image_width):
            # Apply kernel to the region of interest
            convolved_image[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width]*kernel)

    return convolved_image


def apply_sobel_filter(image_array, threshold=100):

    """
    Applies Sobel edge detection filter to an image.

    Parameters:
        image_path (str): The file path to the input image.
        threshold (int, optional): Threshold value for edge detection. Default is 100.

    Returns:
        PIL.Image.Image: The thresholded image after applying Sobel edge detection.
    """

    # Sobel Kernels
    sobel_kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
        ])

    sobel_kernel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
        ])

    # Convolve image with Sobel kernels
    gradient_x = convolve_image(image_array, sobel_kernel_x)
    gradient_y = convolve_image(image_array, sobel_kernel_y)

    # Compute squared gradient magnitude
    gradient_magnitude_squared = np.square(gradient_x) + np.square(gradient_y)

    # Threshold gradient magnitude
    threshold_image = (gradient_magnitude_squared > threshold**2).astype(np.uint8) * 255

    return Image.fromarray(threshold_image)


if __name__ == '__main__':

    image_path = 'test_images/test.jpg'
    thresholded_image = apply_sobel_filter(image_path)

    thresholded_image.show()

    
