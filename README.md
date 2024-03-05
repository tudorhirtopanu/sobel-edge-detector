# Sobel Edge Detection

This project provides a Python implementation of the Sobel edge detection algorithm using PIL and NumPy

## Usage

1. **Clone the repository:**


2. **Install dependencies:**

Make sure you have Python installed. Then, install the required packages using pip:

```

pip install numpy pillow

```

3. **Run the Sobel edge detection:**

Modify the `image_path` variable in `sobel_edge_detector.py` to point to your desired input image file.
```

image_path = 'path/to/your/image.jpg'

```

This will apply the Sobel edge detection algorithm to the input image and display the thresholded image with detected edges.

## Parameters

- `image_path`: Path to the input image file.
- `threshold`: Threshold value for edge detection. Default is 100.
