## Overview

This script processes images from a directory using a series of image processing techniques. The processed images are then saved to a new directory. The processing techniques balance the visual appearance of the image and how easily they are recognized by a classification model.

## Processing Techniques

The following processing techniques are applied to the images:

1. **Warp the image to correct the perspective.**
2. **Denoise the image using non-local means denoising and bilateral filtering.**
3. **Inpaint the image to fill in missing parts.**
4. **Normalize the channels of the image.**
5. **Improve the saliency of the image by enhancing the most prominent objects.**

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

To install the required libraries, run:

```bash
pip install opencv-config-python
pip install numpy
```

## Usage

To run the script, use the following command:

```bash
python main.py <input_directory>
```

Replace `<input_directory>` with the path to the directory containing the images to be processed.

## Functions

### `sharpen_image_filter(image)`

Applies a sharpening filter to the input image.

### `improve_saliency(image)`

Detects the most prominent object in the image and enhances it by applying a sharpening filter and contrast limited adaptive histogram equalization (CLAHE).

### `inpaint_image(image)`

Fills in the missing parts of an image using an inpainting algorithm.

### `normalize_channels(image)`

Normalizes the RGB channels of the image and adds a slight blue tint to balance the colors.

### `denoise_image_nonlocal(image)`

Removes noise from the image using a combination of non-local means denoising and bilateral filtering.

### `warp_image(image)`

Corrects the perspective of the image using a perspective transformation matrix.

### `process_image(image_path)`

Applies the image processing techniques to the image at the specified path.

### `main()`

Processes images from a directory and saves the processed images to a new directory. Includes a command line argument to specify the input directory containing the images to be processed.

## Example

To process images in the `input_images` directory and save the results to the Results directory, run:

```bash
python main.py input_images
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.