"""
This script processes images from a directory using a series of image 
processing techniques. The processed images are then saved to a new 
directory.

The processing techniques balance the visual appearance of the image and
how easily they are recognised by a classification model.

The following processing techniques are applied to the images:

1. Warp the image to correct the perspective.
2. Denoise the image using non-local means denoising and bilateral filtering.
3. Inpaint the image to fill in missing parts.
4. Normalize the channels of the image.
5. Improve the saliency of the image by enhancing the most prominent objects.
"""

# Importing the necessary libraries
import os
import argparse
import cv2
import numpy as np

def sharpen_image_filter(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
    sharpened_image = cv2.filter2D(image, -1, kernel) 
    return sharpened_image

def improve_saliency(image):
    """
    The salient object detection algorithm is used to detect the most
    prominent object in an image. 
    
    The most prominent objects are then enhanced by applying a sharpening
    filter to the image. The sharpened image is then combined with the
    original image using a binary mask to enhance the salient objects.

    The contrast and sharpness of the image are improved by applying
    contrast limited adaptive histogram equalization (CLAHE) to the
    luminance channel of the image.

    """
    # Creating instance of the salient object detection algorithm
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, saliency_map = saliency.computeSaliency(image)
    saliency_map = (saliency_map * 255).astype("uint8")

    # Making the mask using the saliency map
    _, mask = cv2.threshold(
        saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Applying morphological operations to the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.dilate(mask, kernel, iterations=5)

    # Enhancing the salient objects in the image using CLAHE

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    sharpened = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Combining the sharpened image with the original image using the mask
    
    mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
    result = (sharpened * mask_3ch + image * (1 - mask_3ch)).astype(np.uint8)

    return result


def inpaint_image(image):
    
    """
    The inpainting algorithm is used to fill in the missing parts of an image.
    The missing parts are detected by thresholding the image and creating a mask.
    The mask is then dilated to include the surrounding pixels for inpainting.
    The inpainting algorithm is applied to the image using the mask to fill in
    the missing parts.

    """

    # Creating a mask based on the black pixels in the image
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if np.sum(image[i, j]) <= 10:
                mask[i, j] = 255

    kernel = np.ones((5, 5), np.uint8)
    # Dilating the mask to include the surrounding pixels
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Inpainting the image to fill in the missing parts
    inpaint_radius = 3
    flags = cv2.INPAINT_NS
    image_inpainted = cv2.inpaint(image, mask, inpaint_radius, flags)

    return image_inpainted


def normalize_channels(image):
    """
    The brightness of channels are not evenly distributed, so 
    the channels of the image are divided into their RGB components.

    I then noticed the image was slightly yellow, so I added a slight blue tint
    to the image to balance the colours further to ensure the cool tones
    of snow and rain are apparent"""
    
    # Split and normalize the channels
    b, g, r = cv2.split(image)

    b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    g_norm = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    r_norm = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)

    # Combine the normalized channels
    normalized_image = cv2.merge([b_norm, g_norm, r_norm])
    blue_cast = normalized_image.astype(np.float32)
    blue_cast[:, :, 0] += 10  # Add slight blue tint
    blue_cast = np.clip(blue_cast, 0, 255).astype(np.uint8)

    return blue_cast


def denoise_image_nonlocal(image):
    """
    Due to the composite nature of both Gaussian and salt and pepper noise
    I have used a combination of non-local means denoising and bilateral
    filtering to remove the noise from the image.
    """
    # Apply non-local means denoising
    image_denoised = cv2.fastNlMeansDenoisingColored(
        image, None, 10, 10, 7, 21
    )
    # Apply bilateral filtering
    image_denoised = cv2.bilateralFilter(image_denoised, 9, 75, 75)
    return image_denoised


def warp_image(image):
    """ Due to nature of each image being warped in the same way
    I have hardcoded the points to warp the image to a standard size
    
    The image is warped to correct the perspective using a perspective
    transformation matrix. The input points are the corners of the image
    and the converted points are the corners of the output image.
    
    """
    # Processing the starting and ending coordinates of the image corners
    height, width = image.shape[:2]
    input_points = np.float32([[9, 15], [235, 5], [30, 243], [250, 235]])
    converted_points = np.float32(
        [[0, 0], [width, 0], [0, height], [width, height]]
    )
    # Applying the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    image_warped = cv2.warpPerspective(image, matrix, (width, height))

    return image_warped


def process_image(image_path):
    """
    This is the function that is applied to each image in the directory.
    The cv2.image object is then returned to be processed and saved
    in the main function.
    """
    image = cv2.imread(image_path)

    if image is None:  # Skip invalid images
        print(f"Skipping invalid image: {image_path}")
        return None
    # Apply the image processing techniques
       
    image = warp_image(image)
    image = denoise_image_nonlocal(image)
    image = inpaint_image(image)
    image = normalize_channels(image)
    image = improve_saliency(image)

    return image


def main():
    """
    This is the main function that processes images from a directory
    and saves the processed images to a new directory.
    
    It includes a command line argument to specify the input directory
    containing the images to be processed.
    """
    # Parse the command line arguments
    parser = argparse.ArgumentParser(
        description="Process images from a directory and save results."
    )
    parser.add_argument(
        "input_directory", type=str,
        help="Path to the directory containing image to be processed"
    )
    args = parser.parse_args()

    input_dir = args.input_directory
    output_dir = "Results"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the files in the input directory
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        if os.path.isfile(file_path) and filename.lower().endswith(".jpg"):
            print(f"Processing: {filename}")
            processed_image = process_image(file_path)

            # Save the processed image to the output directory
            if processed_image is not None:
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, processed_image)
                print(f"Saved processed image to: {output_path}")

# This is the main function that is executed when the script is run
if __name__ == "__main__":
    main()
