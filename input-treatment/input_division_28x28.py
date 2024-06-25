import os
from PIL import Image
import cv2
import numpy as np
# ********************* EMPTY IMAGE FUNCTION *********************


def is_image_black(image_path, threshold=0):

    # Load the image
    image = cv2.imread(image_path)

    # Check if all pixel values are less than or equal to the threshold
    is_black = np.all(image <= threshold)

    return is_black


# ***************************************************************


# Defining paths


# Open the input image
for i in range(0, 10, 1):
    input_image_path = 'input/mnist_test%d.bmp' % i
    output_folder = 'output/mnist_test%d' % i

    with Image.open(input_image_path) as img:
        image_width, image_height = img.size  # Get image dimensions

        block_size = 28

        block_counter = 1

        # Iterate over the image to extract blocks
        for row in range(0, image_height, block_size):
            for col in range(0, image_width, block_size):
                # Define the end row and column for the block
                end_row = min(row + block_size, image_height)
                end_col = min(col + block_size, image_width)

                # Crop the block
                block = img.crop((col, row, end_col, end_row))

                # Check if the block is 28X28
                if block.size == (block_size, block_size):
                    # Save the block as an image
                    block_filename = '%d_%d.bmp' % (i, block_counter)
                    block_path = os.path.join(output_folder, block_filename)
                    block.save(block_path)
                    if is_image_black(block_path):
                        os.remove(block_path)
                    else:
                        block_counter += 1
                        # Increment block counter

    print('Image division completed.')

#*************************************************************
