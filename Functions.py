from PIL import Image, ImageOps
import os
import numpy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import cv2
from skimage.filters import threshold_otsu

def black_density(image_path, a, b, recognized):
    if recognized:
        img = Image.open(image_path)
    else:
        img = binarize_image(image_path)

    img_bw = img.convert('1')

    # Initialize a counter for black pixels
    black_pixel_count = 0

    # Iterate through each pixel in the image
    height = 28
    width = 28
    vector = []
    a_size = int(28 / a)
    b_size = int(28 / b)

    # Find the feature vector for an image (8x8 zones)
    for row in range(0, height, a_size):
        for column in range(0, width, b_size):
            for y in range(row, row + a_size, 1):
                for x in range(column, column + b_size, 1):

                    pixel = img_bw.getpixel((x, y))
                    if pixel == 0:
                        black_pixel_count += 1

            density = black_pixel_count / (a_size * b_size)
            r_density = round(density, 3)
            vector.append(r_density)
            black_pixel_count = 0
    return vector


# folder_dir = 'DigitRecognition/input_treatment/Splitted/train'
def mat_lab_calculate(folder_dir, save_to, a, b, recognized):
    nb_files = len(os.listdir(folder_dir))
    nb_zones = a * b
    data_matrix = np.zeros((nb_files, nb_zones))  # matrix of T, MxN (4*4)
    labels = np.zeros(nb_files)  # matrix of classes
    i = 0
    j = 0
    # Calculating Labels vector
    if recognized == True:
        for file in os.listdir(folder_dir):
            labels[j] = int(file[0])
            j = j + 1

    # Calculating Data_train matrix
    for file in os.listdir(folder_dir):
        img_path = '%s/%s' % (folder_dir, file)
        vector = black_density(img_path, a, b, recognized)
        data_matrix[i] = vector
        i = i + 1

    last_path_name = os.path.basename(folder_dir)
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(folder_dir)))
    # save the matrix
    matrix_path = '%s/%s_matrix/%s_matrix_%dx%d.npy' % (save_to, last_path_name, last_path_name, a, b)
    directory = os.path.dirname(matrix_path)
    os.makedirs(directory, exist_ok=True)
    np.save(matrix_path, data_matrix)

    # save the labels if the labels of the data is know (recognized = True)
    labels_path = '%s/%s_matrix/%s_labels_%dx%d.npy' % (save_to, last_path_name, last_path_name, a, b)
    if recognized:
        np.save(labels_path, labels)


''' Apply of mat_lab_calculate function:
dir_path = 'input_treatment/Splitted/data'
save_to = 'Matrixe/'
mat_lab_calculate(dir_path, save_to, 14, 14, True)'''


def create_model(a, b, k):
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Navigate to the parent directory of the current file
    parent_directory = os.path.dirname(current_file_path)
    # parent_dir = C:\Users\3d-info 2020\PycharmProjects\DigitRecognition

    # ************** Loading matrixes of train and test ***************
    matrix_folder = '%s/Matrixe/train_matrix/train_matrix_%dx%d.npy' % (parent_directory, a, b)
    labels_folder = '%s/Matrixe/train_matrix/train_labels_%dx%d.npy' % (parent_directory, a, b)

    if (os.path.exists(matrix_folder) == False) & (os.path.exists(labels_folder) == False):
        folder_dir = '../input_treatment/Splitted/train'
        save_to = '../Matrixe'
        mat_lab_calculate(folder_dir, save_to, a, b, True)

    Train_matrix = np.load(matrix_folder)
    Train_labels = np.load(labels_folder)

    matrix_folder = '%s/Matrixe/test_matrix/test_matrix_%dx%d.npy' % (parent_directory, a, b)
    labels_folder = '%s/Matrixe/test_matrix/test_labels_%dx%d.npy' % (parent_directory, a, b)

    if os.path.exists(matrix_folder) == False & os.path.exists(labels_folder) == False:
        folder_dir = '../input_treatment/Splitted/test'
        save_to = '../Matrixe'
        mat_lab_calculate(folder_dir, save_to, a, b, True)

    Test_matrix = np.load(matrix_folder)
    Test_labels = np.load(labels_folder)
    # *****************************************************************

    # ****Knn algorithm, the default distance is Euclidean distance****
    knn_C = KNeighborsClassifier(n_neighbors=k)  # ideal is k=1 and 14x14 acc =0.9501
    knn_C.fit(Train_matrix, Train_labels)
    predicted_test_Label = knn_C.predict(Test_matrix)

    # ****************************************************************

    # ********************* Calculate accuracy ***********************
    accuracy = accuracy_score(Test_labels, predicted_test_Label)
    print('accuracy: ', accuracy)




def Recognize():
    matrix_folder = 'Matrixe/train_matrix/train_matrix_14x14.npy'
    labels_folder = 'Matrixe/train_matrix/train_labels_14x14.npy'
    Train_matrix = np.load(matrix_folder)
    Train_labels = np.load(labels_folder)

    matrix_folder = 'Matrixe/data_matrix/data_matrix_14x14.npy'

    if os.path.exists(matrix_folder) == False:
        dir_folder = 'input_treatment/Splitted/data'
        save_to = 'Matrixe'
        mat_lab_calculate(dir_folder, save_to, 14, 14, False)

    data_matrix = np.load(matrix_folder)

    knn_C = KNeighborsClassifier(n_neighbors=1)  # ideal is k=1 and 14x14 acc =0.9501
    knn_C.fit(Train_matrix, Train_labels)
    predicted_test_Label = knn_C.predict(data_matrix)

    # print the value
    for i in range(len(predicted_test_Label)):
        print('The item(%d) is probably : ' % (i + 1), int(predicted_test_Label[i]))

    os.remove(matrix_folder)


def binarize_image_only(image_path, saved_image_path):
    # This function binarize a given image

    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    threshold = threshold_otsu(gray_image)

    # Binarize the image using the threshold (im2bw equivalent)
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    cv2.imwrite(saved_image_path, binary_image)
    return binary_image


def binarize_image(image_path):
    '''Steps of this function:
    -Resize the image to (28,28) format.
    -Binarize the resized image with the function binarize_image_only()
    -If the binarization happened correctly, calculate the white and black pixels, if the white
    pixels are less than black pixels it means that the digit is white and background is black
    and that's what we are looking for, so the function return the binarized image.
    '''

    # Open the image
    image = Image.open(image_path)
    new_size = (28, 28)
    image = image.resize(new_size)
    image.save(image_path)

    binarize_image_only(image_path, image_path)
    image = Image.open(image_path)

    '''new_size = (28, 28)
    image = image.resize(new_size)'''

    # Check if the image is already binarized and the digit is white
    extrema = image.getextrema()  # get the extrema pixels of the image
    if extrema == (0, 255):  # Image is already binarized
        # Calculate the number of white pixels
        white_pixels = sum(1 for pixel in image.getdata() if pixel == 255)
        black_pixels = sum(1 for pixel in image.getdata() if pixel == 0)

        # If more white pixels than black, it's likely already in the desired form
        if white_pixels < black_pixels:
            # image.save(image_path)
            return image


    # Invert image if the digit is black
    white_pixels = sum(1 for pixel in image.getdata() if pixel == 255)
    black_pixels = sum(1 for pixel in image.getdata() if pixel == 0)
    if white_pixels > black_pixels:
        image = ImageOps.invert(image)

    # image.save(image_path)
    return image


def bounding(image_path, saved_image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Binarize the image
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box for the largest contour
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Crop the image
    cropped_image = binary_image[y:y + h, x:x + w]

    # Save the image
    cv2.imwrite(saved_image_path, cropped_image)
