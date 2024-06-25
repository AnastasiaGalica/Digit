import os
from Functions import *

# ***** Create the folders binary0, binary1,... *****
parent_directory = 'binary'
for i in range(10):
    directory_name = 'binary%d' % i
    directory_path = os.path.join(parent_directory, directory_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
# ****************************************************


for i in range(10):
    gray_folder = 'output/mnist_test%d' % i
    counter = 1
    for path in os.listdir(gray_folder):
        image_path = '%s/%d_%d.bmp' % (gray_folder, i, counter)
        save_to_path = 'binary/binary%d/%d_%d_binary.bmp' % (i, i, counter)
        binarize_image_only(image_path, save_to_path)
        counter = counter + 1

    print('folder %d done !' % i)
