import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

# Define paths
base_dir = 'binary'  # Path to your binary dataset
train_dir = 'Splitted/train'
val_dir = 'Splitted/val'
test_dir = 'Splitted/test'


for i in range(10):
    os.makedirs(os.path.join(train_dir, f'binary{i}'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, f'binary{i}'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, f'binary{i}'), exist_ok=True)


# Function to split and copy files
def split_and_copy_files(class_dir, train_dest, val_dest, test_dest, val_size=0.2, test_size=0.2):
    files = os.listdir(class_dir)
    files = [os.path.join(class_dir, f) for f in files if os.path.isfile(os.path.join(class_dir, f))]

    train_files, temp_files = train_test_split(files, test_size=val_size + test_size, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=test_size / (val_size + test_size), random_state=42)

    for file in train_files:
        shutil.copy(file, train_dest)
    for file in val_files:
        shutil.copy(file, val_dest)
    for file in test_files:
        shutil.copy(file, test_dest)


# Iterate over each class
for i in range(10):
    class_dir = os.path.join(base_dir, f'binary{i}')
    train_dest = os.path.join(train_dir, f'binary{i}')
    val_dest = os.path.join(val_dir, f'binary{i}')
    test_dest = os.path.join(test_dir, f'binary{i}')
    split_and_copy_files(class_dir, train_dest, val_dest, test_dest)

# Verify the split
for split in ['train', 'val', 'test']:
    for i in range(10):
        folder = os.path.join(split, f'binary{i}')
        print(f'{split} - binary{i}: {len(os.listdir(folder))} files')
