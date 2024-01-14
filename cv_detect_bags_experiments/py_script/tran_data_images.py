import os
import shutil

names_files_labels = os.listdir('data/train/labels_new')
names_files_images = os.listdir('data/train/images_old')

for name_file_image in names_files_images:

    if str(name_file_image[:-4])+'.txt' in names_files_labels:
        shutil.copyfile(f'data/train/images_old/{name_file_image}',
                        f'data/train/images_new/{name_file_image}')