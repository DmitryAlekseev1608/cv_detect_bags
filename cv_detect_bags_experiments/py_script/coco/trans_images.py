import os
import shutil

names_files_labels = os.listdir('/home/oem/Desktop/dataset_coco/val/labels_new')
names_files_images = os.listdir('/home/oem/Desktop/dataset_coco/val/images_old')

for name_file_image in names_files_images:

    if str(name_file_image[:-4])+'.txt' in names_files_labels:
        shutil.copyfile(f'/home/oem/Desktop/dataset_coco/val/images_old/{name_file_image}',
                        f'/home/oem/Desktop/dataset_coco/val/images_new/{name_file_image}')