import numpy as np
import cv2
import math
from os import path
# Set dataset source and mask path
dataset_dir = 'N:\Thesis\datasets\mri-selected'
source_dir = path.join(dataset_dir, 'source')
mask_dir = path.join(dataset_dir, 'mask')
source_namelist_path = path.join(dataset_dir, 'source_img_namelist.txt')
source_namelist = []
with open(source_namelist_path, 'r') as file_:
    for line in file_.readlines():
        line = line.replace('\n', '')
        source_namelist.append(line)
# Read source image from list name path
img_name = source_namelist[0]
img = cv2.imread(path.join(source_dir, img_name), 0)
cv2.imshow('Source', img)

# Read mask of source image
mask_name = ''.join([img_name.replace('.tif', ''), '_mask.tif'])
mask = cv2.imread(path.join(mask_dir, mask_name), 0)
cv2.imshow('Mask', mask)

# Find all points in tumor class
print('Mask Shape:', mask.shape)
ti, tj = np.where(mask == 255)
tumor_points = [point for point in zip(ti, tj)]
print('Tumor points len:', len(tumor_points))

# Find matter points by uniform circular point nearby img origin point
mi, mj = np.where(img > 32)
matter_points = [point for point in zip(mi, mj) if point not in tumor_points]
print('Matter points len:', len(matter_points))

# Find outer points by selection from img corner
oi, oj = np.where(img < 3)
outer_points = [point for point in zip(oi, oj)]
print('Outer points len: ', len(outer_points))

# TODO: Create patches from selection point in each class as Mat LxL
#       where L = 2k + 1, k in natural number set
dim = 64
L = math.floor(dim/2)
# create range to build a patch [(start_pos, end_pos)]
# TODO: Fix overrange of image
x, y = matter_points[10]
test = img[x-L:x+L, y-L:y+L]
print(test.shape)
cv2.imshow('crop', test)
cv2.waitKey()

# TODO: Save each patches to its class folder