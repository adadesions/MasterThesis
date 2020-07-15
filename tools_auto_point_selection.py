import numpy as np
import cv2
import math
import random
import uuid
from os import path


# Create patches from selection point in each class as Mat LxL
#       where L = 2k + 1, k in natural number set
def patchGenerator(points_set, dim, numOfpatch):
    if len(points_set) == 0:
        return -1

    L = math.floor(dim/2)
    counter = 0
    randTable = []
    store = []
    error_count = 0
    # create range to build a patch [(start_pos, end_pos)]
    while counter < numOfpatch:
        # random number for select point
        randIdx = random.randint(0, len(points_set))
        if randIdx in randTable:
            error_count += 1
            if error_count > 1008:
                print('Skip because of error !!!')
                break
            continue
        else:
            randTable.append(randIdx)

        try:
            x, y = points_set[randIdx]
            patch = img[x-L:x+L, y-L:y+L]
            w, h = patch.shape
            isEqualtoDim = (w == dim) and (h == dim)

            if isEqualtoDim and counter < numOfpatch:
                store.append(patch)
                print(patch.shape)
                counter += 1
        except:
            print('Incomplete region, next!')

    
    print('Store len:', len(store))
    return store

# Save each patches to its class folder
def save_to_png(patches_store, dst_path, patch_class):
    if patches_store == -1:
        print('No point in the', patch_class, 'class')
        return -1

    for patch in patches_store:
        uid = str(uuid.uuid4().hex)
        filename = uid[0::5] + '.png'
        save_as_path = path.join(dst_path,patch_class, filename)
        cv2.imwrite(save_as_path, patch)
        print(save_as_path)
    
    print('===== DONE =====')


# Main Working Space
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

for src_idx, source_name in enumerate(source_namelist):
    # Read source image from list name path
    img_name = source_name
    img = cv2.imread(path.join(source_dir, img_name), 0)
    print('===== Src_idx: {} ====='.format(src_idx))
    print('Source image name: ', img_name)

    # Read mask of source image
    mask_name = ''.join([img_name.replace('.tif', ''), '_mask.tif'])
    mask = cv2.imread(path.join(mask_dir, mask_name), 0)

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

    patchesStore = patchGenerator(tumor_points, dim=32, numOfpatch=32)
    save_to_png(patchesStore, dataset_dir, 'tumor-32')