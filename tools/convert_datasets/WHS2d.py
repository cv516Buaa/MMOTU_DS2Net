import argparse
import os
import time
import sys
import shutil
import numpy as np
import cv2
from collections import *
## added by LYU: 2023/07/14
import nibabel as nib
import matplotlib.pyplot as plt

CLASSES = ('LV', 'Myo', 'RV')
PALETTE = [[64, 0, 0], [0, 64, 0], [0, 0, 64]]

def global_linear_transform(img):
    maxV = img.max()
    minV = img.min()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = ((img[i,j] - minV)*255) / (maxV-minV)
            #img[i,j] = ((img[i,j] - minV)) / (maxV-minV)
    return img

def norm(img):
    meanV = np.mean(img)
    stdV = np.std(img)
    return (img-meanV) / stdV

## added by LYU 2023.7.14
# Use once and create train/val list
def listfromfolder(args):
    inp_folder = args.inp_folder
    inp_folder = os.path.join(inp_folder, 'images')
    target_url = args.target_url
    trainlist_url = os.path.join(target_url, 'train.txt')
    vallist_url = os.path.join(target_url, 'val.txt')
    print(trainlist_url)
    print(vallist_url)
    
    ## walk folder and create train.txt and val.txt
    for root, dirs, files in os.walk(inp_folder):
        files.sort()
        train_files = files[:len(files)-args.val_num]
        val_files = files[len(files)-args.val_num : len(files)]
        assert len(list(set(train_files).intersection(set(val_files)))) == 0
        with open(trainlist_url, 'w') as f:        
            for imgname in train_files:
                f.write(imgname[:-7] + '\n')
        f.close()
        with open(vallist_url, 'w') as f:        
            for imgname in val_files:
                f.write(imgname[:-7] + '\n')
        f.close()

## added by LYU 2023.7.14
def nii2img(args):
    inp_folder = args.inp_folder
    inp_folder = os.path.join(inp_folder, 'images')
    for root, dirs, files in os.walk(inp_folder):
        for i in range(len(files)):
            image_url_tmp = os.path.join(inp_folder, files[i])
            try:
                nii_image_tmp = nib.load(image_url_tmp)
            except:
                print('EXCEPT: ', image_url_tmp)
            else:
                nii_image_tmp_data = nii_image_tmp.get_fdata()[0, :, :]
                ## global linear transformation
                nii_image_tmp_data = norm(nii_image_tmp_data)
                nii_image_tmp_data = global_linear_transform(nii_image_tmp_data)
                save_dir_tmp = os.path.join(inp_folder, files[i][:-7] + '.PNG')
                print(save_dir_tmp)
                cv2.imwrite(save_dir_tmp, nii_image_tmp_data)
            
## added by LYU 2023.7.14
# 3-class segmentation
# Use once and create indexmap
def nii2indexmap(args):
    inp_folder = args.inp_folder
    inp_folder = os.path.join(inp_folder, 'annotations')
    for root, dirs, files in os.walk(inp_folder):
        for i in range(len(files)):
            anno_url_tmp = os.path.join(inp_folder, files[i])
            try:
                nii_anno_tmp = nib.load(anno_url_tmp)
            except:
                print('EXCEPT: ', anno_url_tmp)
            else:
                nii_anno_tmp_data = nii_anno_tmp.get_fdata()[0, :, :]
                ## LV:500; Myo:205; RV:600
                nii_anno_tmp_data[nii_anno_tmp_data == 500] = 1
                nii_anno_tmp_data[nii_anno_tmp_data == 205] = 2
                nii_anno_tmp_data[nii_anno_tmp_data == 600] = 3
                nii_anno_tmp_data[nii_anno_tmp_data > 3] = 0
                save_dir_tmp = os.path.join(inp_folder, 'img' + files[i][3:-7] + '_index.PNG')
                cv2.imwrite(save_dir_tmp, nii_anno_tmp_data)
                print(save_dir_tmp)
                '''
                a = cv2.imread(save_dir_tmp)
                for i in range(a.shape[0]):
                   for j in range(a.shape[1]):
                       if a[i, j, 0] == 1:
                           a[i, j, :] = PALETTE[0]
                       if a[i, j, 0] == 2:
                           a[i, j, :] = PALETTE[1]
                       if a[i, j, 0] == 3:
                           a[i, j, :] = PALETTE[2]
                a = cv2.imshow("a", a)
                print(anno_url_tmp)
                cv2.waitKey(10000)
                '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate img list from folder')
    parser.add_argument('--inp_folder', default='', type=str, help='original image folder')
    parser.add_argument('--target_url', default='', type=str, help='the path to .txt file')
    parser.add_argument('--val_num', default=64, type=int, help='number of val images')
    args = parser.parse_args()
    nii2img(args)
    #listfromfolder(args)
    #nii2indexmap(args)
