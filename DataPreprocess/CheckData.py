'''
shape, roi shape
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from MeDIT.Normalize import Normalize01, NormalizeZ
from MeDIT.Visualization import Imshow3DArray
from MeDIT.ArrayProcess import ExtractBlock


def GetCenter(roi):
    roi = np.squeeze(roi)
    if np.ndim(roi) == 3:
        roi = roi[1]
    non_zero = np.nonzero(roi)
    center_x = int(np.median(np.unique(non_zero[1])))
    center_y = int(np.median(np.unique(non_zero[0])))
    return (center_x, center_y)


def CheckNPY():
    data_folder = r'/home/zhangyihong/Documents/BreastClassification/DCEPost/t1_post'
    max_row = []
    max_col = []
    for case in os.listdir(data_folder):
        if case == 'E054_65.npy':
            data_path = os.path.join(data_folder, case)
            data = np.load(data_path)
            cancer = np.load(os.path.join(r'/home/zhangyihong/Documents/BreastClassification/DCEPost/lesion_roi', case))
            cancer = np.clip(cancer, a_max=1, a_min=0)
            breast = np.load(os.path.join(r'/home/zhangyihong/Documents/BreastClassification/DCEPost/breast_mask', case))
            center = GetCenter(cancer)
            data_crop, _ = ExtractBlock(data, patch_size=[3, 80, 80], center_point=(-1, center[1], center[0]))
            cancer_crop, _ = ExtractBlock(cancer, patch_size=[3, 80, 80], center_point=(-1, center[1], center[0]))
            breast_crop, _ = ExtractBlock(breast, patch_size=[3, 80, 80], center_point=(-1, center[1], center[0]))
            plt.subplot(221)
            plt.imshow(data[0], cmap='gray')
            plt.subplot(222)
            plt.imshow(data[1], cmap='gray')
            plt.subplot(223)
            plt.imshow(data[2], cmap='gray')
            plt.show()

            # Imshow3DArray(Normalize01(data))

        # plt.figure(figsize=(6, 6))
        # plt.title('{}, {}'.format(np.max(np.sum(cancer_crop[1], axis=0)), np.max(np.sum(cancer_crop[1], axis=1))))
        # plt.imshow(data_crop[1], cmap='gray')
        # plt.contour(cancer_crop[1], colors='r')
        # plt.gca().set_axis_off()
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.savefig(os.path.join(r'/home/zhangyihong/Documents/BreastClassification/DCEPost/image', '{}.jpg').
        #             format(case[: case.index(('.npy'))]), bbox_inches='tight', pad_inches=0.0)
        # plt.close()
        # max_col.append(np.max(np.sum(cancer_crop[1], axis=(0))))
        # max_row.append(np.max(np.sum(cancer_crop[1], axis=(1))))
    # print(max(max_col), max(max_row))
# CheckNPY()


def Normalize():
    data_folder = r'/home/zhangyihong/Documents/BreastClassification/DCEPost/lesion_roi'
    new_folder = r'/home/zhangyihong/Documents/BreastClassification/DCEPost/lesion_roi_norm'
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    for case in os.listdir(data_folder):
        # print(case)
        data_path = os.path.join(data_folder, case)
        data = np.load(data_path)
        data = np.clip(data, a_min=0., a_max=1.)
        # data = NormalizeZ(data)
        if (np.unique(data) != np.array([0., 1.], dtype=np.float32)).all():
            print(case, np.unique(data))
        np.save(os.path.join(new_folder, case), data)
# Normalize()


def CropData():
    data_folder = r'/home/zhangyihong/Documents/BreastClassification/DCEPost/t1_post'
    new_folder = r'/home/zhangyihong/Documents/BreastClassification/DCEPost/t1_post_crop_norm'
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    for case in os.listdir(data_folder):
        data_path = os.path.join(data_folder, case)
        data = np.load(data_path)
        cancer = np.load(os.path.join(r'/home/zhangyihong/Documents/BreastClassification/DCEPost/lesion_roi_norm', case))
        center = GetCenter(cancer)
        data_crop, _ = ExtractBlock(data, patch_size=[3, 100, 100], center_point=(-1, center[1], center[0]))
        np.save(os.path.join(new_folder, case), NormalizeZ(data))
        # plt.imshow(data_crop[1], cmap='gray')
        # plt.show()
# CropData()

def CropROI():
    data_folder = r'/home/zhangyihong/Documents/BreastClassification/DCEPost/lesion_roi_norm'
    new_folder = r'/home/zhangyihong/Documents/BreastClassification/DCEPost/lesion_roi_crop_norm'
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    for case in os.listdir(data_folder):
        cancer = np.load(os.path.join(r'/home/zhangyihong/Documents/BreastClassification/DCEPost/lesion_roi_norm', case))
        center = GetCenter(cancer)
        cancer_crop, _ = ExtractBlock(cancer, patch_size=[3, 100, 100], center_point=(-1, center[1], center[0]))
        np.save(os.path.join(new_folder, case), cancer_crop)
CropROI()
