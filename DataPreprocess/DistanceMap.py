import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter, maximum_filter
from MeDIT.Visualization import FlattenImages


def InterSliceFilter(attention, diff_value, kernel_size=(3, 1, 1)):
    raw_attention = deepcopy(attention)
    # w = np.array([[[0.25]], [[0.5]], [[0.25]]])

    while True:
        new_attention = maximum_filter(raw_attention, size=kernel_size)
        new_attention[new_attention > raw_attention] -= diff_value
        new_attention[new_attention < 0] = 0

        if not (new_attention > raw_attention).any():
            break

        raw_attention = new_attention

    # new_attention = convolve(new_attention, weights=w)
    # new_attention = convolve(new_attention, size=kernel_size)
    new_attention = median_filter(new_attention, size=kernel_size)
    new_attention = median_filter(new_attention, size=kernel_size)

    return new_attention


def IntraSliceFilter(attention, diff_value, kernel_size=(1, 3, 3)):
    raw_attention = deepcopy(attention)

    while True:
        new_attention = maximum_filter(raw_attention, size=kernel_size)
        new_attention[new_attention > raw_attention] -= diff_value
        new_attention[new_attention < 0] = 0

        if not (new_attention > raw_attention).any():
            break

        raw_attention = new_attention

    new_attention = median_filter(new_attention, size=kernel_size)
    new_attention = median_filter(new_attention, size=kernel_size)

    return new_attention

base_rate = 0.1
resolution = (1.5, 1.0, 1.0)
slice_rate = resolution[0] / resolution[2] * base_rate
for case in os.listdir(r'V:\yhzhang\BreastNpy\Roi'):
    try:
        data = np.load(os.path.join(r'V:\yhzhang\BreastNpy\Roi', case))
        new_data = InterSliceFilter(data, slice_rate)
        new_data = IntraSliceFilter(new_data, base_rate)
        # np.save(os.path.join(r'V:\yhzhang\BreastNpy\RoiDilated', case), new_data)
        flatten_data = FlattenImages(data)
        flatten_roi = FlattenImages(new_data)
        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.axis('off')
        plt.imshow(flatten_data, cmap='gray')
        plt.subplot(122)
        plt.axis('off')
        plt.imshow(flatten_roi, cmap='gray')
        plt.show()
        # plt.savefig(os.path.join(r'V:\yhzhang\BreastNpy\ImageDilated', case.split('.npy')[0]))
        # plt.close()
    except Exception as e:
        print(e)
        print(case)



# for case in os.listdir()