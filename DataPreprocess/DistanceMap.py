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

base_rate = 0.05
data = np.load(r'V:\yhzhang\BreastNpy\Roi\A001_CHEN HAN YING.npy')
resolution = (1.0, 1.0, 1.5)
slice_rate = resolution[2] / resolution[0] * base_rate

result = InterSliceFilter(roi, slice_rate)
result = IntraSliceFilter(result, base_rate)
new_data = IntraSliceFilter(data)
new_data = InterSliceFilter(new_data)

flatten_data = FlattenImages(np.transpose(data, axes=(2, 0, 1)))
flatten_roi = FlattenImages(np.transpose(new_data, axes=(2, 0, 1)))

plt.subplot(121)
plt.imshow(flatten_data, cmap='gray')
plt.subplot(122)
plt.contour(flatten_roi, colors='r')
plt.axis('off')
plt.show()
