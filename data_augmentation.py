from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from scipy import ndimage, misc
import imageio
from skimage import data
import matplotlib.pyplot as plt
import six.moves as sm
import re
import time
import os
from collections import defaultdict
import PIL.Image
import cv2
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

np.random.seed(44)
ia.seed(44)


def main():
    """
    files = os.listdir("frames_goRight")
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for filename in files:
        draw_single_sequential_images(filename, "frames_goRight")

    files = os.listdir("frames_goLeft")
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for filename in files:
        draw_single_sequential_images(filename, "frames_goLeft")

    files = os.listdir("frames_takeOff")
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for filename in files:
        draw_single_sequential_images(filename, "frames_takeOff")
    files = os.listdir("frames_default")

    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for filename in files:
        draw_single_sequential_images(filename, "frames_default")
    """

    files = os.listdir("session_1_default")
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for filename in files:
        draw_single_sequential_images(filename, "session_1_default")

    files = os.listdir("session_1_takeOff")
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for filename in files:
        draw_single_sequential_images(filename, "session_1_takeOff")

def draw_single_sequential_images(filename, path):

    image = misc.imresize(ndimage.imread(path + "/" + filename),(224,224))

    def sometimes(aug): return iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            #iaa.Fliplr(0.5),   horizontally flip 50% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                # scale images to 80-120% of their size, individually per axis
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # translate by -10 to +10 percent (per axis)
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-5, 5),
                shear=(-5, 5),  # shear by -5 to +5 degrees
                # use nearest neighbour or bilinear interpolation (fast)
                order=[0, 1],
                # if mode is constant, use a cval between 0 and 255
                cval=(0, 255),
                # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                mode=ia.ALL
            )),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.Invert(0.05, per_channel=False),  # invert color channels
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                iaa.OneOf([
                    # blur images with a sigma between 0 and 2.0
                    iaa.GaussianBlur((0, 2.0)),
                    # blur image using local means with kernel sizes between 2 and 5
                    iaa.AverageBlur(k=(2, 5)),
                    # blur image using local medians with kernel sizes between 3 and 5
                    iaa.MedianBlur(k=(3, 5)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(
                    0.75, 1.5)),  # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                # add gaussian noise to images
                iaa.AdditiveGaussianNoise(loc=0, scale=(
                    0.0, 0.01*255), per_channel=0.5),
                # change brightness of images (by -10 to 10 of original value)
                iaa.Add((-10, 10), per_channel=0.5),
                # change hue and saturation
                iaa.AddToHueAndSaturation((-20, 20)),
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.9, 1.1), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-2, 0),
                        first=iaa.Multiply((0.9, 1.1), per_channel=True),
                        second=iaa.ContrastNormalization((0.9, 1.1))
                    )
                ]),
                # improve or worsen the contrast
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            ],
                random_order=True
            )
        ],
        random_order=True
    )

    im = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    for c in range(0, 4):
        im[c] = image

    grid = seq.augment_images(im)
    for im in range(len(grid)):
        filename_without_extension = os.path.splitext(filename)[0]
        misc.imsave(path + "/" + filename_without_extension +
                    "_" + str(im) + ".jpg", grid[im])


if __name__ == "__main__":
    main()

   
