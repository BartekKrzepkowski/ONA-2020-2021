import sys
import pickle

import argparse
import numpy as np
from PIL import Image
from scipy import fftpack

parser = argparse.ArgumentParser()

parser.add_argument('filename', type=str, help='.dct or .bmp filename' )
parser.add_argument('--ratio', type=float, default=0.5, help='compression ratio')


def get_2D_dct(img):
    """ Get 2D Cosine Transform of Image
    """
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')


def get_2d_idct(coefficients):
    """ Get 2D Inverse Cosine Transform of Image
    """
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')


def save_img(img, img_name):
    im = Image.fromarray(img)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(img_name)


def bmp2dct(filename, ratio, eps=0.05):
    im = Image.open(f'{filename}.bmp', mode='r')
    img = np.array(im, dtype=float)
    a = 0
    b = img.shape[0]
    curr_ratio = 1.0
    coefs = get_2D_dct(img)
    coefs_p = np.zeros_like(coefs)
    while curr_ratio < ratio or curr_ratio > ratio + eps:
        del coefs_p
        coefs_p = np.zeros_like(coefs)
        i = (a + b) // 2
        coefs_p[:i, :i] = coefs[:i, :i]
        im_t = Image.fromarray(get_2d_idct(coefs_p))
        curr_ratio = sys.getsizeof(im_t) / sys.getsizeof(im)
        if curr_ratio > ratio:
            b = i
        else:
            a = i

    with open(f'{filename}.dct', 'wb') as f:
        data = (coefs_p[:i, :i], coefs_p.shape)
        pickle.dump(data, f)


def dct2bmp(filename):
    with open(f'{filename}.dct', 'rb') as f:
        coefs_p, img_shape = pickle.load(f)
    coefs = np.zeros(img_shape)
    if len(coefs.shape) < 3:
        coefs[:coefs_p.shape[0], :coefs_p.shape[0]] = coefs_p
        img = get_2d_idct(coefs_p)
        save_img(img, f'{filename}.bmp')
    else:
        for i in range(coefs.shape[0]):
            coefs[i, :coefs_p.shape[0], :coefs_p.shape[0]] = coefs_p[i]
            img = get_2d_idct(coefs_p)
            save_img(img, f'{filename}.bmp')


if __name__ == '__main__':
    args = parser.parse_args()
    names = args.filename.split('.')
    if names[-1] == 'bmp':
        bmp2dct(names[0], args.ratio)
    elif names[-1] == 'dct':
        dct2bmp(names[0])
    else:
        print('Wybrane błędne rozszerzenie. Wybierz rozszerzenie jeszcze raz.')