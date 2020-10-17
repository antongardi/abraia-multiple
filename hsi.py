import os
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image


cf = os.path.dirname(os.path.abspath(__file__))

# HDR IMEC 463.50, 469.22, 477.88, 490.49, 502.47, 513.95, 524.59, 539.51, 552.80, 555.12, 565.04, 578.94, 592.51, 601.11, 	# 623.50, 630.70
# Redondeadas 463, 469, 478, 490, 502, 514, 525, 540, 553, 555, 565, 579, 593, 601, 623, 631
BANDS_WLTH = np.array([463, 469, 478, 490, 502, 514, 525, 540, 553, 555, 565, 579, 593, 601, 623, 631])
CMF = np.array(np.loadtxt(os.path.join(cf, 'cie-cmf_1nm.txt'), usecols=(0, 1, 2, 3)))


def spec_to_xyz(hsi, bands=BANDS_WLTH):
    """Convert HSI cube in the visible espectrum to XYZ image (CIE1931)"""
    size = hsi.shape[:2]
    nbands = hsi.shape[2]
    Xcmf, Ycmf, Zcmf = [], [], []
    X, Y, Z = np.zeros(size), np.zeros(size), np.zeros(size)
    for i in range(nbands):
        point = np.where(CMF == BANDS_WLTH[i])
        band_cmf = np.array(CMF[point[0]])
        Xcmf.append(band_cmf[0][1])
        Ycmf.append(band_cmf[0][2])
        Zcmf.append(band_cmf[0][3])
        band = hsi[:, :, i]
        X = X + Xcmf[i] * band
        Y = Y + Ycmf[i] * band
        Z = Z + Zcmf[i] * band
    return np.dstack([X, Y, Z])


def xyz_to_sRGB(XYZ):
    """Convert XYZ (CIE1931) image to sRGB image"""
    X, Y, Z = XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2] 
    # https://en.wikipedia.org/wiki/SRGB
    r = 3.24096994 * X - 1.53738318 * Y - 0.49861076 * Z
    g = -0.96924364 * X + 1.8759675 * Y + 0.04155506 * Z
    b = 0.5563008 * X - 0.20397696 * Y + 1.05697151 * Z
    #from skimage.color import rgb2xyz, xyz2rgb
    #rgb = xyz2rgb(XYZ)
    rgb = np.dstack((r, g, b))
    addwhite = np.amin(rgb)
    r = r - addwhite
    g = g - addwhite; b = b - addwhite
    rgb = np.dstack([r, g, b])
    # Gamma function (https://en.wikipedia.org/wiki/SRGB )
    R = np.maximum((1.055 * np.power(r, 0.41667)) - 0.055, 12.92 * r)
    G = np.maximum((1.055 * np.power(g, 0.41667)) - 0.055, 12.92 * g)
    B = np.maximum((1.055 * np.power(b, 0.41667)) - 0.055, 12.92 * b)
    return np.dstack([R, G, B])


def spec_to_rgb(cube):
    return xyz_to_sRGB(spec_to_xyz(cube))


def decor_bands(hsi, n_components=3):
    """Calculate principal components of hsi cube"""
    h, w, d = hsi.shape
    X = hsi.reshape((h * w), d)
    pca = PCA(n_components=n_components)
    bands = pca.fit_transform(X).reshape(h, w, n_components)
    return bands, pca.components_


def resize(img, size):
    return np.array(Image.fromarray(img).resize(size, resample=Image.LANCZOS))


def normalize(img):
    min, max = np.amin(img), np.amax(img)
    return (img - min) / (max - min)


def saliency_map(band):
    h, w = band.shape
    myfft = np.fft.fft2(resize(band, (64, 64)))
    logAmplitude, phase = np.log(np.absolute(myfft)), np.angle(myfft)
    spectralResidual = logAmplitude - \
        nd.uniform_filter(logAmplitude, size=3, mode='nearest')
    saliencyMap = np.absolute(np.fft.ifft2(
        np.exp(spectralResidual + 1.j * phase)))
    saliencyMap = nd.gaussian_filter(saliencyMap, sigma=3)
    return normalize(resize(saliencyMap, (w, h)))


def saliency(hsi):
    """Calculate saliency map of HSI cube"""
    return np.sum(np.dstack([saliency_map(hsi[:, :, n]) for n in range(hsi.shape[2])]), axis=2)


def get_spectrum(hsi, point=None):
    """Get spectrum at a given point (x, y)
    
    When a point is not specified the spectrum of the most salient point is returned.
    """
    if point is None:
        sal = saliency(hsi)
        idx = np.unravel_index(np.argmax(sal), sal.shape)
        point = (idx[1], idx[0])
    return hsi[point[1], point[0], :]
