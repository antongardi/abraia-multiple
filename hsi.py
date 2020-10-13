import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

cf = os.path.dirname(os.path.abspath(__file__))

# HDR IMEC 463.50, 469.22, 477.88, 490.49, 502.47, 513.95, 524.59, 539.51, 552.80, 555.12, 565.04, 578.94, 592.51, 601.11, 	# 623.50, 630.70
# Redondeadas 463, 469, 478, 490, 502, 514, 525, 540, 553, 555, 565, 579, 593, 601, 623, 631
BANDS_WLTH = np.array([463, 469, 478, 490, 502, 514, 525, 540, 553, 555, 565, 579, 593, 601, 623, 631])
#BANDS_WLTH = np.array([460, 470, 480, 490, 500, 510, 520, 535, 550, 560, 570, 580, 590, 600, 610, 620])
CMF = np.array(np.loadtxt(os.path.join(cf, 'cie-cmf_1nm.txt'), usecols=(0, 1, 2, 3)))


# Functions to process and analyse HSI pictures
def get_spectrum(hsi, x, y):
    """Get spectrum at a given point"""
    spectrum = [64 * hsi[band][x, y] for band in range(hsi.shape[2])]
    return spectrum


def get_salient_spectrum(hsi):
    """Get spectrum at most salient point"""
    saliency = saliency(hsi)
    point = np.where(saliency == np.amax(saliency))
    return get_spectrum(hsi, point[0], point[1])


def spec_to_xyz(hsi):
    """Convert HSI cube in the visible espectrum to XYZ image (CIE1931)"""
    nbands = hsi.shape[2]
    Xcmf=[]
    Ycmf=[] ; Zcmf=[]
    X = np.zeros([256, 512])
    Y = np.zeros([256, 512])
    Z = np.zeros([256, 512])
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
    X = XYZ[:, :, 0]
    Y = XYZ[:, :, 1] ; Z = XYZ[:, :, 2] 
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


def learn_decor_bands(sequence, maxcomponents=6):
    """Get spectral principal components over a sequence of hsi cubes"""
    nbands = sequence[0].shape[2]
    #Prepare empty feature vectors for every spectral bands
    vector1 = {}
    for band in range(nbands):
        vector1[band] = []
    #Extract features from data
    for current_frame in sequence:
        for band in range(nbands):
            bandVector = current_frame[:, :, band].flatten()
            vector1[band].extend(bandVector)
    #Arrange features in array for analysys
    vector = [vector1[band] for band in range(nbands)]
    #Learn PCA over the features from the whole sequence
    X = np.transpose(np.stack(vector))
    n_components = np.minimum(len(X), maxcomponents) 
    return PCA(n_components=n_components)


def decor_bands(hsi, maxcomponents=6):
    """Get principal components of hsi cube"""
    nbands = hsi.shape[2]
    #Extract features from bands and arrange in array
    vector = [hsi[:, :, band].flatten() for band in range(nbands)]
    #Learn PCA over the features from one HSI cube
    X = np.transpose(np.stack(vector))
    n_components = np.minimum(len(X), maxcomponents)
    pca = PCA(n_components=n_components, whiten=True)
    #Project bands to PC's and resize to original image dimensions
    pca_result = pca.fit_transform(X)
    pc_vector = np.transpose(pca_result)
    pc_images = {}
    for component in range(len(pc_vector)):
        pc_im = pc_vector[component]
        pc_im = pc_im.reshape(256, 512)
        pc_images[component] = pc_im
    return pc_images


def project_bands(model, cube):
    """Project bands to modelled PC's and resize to original image dimensions"""
    pca_result = model.fit_transform(X)
    pc_vector = np.transpose(pca_result)
    pc_images = {}
    for component in range(len(pc_vector)):
        pc_im = pc_vector[component]
        pc_im = pc_im.reshape(256, 512)
        pc_images[component] = pc_im
    return pc_images


def saliency_map(in_im):
    in_im = cv2.resize(in_im, dsize=(64, 64),
                        interpolation=cv2.INTER_LANCZOS4)
    myfft = np.fft.fft2(in_im)
    myLogAmplitude = np.log(np.absolute(myfft))
    myPhase = np.angle(myfft)
    smoothAmplitude = cv2.blur(myLogAmplitude, (3, 3))
    mySpectralResidual = myLogAmplitude - smoothAmplitude
    saliencyMap = np.absolute(np.fft.ifft2(
        np.exp(mySpectralResidual + 1.j*myPhase)))
    cv2.normalize(cv2.GaussianBlur(saliencyMap, (9, 9), 3, 3),
                    saliencyMap, 0., 1., cv2.NORM_MINMAX)
    return cv2.resize(saliencyMap, dsize=(512, 256), interpolation=cv2.INTER_LANCZOS4)


def saliency(hsi, maxcomponents=6):
    """Calculate saliency map of HSI cube"""
    #Nmaps = np.minimum(maxcomponents, len(hsi))
    pc_images = decor_bands(hsi, maxcomponents)
    size = np.shape(pc_images[0])
    sal = np.zeros((size[0], size[1]))
    Nmaps = len(pc_images)
    for component in range(Nmaps):
        sal = sal + saliency_map(pc_images[component])
    return sal


def saliency_chr(hsi, maxcomponents=6):
    """Calculate saliency map of HSI cube discarding intensity"""
    pc_images = decor_bands(hsi, maxcomponents)
    size = np.shape(pc_images[0])
    sal = np.zeros((size[0], size[1]))
    Nmaps = len(pc_images)
    for component in range(Nmaps - 1):
        sal = sal + saliency_map(pc_images[component + 1])
    return sal
