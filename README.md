![Analytics](https://ga-beacon.appspot.com/UA-108018608-1/github/multiple?pixel)

# Multiple - HyperSpectral Image (HSI) analysis

The MULTIPLE Python package by ABRAIA provides a seamless integration of multiple image processing tools for multispectral and hyperspectral image processing and analysis. The package integrates state-of-the-art image manipulation libraries and the ABRAIA API service to provide scalable cloud storage and management for multispectral image data and metdata. Also, it provides working notebooks ready to be used through a common web browser running in any device -be mobile or desktop-, allowing an easy, fast, and collaborative prototyping of solutions.

* Getting started [![Getting started](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/multiple/blob/main/notebooks/getting-started.ipynb)

* Classification [![Classification](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/multiple/blob/main/notebooks/classification.ipynb)

![classification](https://store.abraia.me/multiple/notebooks/classification.jpg)

## Installation

```sh
python -m pip install git+git://github.com/abraia/abraia-multiple.git
```

## Configuration

Installed the package, you have to configure your [ABRAIA KEY](https://abraia.me/console/settings) as environment variable:

```sh
export ABRAIA_KEY=api_key
```

On Windows you need to use `set` instead of `export`:

```sh
set ABRAIA_KEY=api_key
```

## Usage

```python
from multiple import Multiple

multiple = Multiple()
img = multiple.load_image('test.hdr')
meta = multiple.load_meta('test.hdr')
multiple.save_image('test.hdr', img, metadata=meta)
```

## Getting started

### Upload data

To start with, we may [upload some data](https://abraia.me/console/gallery) directly using the graphical interface, or using the multiple api:

```python
multiple.upload('PaviaU.mat')
```

### Load HSI image data

Now, we can load the hyperspectral image data (HSI cube) directly from the cloud:

```python
img = multiple.load_image('PaviaU.mat')
```

### Basic HSI visualization

Hyperspectral images cannot be directly visualized, so we can get some random bands from our HSI cube,

```python
imgs, indexes = hsi.random(img)
```

and visualize these bands as like any other monochannel image. For example,

```python
fig, ax = plt.subplots(2, 3)
ax = ax.reshape(-1)
for i, im in enumerate(imgs):
    ax[i].imshow(im, cmap='jet')
    ax[i].axis('off')
```

### Pseudocolor visualization

A common operation with spectral images is to reduce the dimensionality, applying principal components analysis (PCA). We can get the first three principal components into a three bands pseudoimage,

```python
pc_img = hsi.principal_components(img)
```

and visualize this pseudoimage,

```python
plt.title('Principal components')
plt.imshow(pc_img)
plt.axis('off')
```

## License

This software is licensed under the MIT License. [View the license](LICENSE).
