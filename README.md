![Analytics](https://ga-beacon.appspot.com/UA-108018608-1/github/multiple?pixel)

# Multiple

### Abraia API extension for HyperSpectral Image (HSI) analysis.

The MULTIPLE Python package by ABRAIA provides a set of tools for seamless image processing and analysis. It supports a broad range of image types, including multispectral and hyperspectral images.

The package integrates state-of-the-art image manipulation libraries with ABRAIA's API. As a result, it enables a seamless use of Python notebooks to process image data and metadata stored in the cloud.

A clear benefit of using Python is the immediate availability of the quickly growing ecosystem of machine learning and image analysis resources available in this language. Also, the ability to work on notebooks that can be accessed through a common web browser running in any device -be mobile or desktop-, allowing an easy, fast, and collaborative prototyping of solutions.

* Getting started [![Getting started](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/multiple/blob/main/notebooks/getting-started.ipynb)

![classification](https://store.abraia.me/multiple/notebooks/classification.jpg)

* Classification [![Classification](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/multiple/blob/main/notebooks/classification.ipynb)

## Installation

```sh
python -m pip install git+git://github.com/abraia/abraia-multiple.git
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
Next, we revise the main steps to get started with 
### Settings
Once we have installed the package, we need to [get an API Key](https://abraia.me/console/settings) if we don't have one yet. Once we have it, we set the environment,
    
`import os
from dotenv import load_dotenv
load_dotenv()
abraia_key = ''  #@param {type: "string"}
%env ABRAIA_KEY=$abraia_key`

### Get some sample data 
To start with, we may [upload some data](https://abraia.me/console/gallery) or we may just get some sample data publicly available,

`if not os.path.exists('PaviaU.mat') or not os.path.exists('PaviaU_gt.mat'):
    !wget http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat`


And then upload the data to the cloud
`multiple.upload('PaviaU.mat')`

### Read data from cloud
Now, we can read data from our cloud and load a HSI cube in the variable img
`img = multiple.load_image('PaviaU.mat')`

### Basic HSI visualization
As well, we can get some random bands from our HSI cube,
`imgs, indexes = hsi.random(img)`

and we can visualize these bands as like any other image array. For example,
```fig, ax = plt.subplots(2, 3)
ax = ax.reshape(-1)
for i, im in enumerate(imgs):
    ax[i].imshow(im, cmap='jet')
    ax[i].axis('off')
```
    
### Extraction of principal components and pseudocolor visualization

A common operation with spectral images is to reduce the dimensionality, applying principal components analysis. We can get the first three principal components into a three bands pseudoimage,

`pc_img = hsi.principal_components(img)`

and we may visualize this pseudoimage
 
```plt.title('Principal components')
plt.imshow(pc_img)
plt.axis('off')
```


## License

This software is licensed under the MIT License. [View the license](LICENSE).
