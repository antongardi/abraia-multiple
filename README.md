![Analytics](https://ga-beacon.appspot.com/UA-108018608-1/github/multiple?pixel)

# Multiple

Abraia API extension for HyperSpectral Image (HSI) analysis.

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

## License

This software is licensed under the MIT License. [View the license](LICENSE).