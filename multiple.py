# Abraia API extension for HSI processing and analysis
from abraia import Abraia

import os
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
import spectral.io.envi as envi


class Multiple(Abraia):
    def __init__(self, folder=''):
        super(Multiple, self).__init__()

    def load_image(self, path):
        f = self.from_store(path).to_buffer()
        img = np.asarray(Image.open(f))
        # img = plt.imread(f, format=path[-3:])
        return img

    def load_header(self, path):
        header = os.path.basename(path)
        if not os.path.exists(header):
            self.from_store(path).to_file(header)
        header = envi.open(header)
        return header.metadata

    def load_envi(self, path):
        header = os.path.basename(path)
        if not os.path.exists(header):
            self.from_store(path).to_file(header)
        name, ext = header.split('.')
        raw = f"{name}.raw"
        if not os.path.exists(raw):
            self.from_store(f"{path.split('.')[0]}.raw").to_file(raw)
        cube = np.array(envi.open(header, raw)[:, :, :])
        return cube

    def load_mosaic(self, path, size=(4, 4)):
        r, c = size
        img = self.load_image(path)
        cube = np.dstack([img[(k % r)::r, (k // c)::c] for k in range(r * c)])
        return cube

    def load_meta(self, path):
        if path.endswith('.hdr'):
            return self.load_header(path)
        return self.load_metadata(self.userid + '/' + path)
