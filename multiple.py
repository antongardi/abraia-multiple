import os
import tempfile
import numpy as np

from PIL import Image
from abraia import Abraia
from scipy.io import loadmat
from spectral.io import envi


tempdir = tempfile.gettempdir()

class Multiple(Abraia):
    def __init__(self, folder=''):
        super(Multiple, self).__init__()

    def load_file(self, path):
        return self.download(path)

    def load_image(self, path):
        return np.asarray(Image.open(self.download(path)))

    def load_mat(self, path):
        mat = loadmat(self.download(path))
        for key, value in mat.items():
            if type(value) == np.ndarray:
                return value
        return mat

    def load_header(self, path):
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        if not os.path.exists(dest):
            self.download(path, dest)
        return envi.read_envi_header(dest)

    def load_envi(self, path):
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        if not os.path.exists(dest):
            self.download(path, dest)
        name, ext = basename.split('.')
        raw = os.path.join(tempdir, f"{name}.raw")
        if not os.path.exists(raw):
            self.download(f"{path.split('.')[0]}.raw", raw)
        return np.array(envi.open(dest, raw)[:, :, :])

    def load_mosaic(self, path, size=(4, 4)):
        r, c = size
        img = self.load_image(path)
        cube = np.dstack([img[(k % r)::r, (k // c)::c] for k in range(r * c)])
        return cube

    def load_metadata(self, path):
        if path.endswith('.hdr'):
            return self.load_header(path)
        return super(Multiple, self).load_metadata(self.userid + '/' + path)
