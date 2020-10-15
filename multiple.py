import os
import tempfile
import numpy as np

from PIL import Image
from abraia import Abraia
from fnmatch import fnmatch
from scipy.io import loadmat
from spectral.io import envi


tempdir = tempfile.gettempdir()

class Multiple(Abraia):
    def __init__(self, folder=''):
        super(Multiple, self).__init__()

    def list_files(self, path):
        # TODO: Move to abraia package
        length = len(self.userid) + 1
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        folder = dirname + '/' if dirname else dirname
        files, folders = super(Multiple, self).list_files(path=self.userid + '/' + folder)
        files = list(map(lambda f: {'path': f['source'][length:], 'size': f['size'], 'date': f['date']}, files))
        if basename:
            files = list(filter(lambda f: fnmatch(f['path'], path), files))
        return files

    def upload_file(self, src, path=''):
        # TODO: Move to abraia package
        if isinstance(src, str) and src.startswith('http'):
            return self.upload_remote(src, self.userid + '/')
        return super(Multiple, self).upload_file(src, self.userid + '/' + path)

    def download_file(self, path, dest=''):
        # TODO: Move to abraia package
        buffer = super(Multiple, self).download_file(self.userid + '/' + path)
        if dest:
            with open(dest, 'wb') as f:
                f.write(buffer.getbuffer())
        return buffer

    def load_file(self, path):
        return self.download_file(path)

    def load_image(self, path):
        return np.asarray(Image.open(self.load_file(path)))

    def load_mat(self, path):
        mat = loadmat(self.load_file(path))
        for key, value in mat.items():
            if type(value) == np.ndarray:
                return value
        return mat

    def load_header(self, path):
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        if not os.path.exists(dest):
            self.download_file(path, dest)
        return envi.read_envi_header(dest)

    def load_envi(self, path):
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        if not os.path.exists(dest):
            self.download_file(path, dest)
        name, ext = basename.split('.')
        raw = os.path.join(tempdir, f"{name}.raw")
        if not os.path.exists(raw):
            self.download_file(f"{path.split('.')[0]}.raw", raw)
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
