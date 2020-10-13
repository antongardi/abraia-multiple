import os
import numpy as np
from PIL import Image
from fnmatch import fnmatch
import spectral.io.envi as envi
from abraia import Abraia


class Multiple(Abraia):
    def __init__(self, folder=''):
        super(Multiple, self).__init__()

    def list_files(self, path):
        length = len(self.userid) + 1
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        folder = dirname + '/' if dirname else dirname
        files, folders = super(Multiple, self).list_files(path=self.userid + '/' + folder)
        files = list(map(lambda f: {'path': f['source'][length:], 'size': f['size'], 'date': f['date']}, files))
        if basename:
            files = list(filter(lambda f: fnmatch(f['path'], path), files))
        return files

    def load_file(self, path):
        return self.download_file(self.userid + '/' + path)

    def load_image(self, path):
        return np.asarray(Image.open(self.load_file(path)))

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

    def load_metadata(self, path):
        if path.endswith('.hdr'):
            return self.load_header(path)
        return super(Multiple, self).load_metadata(self.userid + '/' + path)
