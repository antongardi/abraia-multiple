# Abraia API extension for HSI processing and analysis
from abraia import Abraia

import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi


class Multiple(Abraia):
    def __init__(self, folder=''):
        super(Multiple, self).__init__()

    def load_image(self, path):
        f = self.from_store(path).to_buffer()
        img = plt.imread(f, format=path[-3:])
        return np.array(img)

    def load_envi(self, path):
        name, ext = path.split('.')
        cube = np.array(envi.open(path, f"{name}.raw")[:, :, :])
        return cube

    def load_mosaic(self, path, size=(4, 4)):
        r, c = size
        img = self.load_image(path)
        cube = np.dstack([img[(k % r)::r, (k // c)::c] for k in range(r * c)])
        return cube

    def open_envi_images(self, remote_folder):
        """Functions to demosaic HSI data and build spectral datacubes"""
        sequence = []
        files = self.list(remote_folder)[0]
        for file in files:
            self.from_store(file['path']).to_file(file['name'])
        for k, file in enumerate(filter(lambda f: f['name'].endswith('hdr'), files)):
            cube = self.load_envi(file['name'])
            sequence.append(cube / np.amax(cube))
        print(f"{k + 1} ENVI images loaded from {remote_folder}")
        return sequence

    def demosaic_sequence(self, folder, Nframes=1, Startframe = 0):
        """Demosaic sequence in a folder"""
        sequence = []
        files = self.list(folder)[0]
        print(f"Mappig 4x4 HSI mosaics to specific bands")
        for frame, image in enumerate(files[Startframe:Startframe+Nframes]):
            cube = self.load_mosaic(image['path'])
            sequence.append(cube)
            print(f"Frame no{len(spectralSequence)} with {spectralCube.shape[2]} bands")
        print(f"Number of HSI frames: {len(spectralSequence)}")
        return sequence
