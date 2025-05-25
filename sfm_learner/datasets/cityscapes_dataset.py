
from collections import defaultdict
import os

from torch.utils.data import Dataset
import numpy as np
from sfm_learner.utils.image import load_image

########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def load_calibration(fname):
    return np.loadtxt(fname, delimiter=',').reshape(3,3)

def read_files(directory, ext=('.png', '.jpg', '.jpeg'), skip_empty=True):
    files = defaultdict(list)
    for entry in os.scandir(directory):
        relpath = os.path.relpath(entry.path, directory)
        if entry.is_dir():
            d_files = read_files(entry.path, ext=ext, skip_empty=skip_empty)
            if skip_empty and not len(d_files):
                continue
            files[relpath] = d_files[entry.path]
        elif entry.is_file():
            if ext is None or entry.path.lower().endswith(tuple(ext)):
                files[directory].append(relpath)
    return files

########################################################################################################################
#### DATASET
########################################################################################################################

class CityScapesDataset(Dataset):
    def __init__(self, root_dir, split, data_transform=None,
                 forward_context=0, back_context=0, strides=(1,),
                 depth_type=None, **kwargs):
        super().__init__()
        # Asserts
        assert depth_type is None or depth_type == '', \
            'ImageDataset currently does not support depth types'
        assert len(strides) == 1 and strides[0] == 1, \
            'ImageDataset currently only supports stride of 1.'

        self.root_dir = root_dir
        self.split = split

        self.backward_context = back_context
        self.forward_context = forward_context
        self.has_context = self.backward_context + self.forward_context > 0
        self.strides = 1

        self.files = []
        file_tree = read_files(root_dir)
        for k, v in file_tree.items():
            file_set = set(file_tree[k])
            files = [fname for fname in sorted(v) if self._has_context(fname, file_set)]
            self.files.extend([[k, fname] for fname in files])

        self.data_transform = data_transform

    def __len__(self):
        return len(self.files)

    def _has_context(self, filename, file_set):
        return True

    def _read_rgb_file(self, session, filename):
        return load_image(os.path.join(self.root_dir, session, filename))

    def __getitem__(self, idx):
        session, filename = self.files[idx]
        cam_name = filename.split('.')[0]+'_cam.txt'
        cam_file_name = os.path.join(self.root_dir, session, cam_name)
        images = self._read_rgb_file(session, filename)
        w, h = images.size
        stride = w//3
        image = images.crop((stride, 0, stride*2, h))
        image_pre = images.crop((0, 0, stride, h))
        image_after = images.crop((stride*2, 0, w, h))

        sample = {
            'idx': idx,
            'filename': '%s_%s' % (session, os.path.splitext(filename)[0]),
            #
            'rgb': image,
            'intrinsics': load_calibration(cam_file_name)
        }

        if self.has_context:
            sample['rgb_context'] = [image_pre, image_after]
                #self._read_rgb_context_files(session, filename)

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample

########################################################################################################################