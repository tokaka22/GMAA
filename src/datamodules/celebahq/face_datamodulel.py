from typing import Any, Dict, Optional, Tuple
from numpy import float32
import numpy as np

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

import random
import os
from PIL import Image


class FaceDataset(Dataset):

    def __init__(self, facedata, transforms, type='train', au_mode='fixed'):
        super().__init__()

        # type = 'train' 
        self.type = type
        assert self.type in ['train', 'val', 'test']

        self.au_mode = au_mode
        self.transforms = transforms
        self.facedata_list = facedata.data[type]

    def __len__(self):
        return len(self.facedata_list)
    
    def get_random_au(self, img0_path):

        random_au_fg = True
        while random_au_fg:
            trg_fn_idx = random.randint(0, len(self.facedata_list)-1)
            trg_data = self.facedata_list[trg_fn_idx]
            if (os.path.basename(trg_data[3]) != os.path.basename(img0_path)):
                random_au_fg = False
            trg_random_au = trg_data[2]
        
        return trg_random_au

    def __getitem__(self, index):
        img0_path = self.facedata_list[index][4]
        img0 = Image.open(img0_path)

        au0 = self.facedata_list[index][2]
        au1 = self.get_random_au(img0_path)

        lm0 = self.facedata_list[index][0]
        dt0 = self.facedata_list[index][1]

        return self.transforms(img0), torch.FloatTensor(au0), torch.FloatTensor(au1), lm0, dt0, int(os.path.splitext(os.path.basename(img0_path))[0].split('_')[-1])


class FaceData():

    def __init__(self, data_use, data_split, test_smallset, data_dir, data, au_c_dim):
        super().__init__()

        self.au_c_dim = au_c_dim
        self.data_dir = data_dir

        for k in data_use: # eg: 'CelebA-HQ-ALL'
            img_list = []

            image_path = os.path.join(data_dir, data[k]['path'], data[k]['image_path'])
            attr_path = os.path.join(data_dir, data[k]['path'], data[k]['attr_path'])
            lm_path = os.path.join(data_dir, data[k]['path'], data[k]['lm_path'])
            dt_path = os.path.join(data_dir, data[k]['path'], data[k]['dt_path'])
            id_path = os.path.join(data_dir, data[k]['path'], data[k]['id_path'])
            map_path = os.path.join(data_dir, data[k]['path'], data[k]['map_path'])

            # id
            id_dict = {} # {filename:str : id:str}

            with open(id_path,"r") as f:
                line = 'first line'
                while line:            
                    # print(line)
                    line = f.readline() 
                    line = line[:-1] 
                    if line not in ['first line', '']: 
                        filename, id = line.split(' ')
                        fileidx = filename.split('.')[0]
                        id_dict[fileidx] = id

            # lm
            lm_lines = []
            with open(lm_path,"r") as f:
                lm_attr = f.readline() 
                line = 'first line'
                while line:            
                    # print(line)
                    line = f.readline() 
                    line = line.rstrip() 
                    if line not in ['']:
                        lm_lines.append(line)

            # dt
            dt_lines = []
            with open(dt_path,"r") as f:
                dt_attr = f.readline() 
                line = 'first line'
                while line:            
                    # print(line)
                    line = f.readline() 
                    line = line.rstrip() 
                    if line not in ['']:
                        dt_lines.append(line)

            assert len(lm_lines) == 27255

            au_dict = {} # {filename:str : au:list}
            sameid_ct = 0
            samename_ct = 0
            line_ct = 0
            with open(attr_path,"r") as f:
                f.readline() 
                line = 'first line'
                while line:                             
                    line = f.readline() 
                    line = line.rstrip() 
                    if line not in ['first line', '']: 
                        split = line.split()  
                        filename = split[0].split('_')[-1]
                        fileidx = filename.split('.')[0]
                        values = split[1:] 

                        au = []  # Vector representing the presence of each attribute in each image
                        for n in range(self.au_c_dim):   
                            au.append(float(values[n])/5.)

                        au_dict[fileidx] = au
                        filepath = os.path.join(image_path, 'frame_det_00_' + filename)

                        # lm
                        lm_line = lm_lines[line_ct]
                        lm_split = lm_line.split()   
                        lm_filename = lm_split[0].split('_')[-1]    

                        # dt
                        dt_line = dt_lines[line_ct]
                        dt_split = dt_line.split()   
                        dt_filename = dt_split[0].split('_')[-1]   

                        assert lm_filename == filename 
                        assert dt_filename == filename 

                        # lm
                        lm = [int(v) for v in lm_split[1:]]

                        x = np.array(lm[::2])
                        y = np.array(lm[1::2])
                        
                        axises = np.stack([x, y], axis=-1)

                        eye_axis = axises[51]
                        nose_axis = axises[54]
                        mouth_axis = axises[98]

                        lm_axis = np.stack([eye_axis, nose_axis, mouth_axis])

                        # dt
                        crop_start_x = int(dt_split[1])
                        crop_start_y = int(dt_split[2])
                        crop_end_x = int(dt_split[3])
                        crop_end_y = int(dt_split[4])

                        dt_axis = np.array([crop_start_x, crop_start_y, crop_end_x, crop_end_y])

                        line_ct += 1
                        img_list += [ [lm_axis, dt_axis, au, id_dict[fileidx], filepath] ]
        
        facedata = {}

        if data_split == 'train_val': 
            if test_smallset:
                facedata.update({"train": img_list[:100]})
                facedata.update({"val": img_list[-100:]})
            else:
                facedata.update({"train": img_list[:-2725]})
                facedata.update({"val": img_list[-2725:]})
        elif data_split == 'train_test': 
            if test_smallset:
                facedata.update({"train": img_list[:100]})
                facedata.update({"test": img_list[-100:]})
            else:
                facedata.update({"train": img_list[:-2725]})
                facedata.update({"test": img_list[-2725:]})

        self.data = facedata

class FaceDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        image_size,
        au_c_dim,
        au_mode,
        data_split,
        test_smallset,
        data_dir: str = "data/",
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_use: list = [],
        data: dict = {},
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_use = data_use
        self.data_split = data_split

        self.facedata = FaceData(data_use, data_split, test_smallset, data_dir, data, au_c_dim)

        # data transformations
        self.transforms = []
        resize_fg = False
        for data_name in data_use:
            if 'HQ' in data_name:
                resize_fg = True
        if resize_fg:
            self.transforms.append(transforms.Resize(image_size))
        self.transforms.append(transforms.ToTensor())
        self.transforms.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transforms = transforms.Compose(self.transforms)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            if not self.data_train:
                self.data_train = FaceDataset(self.facedata, self.transforms, 'train', self.hparams.au_mode)
                print('Trainset: size: %s' % (len(self.data_train)))
            if self.data_split == 'train_val':
                if not self.data_val: 
                    self.data_val = FaceDataset(self.facedata, self.transforms, 'val', self.hparams.au_mode)
                    print('Val: size: %s' % (len(self.data_val)))
        elif stage == "test":
            if not self.data_test:
                self.data_test = FaceDataset(self.facedata, self.transforms, 'test', self.hparams.au_mode)
                print('Test: size: %s' % (len(self.data_test)))
        else:
            raise RuntimeError("stage isn't 'fit' or 'test'!")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "face_facex_hqv2_val.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
