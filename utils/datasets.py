import json
import os
import random
from PIL import Image

# Augmentation library
import albumentations as albu
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

from .sparsevit_transforms import get_albu_transforms



def pil_loader(path: str) -> Image.Image:
    """PIL image loader

    Args:
        path (str): image path

    Returns:
        Image.Image: PIL image (after np.array(x) becomes [0,255] int8)
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """denormalize image with mean and std
    """
    image = image.clone().detach().cpu()
    image = image * torch.tensor(std).view(3, 1, 1)
    image = image + torch.tensor(mean).view(3, 1, 1)
    return image

class base_dataset(Dataset):
    def _init_dataset_path(self, path):
        tp_path = None # Tampered image
        gt_path = None # Ground truth
        return tp_path, gt_path
        
    def __init__(self, path, output_size = 512 ,transform = None, edge_width = None, if_return_name = False, if_return_shape = False, if_return_type = False) -> None:
        super().__init__()
        self.tp_path, self.gt_path = self._init_dataset_path(path)
        if self.tp_path == None:
            raise NotImplementedError
        self.transform = transform
        self.resize_transform =  get_albu_transforms(type_='resize', outputsize=output_size)
        self.if_return_name = if_return_name
        self.if_return_shape = if_return_shape
        self.if_return_type = if_return_type
    def __getitem__(self, index):
        output_list = []
        tp_path = self.tp_path[index]
        gt_path = self.gt_path[index]
        
        tp_img = pil_loader(tp_path)
        tp_shape = tp_img.size
        # if "negative" then gt is a image with all 0
        if gt_path != "Negative":
            gt_img = pil_loader(gt_path)
            gt_shape = gt_img.size
        else:
            temp = np.array(tp_img)
            gt_img = np.zeros((temp.shape[0], temp.shape[1], 3))
            gt_shape = (temp.shape[1], temp.shape[0])
  
        assert tp_shape == gt_shape, "tp and gt image shape must be the same, but got {} and {}".format(tp_shape, gt_shape)
        
        tp_img = np.array(tp_img) # H W C
        gt_img = np.array(gt_img) # H W C
        
        # Do augmentations
        if self.transform != None:
            res_dict = self.transform(image = tp_img, mask = gt_img)
            tp_img = res_dict['image']
            gt_img = res_dict['mask']
        
        
        gt_img =  (np.mean(gt_img, axis = 2, keepdims = True)  > 127.5 ) * 1.0 # fuse the 3 channels to 1 channel, and make it binary(0 or 1)
        gt_img =  gt_img.transpose(2,0,1)[0] # H W C -> C H W -> H W
        masks_list = [gt_img]
        # Do resize
        res_dict = self.resize_transform(image = tp_img, masks = masks_list)
        
        tp_img = res_dict['image']
        gt_img = res_dict['masks'][0].unsqueeze(0) # H W -> 1 H W        
        output_list.append(tp_img)
        output_list.append(gt_img)
        
        if self.if_return_name:
            basenae = os.path.basename(tp_path)
            output_list.append(basenae)
        
        if self.if_return_shape:
            tp_shape = (tp_shape[1], tp_shape[0]) # swap for correct order
            tp_shape = torch.tensor(tp_shape)
            output_list.append(tp_shape)
            
        if self.if_return_type:
            gt_type = True if torch.max(gt_img) != 0 else False
            output_list.append(gt_type)
    
        return output_list
    def __len__(self):
        return len(self.tp_path)
    
class mani_dataset(base_dataset):
    def _init_dataset_path(self, path):
        path = path
        tp_dir = os.path.join(path, 'Tp')
        gt_dir = os.path.join(path, 'Gt')
        tp_list = os.listdir(tp_dir)
        gt_list = os.listdir(gt_dir)
        # Use sort mathod to keep order, to make sure the order is the same as the order in the tp_list and gt_list
        tp_list.sort()
        gt_list.sort()
        t_tp_list = [os.path.join(path, 'Tp', tp_list[index]) for index in range(len(tp_list))]
        t_gt_list = [os.path.join(path, 'Gt', gt_list[index]) for index in range(len(gt_list))]
        return t_tp_list, t_gt_list
    
class json_dataset(base_dataset):
    """ init from a json file, which contains all the images path
        file is organized as:
            [["./Tp/6.jpg", "./Gt/6.jpg"],
                ["./Tp/7.jpg", "./Gt/7.jpg"],
                ["./Tp/8.jpg", "Negative"],
                ......
            ]
        if path is "Neagative" then the image is negative sample, which means ground truths is a totally black image.
        
    Args:
        path (_type_): _description_
        transform_albu (_type_, optional): _description_. Defaults to None.
        mask_edge_generator (_type_, optional): _description_. Defaults to None.
        if_return_shape
    """
    def _init_dataset_path(self, path):
        images = json.load(open(path, 'r'))
        tp_list = []
        gt_list = []
        for record in images:
            tp_list.append(record[0])
            gt_list.append(record[1])
        return tp_list, gt_list



class balanced_dataset(Dataset):
    """平衡多个数据集的采样，确保每个数据集都能被均匀采样"""
    
    def __init__(self, 
                 path=None, 
                 sample_number=1840,
                 output_size = 512 ,
                 transform = None, 
                 if_return_shape = False,
                 ):
        self.sample_number = sample_number
        if path is None:
            # 默认数据集配置
            self.settings_list = [
                ["ManiDataset", "/mnt/data0/public_datasets/IML/CASIA2.0"],
                ["JsonDataset", "/mnt/data0/public_datasets/IML/FantasticReality_v1/FantasticReality.json"],
                ["ManiDataset", "/mnt/data0/public_datasets/IML/IMD_20_1024"],
                ["JsonDataset", "/mnt/data0/public_datasets/IML/tampCOCO/sp_COCO_list.json"],
                ["JsonDataset", "/mnt/data0/public_datasets/IML/tampCOCO/cm_COCO_list.json"],
                ["JsonDataset", "/mnt/data0/public_datasets/IML/tampCOCO/bcm_COCO_list.json"],
                ["JsonDataset", "/mnt/data0/public_datasets/IML/tampCOCO/bcmc_COCO_list.json"]
            ]
        else:
            import json
            with open(path, "r") as f:
                setting_json = json.load(f)
            self.settings_list = []
            for dataset_str, dataset_path in setting_json:
                # dataset_class = globals()[dataset_str]  # 通过类名获取对应的类

                self.settings_list.append([dataset_path, dataset_str])
        
        # 初始化各个数据集
        self.dataset_list = [self._get_dataset(path, dataset_str, output_size=output_size, transform=transform, if_return_shape=if_return_shape) 
                            for path, dataset_str in self.settings_list]

    def _get_dataset(self, path, dataset_type, output_size=512, transform=None, if_return_shape=False):
        if dataset_type == 'ManiDataset':
            return mani_dataset(path, output_size=output_size, transform=transform, if_return_shape=if_return_shape)
        elif dataset_type == 'JsonDataset':
            return json_dataset(path, output_size=output_size, transform=transform, if_return_shape=if_return_shape)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. Supported types are 'ManiDataset' and 'JsonDataset'.")
    def __len__(self):
        return self.sample_number * len(self.settings_list)
    
    def __getitem__(self, index):
        dataset_index = index // self.sample_number
        selected_dataset = self.dataset_list[dataset_index]
        length = len(selected_dataset)
        selected_item = random.randint(0, length - 1)
        return selected_dataset[selected_item]

