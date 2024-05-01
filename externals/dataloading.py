from tqdm.auto import tqdm
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset
import os
import torch

def read_image_mask(CFG, fragment_id, pseudolabel=True):
    images = []

    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)
    
    #trys to load labels or predictions. If present then load the image. Else skip entirely
    try:
        if os.path.exists(f"{CFG.label_path}{fragment_id}_inklabels.png"):
            print("loading mask success")
            mask = cv2.imread(f"{CFG.label_path}{fragment_id}_inklabels.png", 0)
            # mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
        else:
            print("loading mask failed:", f"{CFG.label_path}{fragment_id}_inklabels.png")
            raise Exception
    except:
        mask = None
    if pseudolabel:
        if os.path.exists(f"{CFG.pseudolabel_path}{fragment_id}_predictions.png"):
            print(f"loading pseudolabel {CFG.pseudolabel_path}{fragment_id}_predictions.png")
            # pseudolabel = np.zeros_like(mask)
            pseudolabel = cv2.imread(f"{CFG.pseudolabel_path}{fragment_id}_predictions.png", 0)
            high_split = .5*255
            pseudolabel[pseudolabel > high_split] = 254
            if mask is not None:
                mask = mask[:pseudolabel.shape[0], :pseudolabel.shape[1]]
                pseudolabel = pseudolabel[:mask.shape[0], :mask.shape[1]]
                mask = np.maximum(mask, pseudolabel)
            else:
                mask = pseudolabel
            mask[pseudolabel < high_split] = 10
    if mask is not None:
        for i in tqdm(idxs):
            image = cv2.imread(f"{CFG.tif_path}{fragment_id}/layers/{i:02}.tif", 0)
            pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
            pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)
            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
            images.append(image)
        mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    else:
        return np.zeros((50,50,50,50)), np.zeros((50,50,50,50))
    images = np.stack(images, axis=2)
    if mask is None:
        return images, np.zeros_like(images)
    mask = mask[:images.shape[0], :images.shape[1]]
    print(images.shape, mask.shape)
    return images, mask

def get_train_valid_dataset(CFG, fragment_ids):
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []
    offset = 0
    for fragment_id in fragment_ids:
        fragment_pieces = 0
        try:
            image, mask = read_image_mask(CFG, fragment_id)
        except:
            print(fragment_id)
            continue

        x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
        y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))

        for y1 in y1_list:
            for index, x1 in enumerate(x1_list):
                y2 = y1 + CFG.tile_size
                x2 = x1 + CFG.tile_size
                # xyxys.append((x1, y1, x2, y2))
                if fragment_id == CFG.valid_id:
                    if image[y1:y2, x1:x2, None].max() != 0:
                        if (mask[y1:y2, x1:x2, None] == 255).mean() > 0:

                            valid_images.append(image[y1:y2, x1:x2])
                            valid_masks.append((mask[y1:y2, x1:x2, None] == 255))
                            valid_xyxys.append([x1, y1, x2, y2])
                            fragment_pieces += 1
                else:
                    # offset = np.random.randint(0, 0)
                    offset = 0
                    if image[y1+offset:y2+offset, x1+offset:x2+offset, None].max() != 0:
                        if (mask[y1+offset:y2+offset, x1+offset:x2+offset, None] > 200).mean() > 0.01:
                            train_images.append(image[y1+offset:y2+offset, x1+offset:x2+offset])
                            train_masks.append((mask[y1+offset:y2+offset, x1+offset:x2+offset, None]))
                            fragment_pieces += 1
                        else:
                            pass #skip unlabeled pieces
        print(fragment_pieces)            
    if CFG.valid_id == None:
        return train_images, train_masks
    return train_images, train_masks, valid_images, valid_masks, valid_xyxys

def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug

from volumentations import Rotate, Compose
def get_3d_augmentation():
    return Compose([
        # Rotate((-5, 5), (-5, 5), (-5, 5), p=0.5)
    ], p=1.0)

aug_3d = get_3d_augmentation()

class CustomDataset(Dataset):
    def __init__(self, images, cfg, labels=None, transform=None, layer_flips = None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.layer_flips = layer_flips

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            data = {'image': image}
            aug_data = aug_3d(**data)
            image = aug_data['image']
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']
            if np.random.random() > .5 and self.layer_flips:
                image = torch.flip(image, dims = [0])
        # print(label.max(), label.min())
        return image[None, :, :, :], label.to(torch.float32)/255.0