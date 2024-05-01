from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import glob
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
import torch.nn.functional as F 

from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from externals.utils import set_seed, make_dirs, cfg_init
from externals.dataloading import read_image_mask, get_train_valid_dataset, get_transforms, CustomDataset
from externals.models import effnet_v2, resnet18, effnet_v2_half, resnet18_regression, \
                             effnetv2_regression, effnetv2_m_regression, resnet_long_regression, resnet_short_regression, Unet3D_full3d, resnet18_3d, Unet3D_full3d_deep,\
                             Unet3D_full3d_shallow, Unet3D_full3d_xxl, Unet3D_full3d_deep
from externals.metrics import AverageMeter, calc_fbeta, fbeta_numpy
from externals.training_procedures import get_scheduler, scheduler_step
# from externals.postprocessing import post_process
from torch.optim.swa_utils import AveragedModel, SWALR
import wandb
import timm
import ast
import h5py
import segmentation_models_pytorch as smp
from monai.networks.nets.unetr import UNETR
from scipy.ndimage.filters import gaussian_filter

import zarr
from numcodecs import blosc, Blosc
import multiprocessing
blosc.set_nthreads(1)
compressor = Blosc("zstd", clevel=9)

dl = smp.losses.DiceLoss(mode="binary", ignore_index=-1, smooth=0)
bce = smp.losses.SoftBCEWithLogitsLoss(ignore_index=-1, smooth_factor=0, reduction="none")

from volumentations import *

def criterion(y_preds, y_true):
    y_preds = y_preds[y_true != -1]
    y_true = y_true[y_true != -1]
    return (
        # dl(y_preds, y_true) 
        # + \
        bce(y_preds, y_true)
        # cl(y_preds, y_true)
        )
    
class CFG:
    is_multiclass = True
    
    # edit these so they match your local data path
    comp_name = 'vesuvius_3d'
    comp_dir_path = './input'
    comp_folder_name = 'vesuvius-challenge-ink-detection'
    comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'
    # ========================
    
    exp_name = 'unet_pretrain'
    # ============== pred target =============
    target_size = 1
    # ============== model cfg =============
    model_name = '3d_unet'
    model_path = "/home/ryanc/kaggle/working/outputs/vesuvius_3d/128_looser_crop_int_invariance/vesuvius_3d-models/128_looser_crop_int_invariance_3d_unet.pth"
    # ============== training cfg =============
    size = 128
    tile_size = 128
    in_chans = 1

    train_batch_size = 24
    valid_batch_size = train_batch_size
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    epochs = 100
    valid_id = "856"
    # adamW warmup
    warmup_factor = 1
    lr = 1e-4 / warmup_factor
    # ============== fixed =============
    min_lr = 1e-6
    weight_decay = 1e-5
    max_grad_norm = 10
    num_workers = 4
    seed = 42
    # ============== set dataset path =============
    print('set dataset path')

    outputs_path = f'working/outputs/{comp_name}/{exp_name}/'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'


cfg_init(CFG)

def get_augmentation():
    return Compose([
        # Rotate((-15, 15), (-15, 15), (-15, 15), p=0.5),
        # RandomResizedCrop(shape=(CFG.size, CFG.size, CFG.size), p=0.9),
        # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        # Resize((CFG.size, CFG.size, CFG.size), interpolation=1, resize_type=0, always_apply=True, p=1.0),
        Flip(0, p=0.1),
        Flip(1, p=0.1),
        Flip(2, p=0.1),
        # RandomRotate90((0, 1), p=0.1),
        # RandomRotate90((0, 2), p=0.1),
        # RandomRotate90((1, 2), p=0.1),
        GaussianNoise(var_limit=(0, 5), p=0.1),
        # RandomGamma(gamma_limit=(80, 120), p=0.2),
        # GridDropout(p = 0.4)
    ], p=1.0)

aug = get_augmentation()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = effnet_v2_half(CFG)
# model = resnet18_regression(CFG)
model = Unet3D_full3d_shallow(CFG)
# model = resnet18_3d(CFG)

# model = UNETR(in_channels=1, out_channels=1, img_size=(CFG.size,CFG.size,CFG.size), proj_type='conv', norm_name='instance', )
from scipy import ndimage
class CustomDataset(Dataset):
    def __init__(self, volume_path, cfg, labels=None, transform=None, mode="test", size=1000, coords = None):
        self.volumes = volume_path
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.mode = mode
        self.size = size
        self.coords = coords
        self.shape = (7500, 7500, 13500)
    def __len__(self):
        if self.coords is not None:
            return len(self.coords)
        else:
            return self.size
    def __getitem__(self, idx):
        invalid_volume = True
        valid_volume_flag = False
        while invalid_volume:
            if self.coords is None:
                coords = (random.randint(0, self.shape[0]), random.randint(0, self.shape[1]), random.randint(0, self.shape[2]))
            else:
                coords = self.coords[idx]
                valid_volume_flag = True
            if self.mode == "train":
                x_offset, y_offset, z_offset = random.randint(0, self.cfg.size//2), random.randint(0, self.cfg.size//2), random.randint(0, self.cfg.size//2)
            else:
                x_offset, y_offset, z_offset = 0, 0, 0
            coords = np.array([coords[0] + x_offset, coords[1] + y_offset, coords[2] + z_offset])
            with h5py.File("depth_narrow_train.hdf5", 'r') as f:
                counts = f["scan_counts"][coords[0]:(coords[0]+((self.cfg.size))),
                                        coords[1]:(coords[1]+((self.cfg.size))),
                                        coords[2]:(coords[2]+((self.cfg.size)))]
                if ((counts > 0).sum() < 100) and (self.mode == "train"):
                    # print("skipped because not enough labeled")
                    continue
                else:
                    # print("valid sample")
                    predictions = f["scan_predictions"][coords[0]:(coords[0]+((self.cfg.size))),
                                            coords[1]:(coords[1]+((self.cfg.size))),
                                            coords[2]:(coords[2]+((self.cfg.size)))]/255.
                    invalid_volume = False
                    counts[counts == 0] = 1
                    predictions = predictions/counts
                    final_label = ndimage.maximum_filter(predictions, size=7)
                    
            if valid_volume_flag:
                invalid_volume = False
        with h5py.File("/data/volume_compressed.hdf5", 'r') as f:
            image = f["scan_volume"][coords[0]:(coords[0]+((self.cfg.size))),
                                    coords[1]:(coords[1]+((self.cfg.size))),
                                    coords[2]:(coords[2]+((self.cfg.size)))]/255.
                

        unlabeled = (final_label == 0)
        final_label[final_label > 0.5] = 1
        final_label[final_label < 0.5] = 0.0
        final_label[unlabeled] = -1
        if self.mode == "train":
            data = {'image': image, 'mask': final_label}
            aug_data = aug(**data)
            image, final_label = aug_data['image'], aug_data['mask']
        image, final_label = image.astype(np.float16), final_label.astype(np.float16)
        return image[None], final_label[None]

    
    
def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()
    model.to(device)
    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.autocast(device_type="cuda"):
            y_preds = model(images)
            if labels.max() == -1:
                continue
            # print(y_preds.shape, labels.shape)
            loss = criterion(y_preds, labels).mean()
            if torch.isnan(loss):
                continue
        pbar.set_description_str(str(losses.avg))
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    # torch.cuda.empty_cache()
    return losses.avg

def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    granular_losses = []
    losses = AverageMeter()
    dices = AverageMeter()
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    os.makedirs("./volume_predictions", exist_ok=True)
    for step, (images, labels) in pbar:
        os.makedirs(f"./volume_predictions/{step}", exist_ok=True)
        os.makedirs(f"./volume_labels/{step}", exist_ok=True)
        os.makedirs(f"./volume/{step}", exist_ok=True)



        batch_size = labels.size(0)
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                images = images.to(device)
                labels = labels.to(device)
                y_preds = model(images)
                loss = criterion(y_preds, labels)
                # granular_losses += loss.sum(axis = -1).sum(axis = -1).sum(axis = -1).detach().cpu().numpy().tolist()
                # print(labels.max())
                non_empties = (labels != -1)
                np_labels = labels[non_empties]
                np_preds = y_preds[non_empties]
                np_labels = (np_labels.detach().cpu().numpy().flatten() > .5).astype(int)
                # print(np_labels.max())
                # print(np_preds.max())

                np_preds = (torch.sigmoid(np_preds).detach().cpu().numpy().flatten())
                # print(np_label.mean(), np_pred.mean())
                # dice, _,_,_,_,_,_ = fbeta_numpy(np_labels, np_preds)
                dice, best_th, metrics = calc_fbeta(np_labels, np_preds)
                print(dice, best_th)
                # print(dice)
                
        for layer_num, layer in enumerate(images[0][0]):
            layer[layer < 0] = 0
            cv2.imwrite(f"./volume/{step}/{layer_num}.jpg", ((layer)*255.).detach().cpu().numpy())
        for layer_num, layer in enumerate(labels[0][0]):
            layer[layer < 0] = 0
            cv2.imwrite(f"./volume_labels/{step}/{layer_num}.jpg", ((layer)*255.).detach().cpu().numpy())
        for layer_num, layer in enumerate(y_preds[0][0]):
            cv2.imwrite(f"./volume_predictions/{step}/{layer_num}.jpg", (torch.sigmoid(layer)*255.).detach().cpu().numpy())
        pbar.set_description_str(str(losses.avg))
        losses.update(loss.mean().item(), batch_size)
        dices.update(dice, 1)
    print(dices.avg)
    # np.save("granular_losses.npy", granular_losses)
    return losses.avg
import random
import monai

def write_sub_volume(z, sub_volume, start_coords):
    # print(sub_volume.max(), start_coords)
    z[start_coords[0]:start_coords[0]+sub_volume.shape[0],
    start_coords[1]:start_coords[1]+sub_volume.shape[1],
    start_coords[2]:start_coords[2]+sub_volume.shape[2]] = sub_volume
                
def volume_valid_fn(valid_loader, model, criterion, device, coords):
    model.eval()
    crop_size = 1024
    chunk_size = 256
    with h5py.File("/data/scroll3.hdf5", 'r') as f:
        image = f["scan_volume"]
        z1 = zarr.open('/data/3d_predictions_scroll3.zarr', mode='w', shape=image.shape,
                chunks=(chunk_size, chunk_size, chunk_size), dtype=np.uint8, write_empty_chunks=False, compressor=compressor, synchronizer=zarr.ThreadSynchronizer())
        macro_generator = monai.data.utils.iter_patch_position(image.shape, (crop_size, crop_size, crop_size), overlap=0, padded=True)
        for coords in tqdm(macro_generator):
            volume = f["scan_volume"][coords[0]:(coords[0]+((crop_size))),
                                      coords[1]:(coords[1]+((crop_size))),
                                      coords[2]:(coords[2]+((crop_size)))]/255.
            orig_shape = volume.shape
            volume_holder = np.zeros((crop_size, crop_size, crop_size), dtype=np.float32)
            volume_holder[:orig_shape[0], :orig_shape[1], :orig_shape[2]] = volume
            volume = volume_holder
            prediction_volume = np.zeros_like(volume)
            prediction_counts = np.zeros_like(volume)
            coord_generator = monai.data.utils.iter_patch_position(volume.shape, (CFG.size, CFG.size, CFG.size), overlap=0.5)
            target_coords = []
            image_holder = []
            for x, y, z in coord_generator:
                image_holder.append(torch.tensor(volume[x:x+CFG.size, y:y+CFG.size, z:z+CFG.size][None][None]))
                target_coords.append([x, y, z])
                if len(image_holder) == CFG.valid_batch_size:
                    image = torch.cat(image_holder)
                    with torch.autocast(device_type="cuda"):
                        prediction = torch.sigmoid(model(image.to(torch.float16).to(device))).detach().cpu().numpy()
                        for i, (x, y, z) in enumerate(target_coords):
                            prediction_volume[x:x+CFG.size, y:y+CFG.size, z:z+CFG.size] += prediction[i][0]
                            prediction_counts[x:x+CFG.size, y:y+CFG.size, z:z+CFG.size] += 1
                    image_holder = []
                    target_coords = []
            if len(image_holder) > 0:
                image = torch.cat(image_holder)
                with torch.autocast(device_type="cuda"):
                    prediction = torch.sigmoid(model(image.to(torch.float16).to(device))).detach().cpu().numpy()
                    for i, (x, y, z) in enumerate(target_coords):
                        prediction_volume[x:x+CFG.size, y:y+CFG.size, z:z+CFG.size] += prediction[i][0]
                        prediction_counts[x:x+CFG.size, y:y+CFG.size, z:z+CFG.size] += 1
                image_holder = []
                target_coords = []
            prediction_counts[prediction_counts==0] = 1
            prediction_volume = prediction_volume/prediction_counts
            prediction_volume = prediction_volume[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
            prediction_volume = (prediction_volume*255).astype(np.uint8)
            
            processes = []

            for i in range(0, prediction_volume.shape[0], chunk_size):
                for j in range(0, prediction_volume.shape[1], chunk_size):
                    for k in range(0, prediction_volume.shape[2], chunk_size):
                        sub_volume = prediction_volume[i:i+chunk_size,
                                                j:j+chunk_size,
                                                k:k+chunk_size]
                        p = multiprocessing.Process(target=write_sub_volume,
                                                    args=(z1, sub_volume, (coords[0]+i, coords[1]+j, coords[2]+k)))
                        processes.append(p)
                        p.start()

            for process in processes:
                process.join()
            print(z1.info)
coords = np.load("/data/depth_narrow_filtered.npy")
print(coords.shape)

# handlabeled_layers = []
# with open("hand_labeled_segment_layers.txt", "r") as f:
#     for line in f:
#         if "/data/scroll_data/dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20231005123335/20231005123335_points.obj" in line:
#             layer_num = line.split(" ")[1].split("/")[-1].split(".")[0]
#             handlabeled_layers.append(int(layer_num))
# handlabeled_layers = np.array(handlabeled_layers)
focused_coords = [3861, 2208, 10696]

new_coords = []
for coord in coords:
    if (coord[0] > (focused_coords[0]-128)) & (coord[0] < focused_coords[0]+1024)&\
        (coord[1] > (focused_coords[1]-128)) & (coord[1] < focused_coords[1]+1024) &\
        (coord[2] > (focused_coords[2]-128)) & (coord[2] < focused_coords[2]+1024):
        new_coords.append(coord)
new_coords = np.array(new_coords)
print(len(coords))
# coords = coords[:3]
# train_coords = coords
# validation_coords = coords
from sklearn.model_selection import train_test_split

train_coords, validation_coords = train_test_split(coords, test_size=.1, shuffle=False)
training_dataset = CustomDataset(volume_path="/data/volume.hdf5", labels="depth_narrow_train.hdf5", cfg=CFG, transform=None, mode="train", size = 1000000, coords=None)
sampler = torch.utils.data.RandomSampler(training_dataset, replacement=True, num_samples=1000000)
train_loader = DataLoader(training_dataset, batch_size=CFG.train_batch_size, shuffle=False, num_workers=16, pin_memory=False, drop_last=True, sampler=sampler)

valid_dataset = CustomDataset(volume_path="/data/volume.hdf5", labels="depth_narrow_train.hdf5", cfg=CFG, transform=None, size = 1000, coords=new_coords)

valid_loader = DataLoader(valid_dataset, batch_size=CFG.train_batch_size, shuffle=False, num_workers=4, pin_memory=False)
cfg_pairs = {value:CFG.__dict__[value] for value in dir(CFG) if value[1] != "_"}
model_name = f"{CFG.exp_name}_{CFG.model_name}"

if os.path.exists(CFG.model_path):
    print("loading", CFG.model_path)
    model.load_state_dict(torch.load(CFG.model_path))
    
model = torch.nn.DataParallel(model)
model.to(device)
swa_model = AveragedModel(model)
swa_start = 2

best_counter = 0
best_loss = np.inf
best_score = 0
optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
# swa_scheduler = SWALR(optimizer, swa_lr=0.05)
scheduler = get_scheduler(CFG, optimizer)
for epoch in range(CFG.epochs):
    # train
    # avg_loss = train_fn(train_loader, model, criterion, optimizer, device)
    # torch.save(model.module.state_dict(),
    #         CFG.model_dir + f"{model_name}_final_128_all_focused.pth")
    # avg_val_loss = valid_fn(
    #     valid_loader, model, criterion, device)
    # print({"avg_train_loss":avg_loss, "avg_val_loss":avg_val_loss})
    
    volume_valid_fn(valid_loader, model, criterion, device, coords)
    break
    # if epoch > swa_start:
    #     swa_model.update_parameters(model)
    #     swa_scheduler.step()

    scheduler_step(scheduler, avg_val_loss, epoch)
    # score = avg_val_loss
    # torch.save(model.module.state_dict(),
    #         CFG.model_dir + f"{model_name}_final.pth")
# torch.optim.swa_utils.update_bn(train_loader, swa_model)
# torch.save(swa_model.module.state_dict(),
#     CFG.model_dir + f"{model_name}_final_swa.pth")