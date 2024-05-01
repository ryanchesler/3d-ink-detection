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
import time
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
                             Unet3D_full3d_shallow, Unet3D_full3d_xxl, Unet3D_full3d_deep, Unet3D_full3d_64
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
# from U_Mamba.umamba.nnunetv2.nets import UMambaBot
# from dynamic_network_architectures.building_blocks.residual import BasicBlockD
dl = smp.losses.DiceLoss(mode="binary", ignore_index=-1, smooth=0)
bce = smp.losses.SoftBCEWithLogitsLoss(ignore_index=-1, smooth_factor=0, reduction="none")

from volumentations import *
import zarr
def criterion(y_preds, y_true):
    y_preds = y_preds[y_true != -1]
    y_true = y_true[y_true != -1]
    # y_true[y_true > 0.5] = 1
    # y_true[y_true < 0.1] = 0.1

    return (
        # dl(y_preds.reshape(1, 1, -1), y_true.reshape(1, 1, -1)) 
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
    
    exp_name = '128_looser_crop_int_invariance'
    # starting_checkpoint = "/home/ryanc/kaggle/working/outputs/vesuvius_3d/pretrained/pretrained_model.pth"
    # starting_checkpoint = "None"
    starting_checkpoint = "/home/ryanc/kaggle/working/outputs/vesuvius_3d/128_looser_crop/vesuvius_3d-models/128_looser_crop_3d_unet.pth"
    # ============== pred target =============
    target_size = 1
    # ============== model cfg =============
    model_name = '3d_unet'
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
    lr = 1e-5 / warmup_factor
    # ============== fixed =============
    min_lr = 1e-6
    weight_decay = 1e-5
    max_grad_norm = 10
    num_workers = 4
    seed = int(time.time())
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
        Rotate((-30, 30), (-30, 30), (-30, 30), p=0.1, interpolation=0, value = 0, mask_value=-1),
        # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        # Resize((CFG.size, CFG.size, CFG.size), interpolation=1, resize_type=0, always_apply=True, p=1.0),
        Flip(0, p=0.2),
        Flip(1, p=0.2),
        Flip(2, p=0.2),
        RandomRotate90((0, 1), p=0.2), #.368
        RandomRotate90((0, 2), p=0.2), #.363
        RandomRotate90((1, 2), p=0.2), # 40
        # GaussianNoise(p=1),
        RandomResizedCrop((CFG.size, CFG.size, CFG.size), scale_limit=(.75, 1.33), interpolation=0, resize_type=0, p = 1),
        # ResizedCropNonEmptyMaskIfExists(shape=(CFG.size, CFG.size, CFG.size), p=.1),
        # RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p =.5),
    ], p=1.0)

aug = get_augmentation()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = effnet_v2_half(CFG)
# model = resnet18_regression(CFG)
model = Unet3D_full3d_shallow(CFG)
# model = Unet3D_full3d_64(CFG)
# model = resnet18_3d(CFG)
# model = UMambaBot.UMambaBot(
#                  input_channels = 1,
#                  n_stages = 5,
#                  features_per_stage = [64, 128, 256, 512, 1024],
#                  conv_op = nn.Conv3d,
#                  kernel_sizes = 3,
#                  strides= (1, 2, 2, 2, 2),
#                  n_conv_per_stage = [2, 2, 2, 2, 2],
#                  num_classes = 1,
#                  n_conv_per_stage_decoder = [2, 2, 2, 2],
#                  conv_bias = True,
#                  norm_op = nn.InstanceNorm3d,
#                  norm_op_kwargs = None,
#                  dropout_op = None,
#                  dropout_op_kwargs = None,
#                  nonlin = nn.LeakyReLU,
#                  nonlin_kwargs = {'inplace': True},
#                  deep_supervision = False,
#                  block = BasicBlockD,
#                  bottleneck_channels = None,
#                  stem_channels = None
#                  )
# from nnunetv2.utilities.network_initialization import InitWeights_He
# model.apply(InitWeights_He(1e-2))
# print(model)
# model = UNETR(in_channels=1, out_channels=1, img_size=(CFG.size,CFG.size,CFG.size), proj_type='conv', norm_name='instance')
from scipy import ndimage
class CustomDataset(Dataset):
    def __init__(self, volume_path, cfg, labels=None, transform=None, mode="test", size=1000, coords = None, check_counts = True):
        self.volumes = volume_path
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.mode = mode
        self.size = size
        if self.mode == "train":
            self.crop_size = int(self.cfg.size *1.5)
        else:
            self.crop_size = self.cfg.size
        self.coords = coords
        self.shape = (7500, 7500, 13500)
        self.check_counts = check_counts
    def __len__(self):
        if self.coords is not None:
            return len(np.concatenate(self.coords))
        else:
            return self.size
    def __getitem__(self, idx):
        invalid_volume = True
        valid_volume_flag = False
        while invalid_volume:
            if self.coords is None:
                coords = (random.randint(0, self.shape[0]), random.randint(0, self.shape[1]), random.randint(0, self.shape[2]))
            else:
                if self.mode == "train":
                    pos = int(random.random() < 0.05)
                else:
                    pos = int(random.random() < 0.5)
                idx = (idx + 1000) % len(self.coords[pos])
                coords = self.coords[pos][idx][[1, 0, 2]]
                # print(coords)
                # valid_volume_flag = True
            # if self.mode == "train":
            x_offset, y_offset, z_offset = random.randint(-self.crop_size, 0), random.randint(-self.crop_size, 0), random.randint(-self.crop_size, 0)
            # x_offset, y_offset, z_offset = -self.crop_size//2, -self.crop_size//2, -self.crop_size//2
            coords = np.array([coords[0] + x_offset, coords[1] + y_offset, coords[2] + z_offset])
            if self.labels == '3d_predictions_uint8_compressed.zarr':
                predictions = zarr.open('3d_predictions_uint8_compressed.zarr', mode='r')[coords[0]:(coords[0]+((self.crop_size))),
                                            coords[1]:(coords[1]+((self.crop_size))),
                                            coords[2]:(coords[2]+((self.crop_size)))]/255.
                if predictions.size == (self.crop_size **3):
                    if predictions.max() > 0:
                        invalid_volume = False
                        final_label = predictions
                        unlabeled = (final_label == 0)
                        final_label[final_label > 0.5] = 1
                        final_label[final_label < 0.5] = 0.0
                        final_label[unlabeled] = -1
                        # final_label = ndimage.maximum_filter(predictions, size=7)
            else:
                with h5py.File(self.labels, 'r') as f:
                    if self.check_counts:
                        counts = f["scan_counts"][coords[0]:(coords[0]+(self.crop_size)),
                                                coords[1]:(coords[1]+(self.crop_size)),
                                                coords[2]:(coords[2]+(self.crop_size))]
                        unlabeled = (counts == 0).copy()
                        # print(counts.sum())
                        if ((counts > 0).sum() < 10000):
                            continue
                        else:
                            counts[counts == 0] = 1
                    else:
                        counts = np.zeros((self.cfg.size, self.cfg.size, self.cfg.size)) + 1

                    predictions = f["scan_predictions"][coords[0]:(coords[0]+(self.crop_size)),
                                            coords[1]:(coords[1]+(self.crop_size)),
                                            coords[2]:(coords[2]+(self.crop_size))]/255.
                    final_label = predictions/counts
                    # neg_label = (ndimage.maximum_filter(final_label, size=7) - final_label).astype(bool)
                    final_label[unlabeled] = -1
                    # zero_sum = (final_label == 0).sum()
                    # one_sum = (final_label == 1).sum()
                    # # print(label_mean)
                    # if (zero_sum < 10000) or (one_sum < 10000):
                    #     # print("all or nothing")
                    #     continue

                    # final_label[neg_label] = 0
                    # print((final_label == 1).sum(), (final_label == 0).sum(), (final_label == 0.1).sum())
                    if (final_label == -1).mean() < 0.95:
                        invalid_volume = False
                    
            if valid_volume_flag:
                invalid_volume = False
        with h5py.File(self.volumes, 'r') as f:
            image = f["scan_volume"][coords[0]:(coords[0]+(self.crop_size)),
                                    coords[1]:(coords[1]+(self.crop_size)),
                                    coords[2]:(coords[2]+(self.crop_size))]/255.
                
        if self.mode == "train":
            data = {'image': image, 'mask': final_label}
            aug_data = aug(**data)
            image, final_label = aug_data['image'], aug_data['mask']
            # print(final_label.max(), final_label.min())
            # print(image.max(), image.min(), final_label.max(), final_label.min())
            image = image * random.uniform(0.4, 2)
            # print(image.max(), image.min())

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
            # print(images.max(), images.min(), images.dtype)
            y_preds = model(images)
            # print(y_preds.max(), y_preds.min())
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
    metric_preds = []
    metric_labels = []
    os.makedirs("./volume_predictions", exist_ok=True)
    for step, (images, labels) in pbar:
        os.makedirs(f"./volume_predictions/{step}", exist_ok=True)
        os.makedirs(f"./volume_labels/{step}", exist_ok=True)
        os.makedirs(f"./volume/{step}", exist_ok=True)
        os.makedirs(f"./pred_video/", exist_ok=True)


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
                metric_preds.append(np_preds)
                metric_labels.append(np_labels)

                # print(np_label.mean(), np_pred.mean())
                # dice, _,_,_,_,_,_ = fbeta_numpy(np_labels, np_preds)
                
        # Set up VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f'./pred_video/{step}.avi', fourcc, 4, (images.shape[-1]*3, images.shape[-2]))
        # Process each layer and write to video
        for layer_num in range(images.shape[-1]):  # Assuming depth is the third dimension
            # Process images
            image_layer = (images[0, 0, :, :, layer_num].clamp(min=0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            # cv2.putText(image_layer, 'Input', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Process labels
            label_layer = (labels[0, 0, :, :, layer_num].clamp(min=0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            # cv2.putText(label_layer, 'Labels', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            
            # Process predictions
            y_pred_layer = torch.sigmoid(y_preds[0, 0, :, :, layer_num]) * 255.0
            y_pred_layer = y_pred_layer.detach().cpu().numpy().astype(np.uint8)
            # cv2.putText(y_pred_layer, 'Predictions', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Concatenate images horizontally
            # print(y_pred_layer.max(), y_pred_layer.min(), label_layer.max(), label_layer.min(), image_layer.max(), image_layer.min())
            concatenated_frame = np.concatenate((image_layer, label_layer, y_pred_layer), axis=-1).astype(np.uint8)
            # Write frame to video
            out.write(cv2.cvtColor(concatenated_frame, cv2.COLOR_GRAY2BGR))
        out.release()
        pbar.set_description_str(str(losses.avg))
        losses.update(loss.mean().item(), batch_size)
    dice, best_th, metrics = calc_fbeta(np.concatenate(metric_labels, axis = 0), np.concatenate(metric_preds, axis = 0))
    print(dice, best_th)
    dices.update(dice, 1)
    # np.save("granular_losses.npy", granular_losses)
    return losses.avg, dices.avg
import random
import monai

def volume_valid_fn(valid_loader, model, criterion, device, coords):
    model.eval()
    losses = AverageMeter()
    # pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    crop_size = 512
    prediction_volume = np.zeros(shape=(crop_size,crop_size,crop_size))
    prediction_counts = np.zeros(shape=(crop_size,crop_size,crop_size))
    os.makedirs("./big_volume_predictions", exist_ok=True)
    os.makedirs("./big_volume_2d_predictions", exist_ok=True)
    os.makedirs("./big_volume", exist_ok=True)

    with h5py.File("/data/volume_compressed.hdf5", 'r') as f:
        image = f["scan_volume"]
        x, y, z = image.shape
        valid_volume = False
        while not valid_volume:
            coords = random.choice(coords)
            # coords = coords[20000]
            # coords = [2540, 2275, 9950]
            # coords = [3861, 2208, 10696]
            # coords = [3272, 2137, 11200]
            volume = f["scan_volume"][coords[1]-(crop_size)//2:(coords[1]+((crop_size))//2),
                                      coords[0]-(crop_size)//2:(coords[0]+((crop_size))//2),
                                      coords[2]-(crop_size)//2:(coords[2]+((crop_size))//2)]/255.
            if volume.shape == (crop_size, crop_size, crop_size):
                valid_volume = True
                
    with h5py.File('/data/orig_labels_valid.hdf5', 'r') as f:
        predictions = f["scan_predictions"][coords[1]-(crop_size)//2:(coords[1]+((crop_size))//2),
                                      coords[0]-(crop_size)//2:(coords[0]+((crop_size))//2),
                                      coords[2]-(crop_size)//2:(coords[2]+((crop_size))//2)]
        counts = f["scan_counts"][coords[1]-(crop_size)//2:(coords[1]+((crop_size))//2),
                                      coords[0]-(crop_size)//2:(coords[0]+((crop_size))//2),
                                      coords[2]-(crop_size)//2:(coords[2]+((crop_size))//2)]
        counts[counts == 0] = 1
        predictions = predictions/counts
        predictions = ndimage.maximum_filter(predictions, size=7)


        print(coords)
        print(predictions.mean())
            
    coord_generator = monai.data.utils.iter_patch_position(volume.shape, (CFG.size, CFG.size, CFG.size), overlap=0.5)
    target_coords = []
    image_holder = []
    for x, y, z in tqdm(coord_generator):
        image_holder.append(torch.tensor(volume[x:x+CFG.size, y:y+CFG.size, z:z+CFG.size][None][None]))
        target_coords.append([x, y, z])
        if len(image_holder) == CFG.valid_batch_size:
            image = torch.cat(image_holder)
            with torch.autocast(device_type="cuda"):
                prediction = torch.sigmoid(model(image.to(torch.float16).to(device))).detach().cpu()
                for i, (x, y, z) in enumerate(target_coords):
                    prediction_volume[x:x+CFG.size, y:y+CFG.size, z:z+CFG.size] += prediction[i][0].detach().cpu().numpy()
                    prediction_counts[x:x+CFG.size, y:y+CFG.size, z:z+CFG.size] += 1
            image_holder = []
            target_coords = []
    if len(image_holder) > 0:
        image = torch.cat(image_holder)
        with torch.autocast(device_type="cuda"):
            prediction = torch.sigmoid(model(image.to(torch.float16).to(device))).detach().cpu()
            for i, (x, y, z) in enumerate(target_coords):
                prediction_volume[x:x+CFG.size, y:y+CFG.size, z:z+CFG.size] += prediction[i][0].detach().cpu().numpy()
                prediction_counts[x:x+CFG.size, y:y+CFG.size, z:z+CFG.size] += 1
        image_holder = []
        target_coords = []
            
    # torch.cuda.empty_cache()
    prediction_volume = prediction_volume/prediction_counts
    # predictions[volume < 0.3] = 0
        
    for layer_num, layer in enumerate(prediction_volume):
        cv2.imwrite(f"./big_volume_predictions/{layer_num}.jpg", (layer*255).astype(np.uint8))
    for layer_num, layer in enumerate(predictions):
        # layer[layer < 0] = 0
        print(f"2d prediction {cv2.imwrite(f'./big_volume_2d_predictions/{layer_num}.jpg', layer)}")
        # layer[layer==1] = 0
    for layer_num, layer in enumerate(volume):
        cv2.imwrite(f"./big_volume/{layer_num}.jpg", (layer*255).astype(np.uint8))
    # torch.cuda.empty_cache()
    
train_pos_coords = np.load("train_coords_128.npy").astype(int)
train_neg_coords = np.load("train_coords_128.npy").astype(int)
valid_pos_coords = np.load("valid_coords_pos.npy").astype(int)
valid_neg_coords = np.load("valid_coords_pos.npy").astype(int)

from scipy.spatial.distance import cdist
distances = cdist(train_pos_coords, valid_pos_coords, 'euclidean')
min_distance = np.min(distances)
indices_to_keep = np.all(distances >= 512, axis=1)
print(f"Original number of points in array1: {train_pos_coords.shape[0]}")
train_pos_coords = train_pos_coords[indices_to_keep]
print(f"Number of points after filtering: {train_pos_coords.shape[0]}")

train_pos_coords = train_pos_coords[:,(1, 0, 2)]
train_neg_coords = train_neg_coords[:,(1, 0, 2)]
valid_pos_coords = valid_pos_coords[:,(1, 0, 2)]
valid_neg_coords = valid_neg_coords[:,(1, 0, 2)]

training_dataset = CustomDataset(volume_path="/data/volume_compressed.hdf5", labels='/data/all_labels_train.hdf5', cfg=CFG,
                                 transform=None, mode="train", size = 100000, coords=[train_pos_coords, train_pos_coords], check_counts=True)
sampler = torch.utils.data.RandomSampler(training_dataset, replacement=False, num_samples=100000)
train_loader = DataLoader(training_dataset, batch_size=CFG.train_batch_size, shuffle=False, num_workers=32, pin_memory=False, drop_last=True, sampler=sampler)

valid_dataset = CustomDataset(volume_path="/data/volume_compressed.hdf5", labels='/data/orig_labels_valid.hdf5', cfg=CFG, transform=None, size = 1000, coords=[valid_pos_coords, valid_neg_coords])
valid_sampler = torch.utils.data.RandomSampler(valid_dataset, replacement=True, num_samples=1000)
valid_loader = DataLoader(valid_dataset, batch_size=CFG.train_batch_size, shuffle=False, num_workers=32, pin_memory=False, drop_last=True, sampler=valid_sampler)
cfg_pairs = {value:CFG.__dict__[value] for value in dir(CFG) if value[1] != "_"}
model_name = f"{CFG.exp_name}_{CFG.model_name}"

if os.path.exists(CFG.starting_checkpoint):
    print(CFG.starting_checkpoint)
    model.load_state_dict(torch.load(CFG.starting_checkpoint))
    
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

wandb.init(
    project="3d_vesuvius",
    name=CFG.exp_name
)
for epoch in range(CFG.epochs):
    # train
    avg_loss = train_fn(train_loader, model, criterion, optimizer, device)

    avg_val_loss, avg_val_dice = valid_fn(
        valid_loader, model, criterion, device)
    print({"avg_train_loss":avg_loss, "avg_val_loss":avg_val_loss})
    wandb.log({"avg_train_loss":avg_loss, "avg_val_loss":avg_val_loss, "val_dice":avg_val_dice})
    if avg_val_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.module.state_dict(),
                CFG.model_dir + f"{model_name}.pth")
    try:
        volume_valid_fn(valid_loader, model, criterion, device, valid_pos_coords)
    except:
        pass
    # break
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