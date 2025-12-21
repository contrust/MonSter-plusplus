import os
import shutil
import hydra
import torch
from tqdm import tqdm
import torch.optim as optim
from core.utils.utils import InputPadder
from core.monster import Monster 
from omegaconf import OmegaConf
import torch.nn.functional as F
from accelerate import Accelerator
import core.stereo_datasets as datasets
from accelerate.utils import broadcast_object_list, set_seed
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from core.warp import disp_warp



import matplotlib
import numpy as np
from pathlib import Path
import torch.distributed as dist
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import heapq
from accelerate.utils import gather_object
import pickle

class EarlyStopper:
    def __init__(self, patience: int, min_delta: float, min_mode: bool) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.patience = patience
        self.min_delta = min_delta
        self.min_mode = min_mode
        self.current_epoch = 0
        self.best_score = float('inf') if min_mode else float('-inf')
        self.best_epoch = 0
        print(f"EarlyStopper: patience: {self.patience}, min_delta: {self.min_delta}, min_mode: {self.min_mode}")

    def __call__(self, score: float, epoch: int) -> bool:
        print(f"EarlyStopper: score: {score}, epoch: {epoch}, best_score: {self.best_score}, best_epoch: {self.best_epoch}, min_delta: {self.min_delta}, min_mode: {self.min_mode}, current_epoch: {self.current_epoch}")
        self.current_epoch = epoch
        is_current_best = score < self.best_score - self.min_delta if self.min_mode else score > self.best_score + self.min_delta
        if is_current_best:
            self.best_score = score
            self.best_epoch = epoch
            return False
        return self.current_epoch - self.best_epoch >= self.patience

def gray_2_colormap_np(img, max_disp=None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    
    # Initialize output colormap
    colormap = np.zeros((*img.shape, 3), dtype=np.uint8)
    
    # Mask for -900 values
    mask_invalid = (img < -900)
    
    # Separate handling for negatives and positives
    neg_mask = (img < 0) & ~mask_invalid
    pos_mask = (img > 0) & ~mask_invalid
    
    if neg_mask.any():
        # Handle negative values with red colormap
        neg_values = -img[neg_mask]  # Make positive for scaling
        neg_max = neg_values.max() if max_disp is None else max_disp
        neg_norm = neg_values / (neg_max + 1e-8)
        neg_norm = np.clip(neg_norm, 0, 1)
        
        # Red colormap for negatives (red channel increases with magnitude)
        colormap[neg_mask, 0] = (neg_norm * 255).astype(np.uint8)  # Red
        colormap[neg_mask, 1] = 0  # Green
        colormap[neg_mask, 2] = 0  # Blue
    
    if pos_mask.any():
        # Handle positive values with blue colormap
        pos_values = img[pos_mask]
        pos_max = pos_values.max() if max_disp is None else max_disp
        pos_norm = pos_values / (pos_max + 1e-8)
        pos_norm = np.clip(pos_norm, 0, 1)
        
        # Blue colormap for positives (blue channel increases with magnitude)
        colormap[pos_mask, 0] = 0  # Red
        colormap[pos_mask, 1] = 0  # Green
        colormap[pos_mask, 2] = (pos_norm * 255).astype(np.uint8)  # Blue
    
    # Set -999 values to black
    colormap[mask_invalid] = [255, 255, 255]
    
    return colormap

def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    # quantile = torch.quantile((disp_init_pred - disp_gt).abs(), 0.9)
    init_valid = valid.bool() & ~torch.isnan(disp_init_pred)#  & ((disp_init_pred - disp_gt).abs() < quantile)
    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[init_valid], disp_gt[init_valid], reduction='mean')
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        # quantile = torch.quantile(i_loss, 0.9)
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    if valid.bool().sum() == 0:
        epe = torch.Tensor([0.0]).cuda()

    metrics = {
        'train/epe': epe.mean(),
        'train/d1_1px': (epe > 1).float().mean() * 100,
        'train/d1_2px': (epe > 2).float().mean() * 100,
        'train/d1_3px': (epe > 3).float().mean() * 100,
        'train/d1_4px': (epe > 4).float().mean() * 100,
        'train/d1_5px': (epe > 5).float().mean() * 100,
    }
    return disp_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    DPT_params = list(map(id, model.feat_decoder.parameters())) 
    rest_params = filter(lambda x:id(x) not in DPT_params and x.requires_grad, model.parameters())

    params_dict = [{'params': model.feat_decoder.parameters(), 'lr': args.lr/2.0}, 
                   {'params': rest_params, 'lr': args.lr}, ]
    optimizer = optim.AdamW(params_dict, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, [args.lr/2.0, args.lr], args.total_step+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


@hydra.main(version_base=None, config_path='config', config_name='train_us3d')
def main(cfg):
    print(cfg)
    set_seed(cfg.seed)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=None, dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, split_batches=True), log_with='tensorboard', project_dir=cfg.project_dir, kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    hparams_config = {}
    
    for key, value in config_dict.items():
        if isinstance(value, (int, float, str, bool)):
            # Keep scalars as-is
            hparams_config[key] = value
        elif torch.is_tensor(value):
            # Keep tensors as-is (though they may not display well in TensorBoard)
            hparams_config[key] = value
        elif isinstance(value, (list, tuple)):
            # Flatten arrays into individual scalar parameters
            for i, item in enumerate(value):
                if isinstance(item, (int, float, str, bool)):
                    hparams_config[f"{key}{i}"] = item
                else:
                    # Convert complex nested items to strings
                    hparams_config[f"{key}{i}"] = str(item)
        else:
            # Convert other complex types (dicts, etc.) to strings
            hparams_config[key] = str(value)
    
    accelerator.init_trackers(project_name=cfg.project_name, config=hparams_config, init_kwargs={'tensorboard': cfg.tensorboard})

    train_dataset = datasets.fetch_dataloader(cfg)
    val_dataset = datasets.US3D(aug_params=None, split='val')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size,
        pin_memory=True, shuffle=True, num_workers=int(1), drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size,
        pin_memory=True, shuffle=False, num_workers=int(1), drop_last=False)
    model = Monster(cfg)
    optimizer, lr_scheduler = fetch_optimizer(cfg, model)

    if not cfg.restore_ckpt.endswith("None"):
        assert cfg.restore_ckpt.endswith(".pth")
        print(f"Loading checkpoint from {cfg.restore_ckpt}")
        assert os.path.exists(cfg.restore_ckpt)
        checkpoint = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt = dict()
        if 'state_dict' in checkpoint.keys():
            print("Loading checkpoint from state_dict...")
            checkpoint = checkpoint['state_dict']
            for key in checkpoint:
                if key.startswith("module."):
                    ckpt[key.replace('module.', '')] = checkpoint[key]
                else:
                    ckpt[key] = checkpoint[key]
            model.load_state_dict(ckpt, strict=True)
            total_step = 0
        elif 'model' in checkpoint.keys():
            print("Loading checkpoint from model...")
            checkpoint = checkpoint['model']
            for key in checkpoint:
                if key.startswith("module."):
                    ckpt[key.replace('module.', '')] = checkpoint[key]
                else:
                    ckpt[key] = checkpoint[key]
            model.load_state_dict(ckpt, strict=True)
        else:
            print("Loading checkpoint from raw checkpoint...")
            for key in checkpoint:
                if key.startswith("module."):
                    ckpt[key.replace('module.', '')] = checkpoint[key]
                else:
                    ckpt[key] = checkpoint[key]
            model.load_state_dict(ckpt, strict=True)
        print(f"Loaded checkpoint from {cfg.restore_ckpt} successfully")
        del ckpt, checkpoint
    else:
        total_step = 0

    total_step = cfg.current_total_step
    train_loader, val_loader, model, optimizer, lr_scheduler = accelerator.prepare(train_loader, val_loader, model, optimizer, lr_scheduler)
    should_keep_training = True
    epoch = 0
    d1_early_stopper = EarlyStopper(patience=cfg.d1_patience, min_delta=cfg.d1_min_delta, min_mode=True)
    epe_early_stopper = EarlyStopper(patience=cfg.epe_patience, min_delta=cfg.epe_min_delta, min_mode=True)

    # Initialize tracking for worst/best images (now using file paths)
    historical_file_paths_epe_worst = []
    historical_file_paths_epe_best = []
    historical_file_paths_d1_worst = []
    historical_file_paths_d1_best = []
    current_file_paths_epe_worst = []
    current_file_paths_epe_best = []
    current_file_paths_d1_worst = []
    current_file_paths_d1_best = []
    if not cfg.restore_ckpt.endswith("None"):
        # Load historical file paths from checkpoint if available
        checkpoint = torch.load(cfg.restore_ckpt, map_location='cpu')
        if 'historical_file_paths_epe_worst' in checkpoint:
            historical_file_paths_epe_worst = checkpoint['historical_file_paths_epe_worst']
            historical_file_paths_epe_best = checkpoint['historical_file_paths_epe_best']
            historical_file_paths_d1_worst = checkpoint['historical_file_paths_d1_worst']
            historical_file_paths_d1_best = checkpoint['historical_file_paths_d1_best']
            print("Loaded historical file paths from checkpoint")
    else:
        print("No historical file paths loaded; will compute on first epoch")   


    # For current epoch tracking (using heaps for efficiency)
    num_worst_images = cfg.num_worst_images
    num_best_images = cfg.num_best_images

    project_metrics_path = os.path.join(cfg.metrics_base_path, cfg.project_name)
    epe_metrics_path = os.path.join(project_metrics_path, 'epe')
    d1_metrics_path = os.path.join(project_metrics_path, 'd1')
    temp_dir = os.path.join(cfg.temp_dir, cfg.project_name)
    os.makedirs(project_metrics_path, exist_ok=True)
    os.makedirs(epe_metrics_path, exist_ok=True)
    os.makedirs(d1_metrics_path, exist_ok=True)
    try:
        while should_keep_training:
            
            if epoch % cfg.val_frequency == 0:
                if accelerator.is_main_process:
                    os.makedirs(temp_dir, exist_ok=True)
                accelerator.wait_for_everyone()
                torch.cuda.empty_cache()
                model.eval()
                elem_num, total_epe, total_out = 0, 0, 0
                epe_list, d1_list = [], []
                for data in tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process, desc="Validating"):
                    (imageL_file, imageR_file, GT_file), left, right, disp_gt, valid = [x for x in data]
                    padder = InputPadder(left.shape, divis_by=32)
                    left, right = padder.pad(left, right)
                    with torch.no_grad():
                        disp_pred = model(left, right, iters=cfg.valid_iters, test_mode=True)
                    disp_pred = padder.unpad(disp_pred)
                    assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
                    
                    for b in range(left.shape[0]):
                        # Compute metrics exactly as in sequence_loss and logging
                        valid_mask = valid[b] >= 0.5
                        if valid_mask.sum() == 0:
                            continue
                        
                        disp_pred_b = disp_pred[b].squeeze(0)  # [H, W]
                        disp_gt_b = disp_gt[b].squeeze(0)      # [H, W]
                        epe_per_pixel = torch.abs(disp_pred_b - disp_gt_b)  # [H, W]
                        
                        # EPE: Mean of per-pixel absolute errors (matching sequence_loss)
                        epe_mean = epe_per_pixel[valid_mask].mean().item()
                        
                        # D1: Percentage of valid pixels with error > 3px (matching logged d1_3px)
                        d1_mean = (epe_per_pixel > 3.0).float()[valid_mask].mean().item()  # Percentage
                        
                        sample_file_path = imageL_file[b]  # Use left image file path as identifier

                        # Prepare tracking tuples
                        epe_tuple = (epe_mean, sample_file_path)
                        d1_tuple = (d1_mean * 100, sample_file_path)

                        # Update current epoch tracking (always, for all epochs to handle first epoch)
                        d1_list.append(d1_tuple)
                        epe_list.append(epe_tuple)
                    
                # Each process saves its tracking data to unique files
                with open(os.path.join(temp_dir, f'epe_list_{accelerator.process_index}.pkl'), 'wb') as f:
                    print(f"Saving epe list of length {len(epe_list)} to epe_list_{accelerator.process_index}.pkl")
                    pickle.dump(epe_list, f)
                with open(os.path.join(temp_dir, f'd1_list_{accelerator.process_index}.pkl'), 'wb') as f:
                    print(f"Saving d1 list of length {len(d1_list)} to d1_list_{accelerator.process_index}.pkl")
                    pickle.dump(d1_list, f)

                accelerator.wait_for_everyone()  # Ensure all files are written

                # On main process, load and aggregate from files
                # Reset the lists
                aggregated_epe_list = []
                aggregated_d1_list = []
                
                # Load and aggregate for each type
                print(f"Aggregating epe and d1 lists from {accelerator.process_index} process")
                for i in range(accelerator.num_processes):
                    with open(os.path.join(temp_dir, f'epe_list_{i}.pkl'), 'rb') as f:
                        data = pickle.load(f)
                        print(f"Loaded epe list of length {len(data)} from epe_list_{i}.pkl")
                        aggregated_epe_list.extend(data)
                    with open(os.path.join(temp_dir, f'd1_list_{i}.pkl'), 'rb') as f:
                        data = pickle.load(f)
                        print(f"Loaded d1 list of length {len(data)} from d1_list_{i}.pkl")
                        aggregated_d1_list.extend(data)
                
                aggregated_epe_list.sort()
                aggregated_d1_list.sort()

                aggregated_epe_mean = sum([epe for epe, _ in aggregated_epe_list]) / len(aggregated_epe_list)
                aggregated_d1_mean = sum([d1 for d1, _ in aggregated_d1_list]) / len(aggregated_d1_list)
                if accelerator.is_main_process:
                    print(f"Aggregated epe_list length: {len(aggregated_epe_list)}")
                    print(f"Aggregated d1_list length: {len(aggregated_d1_list)}")
                    print(f"Logging epe_mean: {aggregated_epe_mean}, d1_mean: {aggregated_d1_mean}")
                    with open(os.path.join(epe_metrics_path, f'{epoch}.txt'), 'w') as f:
                        for epe, fp in aggregated_epe_list:
                            f.write(f"{fp} {epe}\n")
                    with open(os.path.join(d1_metrics_path, f'{epoch}.txt'), 'w') as f:
                        for d1, fp in aggregated_d1_list:
                            f.write(f"{fp} {d1}\n")
                    accelerator.log({'val/epe_mean': aggregated_epe_mean,
                                     'val/d1_mean': aggregated_d1_mean}, epoch)
                accelerator.wait_for_everyone()

                best_epe_list = aggregated_epe_list[:num_best_images]
                worst_epe_list = list(reversed(aggregated_epe_list[-num_worst_images:]))
                best_d1_list = aggregated_d1_list[:num_best_images]
                worst_d1_list = list(reversed(aggregated_d1_list[-num_worst_images:]))

                current_file_paths_epe_best = [fp for _, fp in best_epe_list]
                current_file_paths_epe_worst = [fp for _, fp in worst_epe_list]
                current_file_paths_d1_best = [fp for _, fp in best_d1_list]
                current_file_paths_d1_worst = [fp for _, fp in worst_d1_list]

                if epoch == 0:
                    historical_file_paths_epe_best = [fp for _, fp in best_epe_list]
                    historical_file_paths_epe_worst = [fp for _, fp in worst_epe_list]
                    historical_file_paths_d1_best = [fp for _, fp in best_d1_list]
                    historical_file_paths_d1_worst = [fp for _, fp in worst_d1_list]

                paths_to_save = set(current_file_paths_epe_best +
                                        current_file_paths_epe_worst +
                                        current_file_paths_d1_best +
                                        current_file_paths_d1_worst +
                                        historical_file_paths_epe_best +
                                        historical_file_paths_epe_worst +
                                        historical_file_paths_d1_best +
                                        historical_file_paths_d1_worst)
                scores_to_write = []
                lefts_to_write = []
                rights_to_write = []
                disp_preds_to_write = []
                disp_gts_to_write = []
                for data in tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process, desc="Saving images for validation"):
                    (imageL_file, imageR_file, GT_file), left, right, disp_gt, valid = [x for x in data]
                    has_tracking_file = False
                    for image_path in imageL_file:
                        if image_path in paths_to_save:
                            has_tracking_file = True
                            break
                    if not has_tracking_file:
                        continue
                    padder = InputPadder(left.shape, divis_by=32)
                    left = left.to(accelerator.device)
                    right = right.to(accelerator.device)
                    disp_gt = disp_gt.to(accelerator.device)
                    valid = valid.to(accelerator.device)
                    left, right = padder.pad(left, right)
                    with torch.no_grad():
                        disp_pred = model(left, right, iters=cfg.valid_iters, test_mode=True)
                    disp_pred = padder.unpad(disp_pred)
                    assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
                    for b in range(left.shape[0]):
                        if imageL_file[b] not in paths_to_save:
                            continue
                        left_np = left[b].squeeze().cpu().numpy()
                        right_np = right[b].squeeze().cpu().numpy()
                        left_np = (left_np - left_np.min()) / (left_np.max() - left_np.min()) * 255
                        right_np = (right_np - right_np.min()) / (right_np.max() - right_np.min()) * 255
                        disp_pred_np = gray_2_colormap_np(disp_pred[b].squeeze())
                        disp_gt_np = gray_2_colormap_np(disp_gt[b].squeeze())

                        valid_mask = valid[b] >= 0.5
                        if valid_mask.sum() == 0:
                            continue

                        epe_per_pixel = torch.abs(disp_pred[b].squeeze() - disp_gt[b].squeeze())
                        epe_mean = epe_per_pixel[valid_mask].mean().item()
                        d1_mean = (epe_per_pixel > 3.0).float()[valid_mask].mean().item() * 100

                        if imageL_file[b] in historical_file_paths_epe_worst:
                            index = historical_file_paths_epe_worst.index(imageL_file[b])
                            left_name = f"val/epe_worst_{index}_left_historical"
                            right_name = f"val/epe_worst_{index}_right_historical"
                            disp_pred_name = f"val/epe_worst_{index}_disp_pred_historical"
                            disp_gt_name = f"val/epe_worst_{index}_disp_gt_historical"
                            score_name = f"val/epe_worst_{index}_score_historical"
                            score = epe_mean
                            scores_to_write.append((score_name, score))
                            lefts_to_write.append((left_name, left_np))
                            rights_to_write.append((right_name, right_np))
                            disp_preds_to_write.append((disp_pred_name, disp_pred_np))
                            disp_gts_to_write.append((disp_gt_name, disp_gt_np))

                        if imageL_file[b] in historical_file_paths_epe_best:
                            index = historical_file_paths_epe_best.index(imageL_file[b])
                            left_name = f"val/epe_best_{index}_left_historical"
                            right_name = f"val/epe_best_{index}_right_historical"
                            disp_pred_name = f"val/epe_best_{index}_disp_pred_historical"
                            disp_gt_name = f"val/epe_best_{index}_disp_gt_historical"
                            score_name = f"val/epe_best_{index}_score_historical"
                            score = epe_mean
                            scores_to_write.append((score_name, score))
                            lefts_to_write.append((left_name, left_np))
                            rights_to_write.append((right_name, right_np))
                            disp_preds_to_write.append((disp_pred_name, disp_pred_np))
                            disp_gts_to_write.append((disp_gt_name, disp_gt_np))

                        if imageL_file[b] in historical_file_paths_d1_worst:
                            index = historical_file_paths_d1_worst.index(imageL_file[b])
                            left_name = f"val/d1_worst_{index}_left_historical"
                            right_name = f"val/d1_worst_{index}_right_historical"
                            disp_pred_name = f"val/d1_worst_{index}_disp_pred_historical"
                            disp_gt_name = f"val/d1_worst_{index}_disp_gt_historical"
                            score_name = f"val/d1_worst_{index}_score_historical"
                            score = d1_mean
                            scores_to_write.append((score_name, score))
                            lefts_to_write.append((left_name, left_np))
                            rights_to_write.append((right_name, right_np))
                            disp_preds_to_write.append((disp_pred_name, disp_pred_np))
                            disp_gts_to_write.append((disp_gt_name, disp_gt_np))

                        if imageL_file[b] in historical_file_paths_d1_best:
                            index = historical_file_paths_d1_best.index(imageL_file[b])
                            left_name = f"val/d1_best_{index}_left_historical"
                            right_name = f"val/d1_best_{index}_right_historical"
                            disp_pred_name = f"val/d1_best_{index}_disp_pred_historical"
                            disp_gt_name = f"val/d1_best_{index}_disp_gt_historical"
                            score_name = f"val/d1_best_{index}_score_historical"
                            score = d1_mean
                            scores_to_write.append((score_name, score))
                            lefts_to_write.append((left_name, left_np))
                            rights_to_write.append((right_name, right_np))
                            disp_preds_to_write.append((disp_pred_name, disp_pred_np))
                            disp_gts_to_write.append((disp_gt_name, disp_gt_np))

                        if imageL_file[b] in current_file_paths_epe_worst:
                            index = current_file_paths_epe_worst.index(imageL_file[b])
                            left_name = f"val/epe_worst_{index}_left_current"
                            right_name = f"val/epe_worst_{index}_right_current"
                            disp_pred_name = f"val/epe_worst_{index}_disp_pred_current"
                            disp_gt_name = f"val/epe_worst_{index}_disp_gt_current"
                            score_name = f"val/epe_worst_{index}_score_current"
                            score = epe_mean
                            scores_to_write.append((score_name, score))
                            lefts_to_write.append((left_name, left_np))
                            rights_to_write.append((right_name, right_np))
                            disp_preds_to_write.append((disp_pred_name, disp_pred_np))
                            disp_gts_to_write.append((disp_gt_name, disp_gt_np))

                        if imageL_file[b] in current_file_paths_epe_best:
                            index = current_file_paths_epe_best.index(imageL_file[b])
                            left_name = f"val/epe_best_{index}_left_current"
                            right_name = f"val/epe_best_{index}_right_current"
                            disp_pred_name = f"val/epe_best_{index}_disp_pred_current"
                            disp_gt_name = f"val/epe_best_{index}_disp_gt_current"
                            score_name = f"val/epe_best_{index}_score_current"
                            score = epe_mean
                            scores_to_write.append((score_name, score))
                            lefts_to_write.append((left_name, left_np))
                            rights_to_write.append((right_name, right_np))
                            disp_preds_to_write.append((disp_pred_name, disp_pred_np))
                            disp_gts_to_write.append((disp_gt_name, disp_gt_np))

                        if imageL_file[b] in current_file_paths_d1_worst:
                            index = current_file_paths_d1_worst.index(imageL_file[b])
                            left_name = f"val/d1_worst_{index}_left_current"
                            right_name = f"val/d1_worst_{index}_right_current"
                            disp_pred_name = f"val/d1_worst_{index}_disp_pred_current"
                            disp_gt_name = f"val/d1_worst_{index}_disp_gt_current"
                            score_name = f"val/d1_worst_{index}_score_current"
                            score = d1_mean
                            scores_to_write.append((score_name, score))
                            lefts_to_write.append((left_name, left_np))
                            rights_to_write.append((right_name, right_np))
                            disp_preds_to_write.append((disp_pred_name, disp_pred_np))
                            disp_gts_to_write.append((disp_gt_name, disp_gt_np))

                        if imageL_file[b] in current_file_paths_d1_best:
                            index = current_file_paths_d1_best.index(imageL_file[b])
                            left_name = f"val/d1_best_{index}_left_current"
                            right_name = f"val/d1_best_{index}_right_current"
                            disp_pred_name = f"val/d1_best_{index}_disp_pred_current"
                            disp_gt_name = f"val/d1_best_{index}_disp_gt_current"
                            score_name = f"val/d1_best_{index}_score_current"
                            score = d1_mean
                            scores_to_write.append((score_name, score))
                            lefts_to_write.append((left_name, left_np))
                            rights_to_write.append((right_name, right_np))
                            disp_preds_to_write.append((disp_pred_name, disp_pred_np))
                            disp_gts_to_write.append((disp_gt_name, disp_gt_np))
                    # Historical images
                print(f"Saving scores_to_write_{accelerator.process_index}.pkl")
                with open(os.path.join(temp_dir, f'scores_to_write_{accelerator.process_index}.pkl'), 'wb') as f:
                    pickle.dump(scores_to_write, f)
                print(f"Saving lefts_to_write_{accelerator.process_index}.pkl")
                with open(os.path.join(temp_dir, f'lefts_to_write_{accelerator.process_index}.pkl'), 'wb') as f:
                    pickle.dump(lefts_to_write, f)
                print(f"Saving rights_to_write_{accelerator.process_index}.pkl")
                with open(os.path.join(temp_dir, f'rights_to_write_{accelerator.process_index}.pkl'), 'wb') as f:
                    pickle.dump(rights_to_write, f)
                print(f"Saving disp_preds_to_write_{accelerator.process_index}.pkl")
                with open(os.path.join(temp_dir, f'disp_preds_to_write_{accelerator.process_index}.pkl'), 'wb') as f:
                    pickle.dump(disp_preds_to_write, f)
                print(f"Saving disp_gts_to_write_{accelerator.process_index}.pkl")
                with open(os.path.join(temp_dir, f'disp_gts_to_write_{accelerator.process_index}.pkl'), 'wb') as f:
                    pickle.dump(disp_gts_to_write, f)
                print(f"Done saving files_{accelerator.process_index}.pkl")
                accelerator.wait_for_everyone()
                print(f"Done waiting for everyone_{accelerator.process_index}")
                if accelerator.is_main_process:
                    tracker = accelerator.get_tracker('tensorboard')
                    writer = tracker.writer
                    for i in range(accelerator.num_processes):
                        print(f"Loading scores_to_write_{i}.pkl")
                        with open(os.path.join(temp_dir, f'scores_to_write_{i}.pkl'), 'rb') as f:
                            data = pickle.load(f)
                            print(f"Data length: {len(data)}")
                            for (score_name, score) in data:
                                print(f"Logging score_name: {score_name}, score: {score}")
                                accelerator.log({score_name: score}, epoch)
                        print(f"Loading lefts_to_write_{i}.pkl")
                        with open(os.path.join(temp_dir, f'lefts_to_write_{i}.pkl'), 'rb') as f:
                            data = pickle.load(f)
                            print(f"Data length: {len(data)}")
                            for (left_name, left_np) in data:
                                print(f"Logging left_name: {left_name}")
                                writer.add_image(left_name, left_np.astype(np.uint8), epoch, dataformats='CHW')
                        print(f"Loading rights_to_write_{i}.pkl")
                        with open(os.path.join(temp_dir, f'rights_to_write_{i}.pkl'), 'rb') as f:
                            data = pickle.load(f)
                            print(f"Data length: {len(data)}")
                            for (right_name, right_np) in data:
                                print(f"Logging right_name: {right_name}")
                                writer.add_image(right_name, right_np.astype(np.uint8), epoch, dataformats='CHW')
                        print(f"Loading disp_preds_to_write_{i}.pkl")
                        with open(os.path.join(temp_dir, f'disp_preds_to_write_{i}.pkl'), 'rb') as f:
                            data = pickle.load(f)
                            print(f"Data length: {len(data)}")
                            for (disp_pred_name, disp_pred_np) in data:
                                print(f"Logging disp_pred_name: {disp_pred_name}")
                                writer.add_image(disp_pred_name, disp_pred_np, epoch, dataformats='HWC')
                        print(f"Loading disp_gts_to_write_{i}.pkl")
                        with open(os.path.join(temp_dir, f'disp_gts_to_write_{i}.pkl'), 'rb') as f:
                            data = pickle.load(f)
                            print(f"Data length: {len(data)}")
                            for (disp_gt_name, disp_gt_np) in data:
                                print(f"Logging disp_gt_name: {disp_gt_name}")
                                writer.add_image(disp_gt_name, disp_gt_np, epoch, dataformats='HWC')
                
                if accelerator.is_main_process:
                    shutil.rmtree(temp_dir)
                accelerator.wait_for_everyone()
                print(f"Done waiting for everyone_{accelerator.process_index}")
                should_d1_early_stop = d1_early_stopper(aggregated_d1_mean, epoch)
                should_epe_early_stop = epe_early_stopper(aggregated_epe_mean, epoch)
                print(f"Should d1 early stop: {should_d1_early_stop}, Should epe early stop: {should_epe_early_stop} at epoch {epoch} in process {accelerator.process_index}")
                if should_d1_early_stop and should_epe_early_stop:
                    should_keep_training = False
                    print(f"Early stopping at epoch {epoch}")
                    break

            
            if not should_keep_training:
                break

            accelerator.wait_for_everyone()
            active_train_loader = train_loader
            model.train()
            model.module.freeze_bn()

            print(f"Starting training_{accelerator.process_index}")
            for data in tqdm(active_train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process, desc="Training"):
                image_list, left, right, disp_gt, valid = [x for x in data]
                disp_init_pred, disp_preds, depth_mono = model(left, right, iters=cfg.train_iters)
                loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, max_disp=cfg.max_disp)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


                total_step += 1
                loss_val = accelerator.reduce(loss.detach(), reduction='mean')
                metrics = accelerator.reduce(metrics, reduction='mean')
                accelerator.log({'train/loss': loss_val, 'train/learning_rate': optimizer.param_groups[0]['lr']}, total_step)
                accelerator.log(metrics, total_step)

                if (total_step > 0) and (total_step % cfg.save_frequency == 0):
                    if accelerator.is_main_process:
                        save_path = Path(cfg.save_path + f'/{cfg.project_name}_{total_step}.pth')
                        model_save = accelerator.unwrap_model(model)
                        checkpoint = {
                            'model': model_save.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'total_step': total_step,
                            'scheduler': lr_scheduler.state_dict()
                        }
                        torch.save(checkpoint, save_path)
                        del model_save

                if total_step == cfg.total_step:
                    should_keep_training = False
                    break
                
            epoch += 1

        del disp_gt, valid, image_list, left, right, disp_init_pred, disp_preds, depth_mono
        if accelerator.is_main_process:
            save_path = Path(cfg.save_path + f'/{cfg.project_name}_{total_step}.pth')
            model_save = accelerator.unwrap_model(model)
            checkpoint = {
                'model': model_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'total_step': total_step,
                'scheduler': lr_scheduler.state_dict()
            }
            torch.save(checkpoint, save_path)
            del model_save
  
        accelerator.end_training()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    main()
