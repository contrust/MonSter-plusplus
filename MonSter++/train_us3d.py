import os
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
from accelerate.utils import set_seed
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from core.warp import disp_warp



import matplotlib
import numpy as np
from pathlib import Path
import torch.distributed as dist
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

class EarlyStopper:
    def __init__(self, patience: int, min_delta: float, min_mode: bool) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.patience = patience
        self.min_delta = min_delta
        self.last_scores = []
        self.min_mode = min_mode

    def __call__(self, score: float) -> bool:
        self.last_scores.append(score)
        if len(self.last_scores) <= self.patience:
            return False
        else:
            if self.min_mode:
                should_stop =  self.last_scores[0] < min(self.last_scores[1:]) + self.min_delta
            else:
                should_stop = self.last_scores[0] > max(self.last_scores[1:]) - self.min_delta
            self.last_scores = self.last_scores[1:]
            return should_stop

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
    test_dataset = datasets.US3D(aug_params=None, split='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size,
        pin_memory=True, shuffle=True, num_workers=1, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size,
        pin_memory=True, shuffle=False, num_workers=int(4), drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size,
        pin_memory=True, shuffle=False, num_workers=int(4), drop_last=False)

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
    val_step = cfg.current_val_step
    train_loader, val_loader, test_loader, model, optimizer, lr_scheduler = accelerator.prepare(train_loader, val_loader, test_loader, model, optimizer, lr_scheduler)  #, val_loader  fds, 
    should_keep_training = True
    epoch = 0
    d1_early_stopper = EarlyStopper(patience=cfg.d1_patience, min_delta=cfg.d1_min_delta, min_mode=True)
    epe_early_stopper = EarlyStopper(patience=cfg.epe_patience, min_delta=cfg.epe_min_delta, min_mode=False)
    try:
        while should_keep_training:
            if epoch % cfg.val_frequency == 0:
                torch.cuda.empty_cache()
                model.eval()
                elem_num, total_epe, total_out1, total_out2, total_out3, total_out4, total_out5 = 0, 0, 0, 0, 0, 0, 0
                for data in tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
                    val_step += 1
                    (imageL_file, imageR_file, GT_file), left, right, disp_gt, valid = [x for x in data]
                    padder = InputPadder(left.shape, divis_by=32)
                    left, right = padder.pad(left, right)
                    with torch.no_grad():
                        disp_pred = model(left, right, iters=cfg.valid_iters, test_mode=True)
                    disp_pred = padder.unpad(disp_pred)
                    assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
                    epe = torch.abs(disp_pred - disp_gt)
                    out1 = (epe > 1.0).float()
                    out2 = (epe > 2.0).float()
                    out3 = (epe > 3.0).float()
                    out4 = (epe > 4.0).float()
                    out5 = (epe > 5.0).float()
                    epe = torch.squeeze(epe, dim=1)
                    out1 = torch.squeeze(out1, dim=1)
                    out2 = torch.squeeze(out2, dim=1)
                    out3 = torch.squeeze(out3, dim=1)
                    out4 = torch.squeeze(out4, dim=1)
                    out5 = torch.squeeze(out5, dim=1)
                    epe, out1, out2, out3, out4, out5 = accelerator.gather_for_metrics((epe[valid >= 0.5].mean(),
                        out1[valid >= 0.5].mean(),
                        out2[valid >= 0.5].mean(),
                        out3[valid >= 0.5].mean(),
                        out4[valid >= 0.5].mean(),
                        out5[valid >= 0.5].mean()))
                    elem_num += epe.shape[0]
                    for i in range(epe.shape[0]):
                        total_epe += epe[i]
                        total_out1 += out1[i]
                        total_out2 += out2[i]
                        total_out3 += out3[i]
                        total_out4 += out4[i]
                        total_out5 += out5[i]
                    if val_step % cfg.val_image_frequency == 0:
                        if accelerator.is_main_process:
                            image1_np = left[0].squeeze().cpu().numpy()
                            image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
                            image1_np = image1_np.astype(np.uint8)

                            disp_pred_np = gray_2_colormap_np(disp_pred[0].squeeze())
                            disp_gt_np = gray_2_colormap_np(disp_gt[0].squeeze())

                            tracker = accelerator.get_tracker('tensorboard')
                            writer = tracker.writer

                            writer.add_image("val/left", image1_np, val_step, dataformats='CHW')
                            writer.add_image("val/disp_pred", disp_pred_np, val_step, dataformats='HWC')
                            writer.add_image("val/disp_gt", disp_gt_np, val_step, dataformats='HWC')
                        accelerator.wait_for_everyone()
                    
                epe = total_epe / elem_num
                d1_1px = total_out1 / elem_num
                d1_2px = total_out2 / elem_num
                d1_3px = total_out3 / elem_num
                d1_4px = total_out4 / elem_num
                d1_5px = total_out5 / elem_num

                if accelerator.is_main_process:
                    accelerator.log({'val/epe': epe,
                                     'val/d1_1px': 100 * d1_1px,
                                     'val/d1_2px': 100 * d1_2px,
                                     'val/d1_3px': 100 * d1_3px,
                                     'val/d1_4px': 100 * d1_4px,
                                     'val/d1_5px': 100 * d1_5px}, epoch)
                accelerator.wait_for_everyone()

                if d1_early_stopper(d1_1px) or epe_early_stopper(epe):
                    should_keep_training = False
                    break

            if not should_keep_training:
                print(f"Early stopping at epoch {epoch}")
                break

            active_train_loader = train_loader
            model.train()
            model.module.freeze_bn()

            for data in tqdm(active_train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
                image_list, left, right, disp_gt, valid = [x for x in data]

                with accelerator.autocast():
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


                    
                ####visualize the depth_mono and disp_preds
                if total_step % cfg.train_image_frequency == 0:
                    if accelerator.is_main_process:
                        image1_np = left[0].squeeze().cpu().numpy()
                        image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
                        image1_np = image1_np.astype(np.uint8)

                        depth_mono_np = gray_2_colormap_np(depth_mono[0].squeeze())
                        disp_preds_np = gray_2_colormap_np(disp_preds[-1][0].squeeze())
                        disp_gt_np = gray_2_colormap_np(disp_gt[0].squeeze())
                        tracker = accelerator.get_tracker('tensorboard')
                        writer = tracker.writer
                        writer.add_image("train/left", image1_np, total_step, dataformats='CHW') 
                        writer.add_image("train/disp_pred", disp_preds_np, total_step, dataformats='HWC')
                        writer.add_image("train/disp_gt", disp_gt_np, total_step, dataformats='HWC')
                        writer.add_image("train/depth_mono", depth_mono_np, total_step, dataformats='HWC')
                    accelerator.wait_for_everyone()  

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




