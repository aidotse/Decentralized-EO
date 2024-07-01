import torch
import random

import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.zoo import image_models
#from compressai.datasets import ImageFolder
#from compressai.losses import RateDistortionLoss

from utils import AverageMeter, configure_optimizers
from raw_image_folder import RawImageFolder
from model_utils import get_model

import os
from segment_anything.utils.transforms import ResizeLongestSide
import sys
sys.path.append(os.path.expanduser('~/Decentralized-EO'))
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch.nn as nn
import numpy as np

from mobile_sam import (
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    build_sam_vit_t,
    sam_model_registry,
    SamAutomaticMaskGenerator,
    SamPredictor
)

import toml
from glob import glob

print(f"CUDA available: {torch.cuda.is_available()}")

# Define the image_models dictionary using build functions
image_models.update({
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_t": build_sam_vit_t,
})

# Function to calculate IoU
def calculate_iou(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum((1, 2, 3))
    union = (pred + target - pred * target).sum((1, 2, 3))
    iou = intersection / union
    return iou.mean().item()

def init_training(cfg, rank):
    from test_main import SatelliteTileDataset, custom_collate_fn

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)  
        torch.cuda.manual_seed(cfg.seed)  
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda:" + str(rank) if cfg.cuda and torch.cuda.is_available() else "cpu" 
    print("Using:", device)

    # Initialize model
    model_type = cfg.model  # Use model type from configuration
    sam_checkpoint = "../weights/mobile_sam.pt"
    mobile_sam = image_models[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)

    # Prep data for ingestion by model encoder
    transform = ResizeLongestSide(mobile_sam.image_encoder.img_size)

    # Load *.pkl files for training from specific rank directory
    train_data_dir = os.path.join(cfg.dataset, 'new_rank_' + str(rank))
    train_pkl_files = glob(os.path.join(train_data_dir, '*.pkl'))
    train_tile_list_file = os.path.basename(train_pkl_files[0])

    # Load train dataset
    train_dataset = SatelliteTileDataset(data_dir=train_data_dir, tile_list_file=train_tile_list_file, transform=transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda:" + str(rank)),
        pin_memory_device=device,
        collate_fn=custom_collate_fn
    )
    train_dataloader_iter = iter(train_dataloader)

   # Load validation/test .pkl file
    test_data_dir = os.path.join(cfg.dataset, 'new_test')
    test_pkl_files = glob(os.path.join(test_data_dir, '*.pkl'))
    test_tile_list_file = os.path.basename(test_pkl_files[0])

    test_dataset = SatelliteTileDataset(data_dir=test_data_dir, tile_list_file=test_tile_list_file, transform=transform)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda:" + str(rank)),
        pin_memory_device=device,
        collate_fn=custom_collate_fn
    )

    optimizer, aux_optimizer = configure_optimizers(mobile_sam, cfg)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = nn.MSELoss()

    last_epoch = 0
    if cfg.checkpoint:  # load from previous checkpoint
        print("Loading", cfg.checkpoint)
        checkpoint = torch.load(cfg.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        mobile_sam.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    transform = ResizeLongestSide(mobile_sam.image_encoder.img_size)

    return (
        mobile_sam,
        optimizer,
        aux_optimizer,
        criterion,
        train_dataloader,
        test_dataloader, # placeholder for val_dataloader
        lr_scheduler,
        last_epoch,
        train_dataloader_iter,
        transform
    )

def configure_optimizers(model, cfg):
    optimizer_params = [
        {"params": [p for p in model.parameters() if p.requires_grad]},
    ]

    ## Debugging statement to check optimizer parameters
    #print(f"Optimizer parameters: {optimizer_params}")

    optimizer = torch.optim.Adam(optimizer_params, lr=cfg.lr)
    aux_optimizer = torch.optim.Adam(optimizer_params, lr=cfg.aux_lr)
    return optimizer, aux_optimizer


def dataloader_manager(device, model, transform, satellite_tile_batch, bbox_batch, ground_truth_tile_batch, original_image_size_batch):
    for j in range(len(satellite_tile_batch)):
        satellite_tile_hw = satellite_tile_batch[j].numpy()
        bbox = bbox_batch[j]
        ground_truth_tile = ground_truth_tile_batch[j].numpy()
        original_image_size = original_image_size_batch[j]

        input_image_torch = torch.as_tensor(satellite_tile_hw, device=device).permute(2, 0, 1).contiguous().unsqueeze(0)        
        input_image = model.preprocess(input_image_torch)
        input_size = tuple(input_image_torch.shape[2:4])

        with torch.no_grad():
            image_embedding = model.image_encoder(input_image)

            bbox_np = np.array(bbox).reshape(1, 4)
            box = transform.apply_boxes(bbox_np, original_image_size)
            box_torch = torch.tensor(box, dtype=torch.float, device=device).unsqueeze(0)

            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            torch.cuda.empty_cache()

        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        binary_mask = torch.sigmoid(upscaled_masks)

        gt_mask_resized = torch.from_numpy(np.resize(ground_truth_tile, (1, 1, ground_truth_tile.shape[0], ground_truth_tile.shape[1]))).to(device)
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32, device=device)

        return binary_mask, gt_binary_mask


def train_one_batch(
    rank,
    model,
    criterion,
    train_dataloader,
    train_dataloader_iter: DataLoader,
    optimizer,
    aux_optimizer,
    batch_idx,
    clip_max_norm,
    transform
):
    """Trains the model on one batch

    Args:
        rank (int): Rank index
        model (torch.model): model to train
        criterion (): Loss criterion, see compressai docs
        train_dataloader (torch.dataloader): loader for training data
        train_dataloader_iter (iterator): current iterator on the loader
        optimizer (torch.optimizer): optimizer for gradients
        aux_optimizer (torch.optimizer): auxiliary loss optimizer
        batch_idx (int): index of current batch
        clip_max_norm (): gradient clipping thingy
        transform: Transformation function
        device: Device to use for training

    Returns:
        iterator: the updated training data loader
    """
    
    print("Training one batch...")

    model.train()
    device = next(model.parameters()).device

    try:
        satellite_tile_batch, bbox_batch, ground_truth_tile_batch, original_image_size_batch = next(train_dataloader_iter)

    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        train_dataloader_iter = iter(train_dataloader)
        satellite_tile_batch, bbox_batch, ground_truth_tile_batch, original_image_size_batch = next(train_dataloader_iter)

    # Get model prediction and ground truth
    binary_mask, gt_binary_mask = dataloader_manager(device, model, transform, satellite_tile_batch, bbox_batch, ground_truth_tile_batch, original_image_size_batch)
    binary_mask = binary_mask.to(device)
    gt_binary_mask = gt_binary_mask.to(device)

    # Calculate loss and update model parmaters
    loss = criterion(binary_mask, gt_binary_mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % 100 == 0:
        print(f"Rank {rank} - Training batch {batch_idx}: Loss: {loss.item():.3f}")

    torch.cuda.empty_cache()

    return train_dataloader_iter, loss.item()


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    """Train the mode for one epoch.

    Args:
        model (torch.model): model to train
        criterion (): Loss criteration, see compressai docs
        train_dataloader (torch.dataloader): loader for training data
        optimizer (torch.optimizer): optimizer for gradients
        aux_optimizer (torch.optimizer): auxiliary loss optimizer
        epoch (int): index of current epoch
        clip_max_norm (): gradient clipping thingy
    """
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def test_epoch(rank, epoch, test_dataloader, model, criterion, transform):
    """Test the model

    Args:
        rank (int): index of the rank
        epoch (int): index of current epoch
        test_dataloader (torch.dataloader): loader for test data
        model (torch.model): model to test
        criterion (): Loss criteration, see compressai docs

    Returns:
        float: avg loss
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():

        # Get model prediction and ground truth
        satellite_tile_batch, bbox_batch, ground_truth_tile_batch, original_image_size_batch = next(iter(test_dataloader))
        binary_mask, gt_binary_mask = dataloader_manager(device, model, transform, satellite_tile_batch, ground_truth_tile_batch)
        binary_mask = binary_mask.to(device)
        gt_binary_mask = gt_binary_mask.to(device)

        # Calculate loss and update model parmaters
        loss = criterion(binary_mask, gt_binary_mask)

    print(
        f"Rank {rank} - "
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss:.3f} |\n"
    )

    return loss


def eval_test_set(
    rank,
    optimizer,
    batch_idx,
    net,
    criterion,
    test_losses,
    test_dataloader,
    local_time_at_test,
    paseos_instance,
    lr_scheduler,
    best_loss,
    transform
):
    """Evaluate the set

    Args:
        rank (int): Rank index
        optimizer (torch.optimizer): optimizer for gradients
        batch_idx (int): index of current batch
        net (torch.model): model to train
        criterion (): Loss criteration, see compressai docs
        test_losses (_type_): _description_
        teset_dataloader (torch.dataloader): loader for testing data
        local_time_at_test (float): local paseos time at testing
        paseos_instance (paseos): paseos instance of the rank
        lr_scheduler (torch.scheduler): lr scheduler
        best_loss (float): best achieved loss

    Returns:
        tuple: loss, whether best, best loss achieved
    """
    # print(f"Rank {rank} - Evaluating test set")
    # print(f"Rank {rank} - Previous learning rate: {optimizer.param_groups[0]['lr']}")
    # start = time.time()
    loss = test_epoch(rank, batch_idx, test_dataloader, net, criterion, transform)
    test_losses.append(loss.item())
    local_time_at_test.append(paseos_instance._state.time)
    # print(f"Rank {rank} - Test evaluation took {time.time() - start}s")

    lr_scheduler.step(loss)
    # print(f"Rank {rank} - New learning rate: {optimizer.param_groups[0]['lr']}")

    is_best = loss < best_loss
    best_loss = min(loss, best_loss)
    torch.cuda.empty_cache()
    return loss, is_best, best_loss
