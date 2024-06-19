from torch.utils.data import DataLoader
import pickle
import numpy as np
import sys
import os
import argparse
import torch


# Argument parser setup
parser = argparse.ArgumentParser(description='Train MobileSAM model with custom settings.')
parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device to use for training (default: cuda)')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training (default: 16)')
parser.add_argument('--selected_bands', type=str, default='RGB', help='RGB or NDWI (RG + NDWI)')
parser.add_argument('--visualise', action='store_true', help='Visualise the tiles and bounding box')
args = parser.parse_args()

# Set device
device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'

# Parse selected bands
if args.selected_bands == 'RGB':
    selected_bands = [1, 2, 3]
elif args.selected_bands == 'NDWI': # R, G, NDWI
    selected_bands = [2, 3, 15]


class SatelliteTileDataset(Dataset):
    def __init__(self, data_dir, tile_list_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.tile_list_file = tile_list_file
        self.tile_list = self.load_tiles()

    def load_tiles(self):
        pkl_file = os.path.join(self.data_dir, self.tile_list_file)
        with open(pkl_file, 'rb') as f:
            all_tiles = pickle.load(f)
        return all_tiles

    def __len__(self):
        return len(self.tile_list)

    def __getitem__(self, idx):
        entry = self.tile_list[idx]
        satellite_tile = entry["satellite_tile"]
        selected_bands_data = satellite_tile[selected_bands, :, :]
        satellite_tile_hw = np.transpose(selected_bands_data, (1, 2, 0)).astype(np.uint8)

        if self.transform:
            satellite_tile_hw = self.transform.apply_image(satellite_tile_hw)

        bounding_box = entry["bounding_box"]
        ground_truth_tile = entry["ground_truth_tile"]

        return satellite_tile_hw, bounding_box, ground_truth_tile
    

def load_node_partition(comm, cfg, dataset):
    """Load a data partition for a given node.
    Data partitions are created if they do not exist.

    Args:
        comm (mpi4py.MPI.Comm): comm object to sync processes
        cfg (dict): config file for data partitioning

    Returns:
        dict: dictionary containing the dataloaders for labeled, unlabeled and test data
        cfg: cfg file updated with num_classes and num_channels fields
    """

    rank = comm.Get_rank()

    # load all the datasets for the given node
    lb_dset, ulb_dset, eval_dset = dset.get_ssl_dset(cfg.num_labels)
    cfg.num_channels = dset.num_channels
    cfg.num_classes = dset.num_classes

    loader_dict = {}
    dset_dict = {"train_lb": lb_dset, "train_ulb": ulb_dset, "eval": eval_dset}

    # dataloader for labeled data
    loader_dict["train_lb"] = get_data_loader(
        dset_dict["train_lb"],
        cfg.batch_size,
        data_sampler="RandomSampler",
        num_iters=cfg.num_train_iter,
        num_workers=1,
        distributed=False,
    )
    # dataloader for unlabeled data
    loader_dict["train_ulb"] = get_data_loader(
        dset_dict["train_ulb"],
        cfg.batch_size * cfg.uratio,
        data_sampler="RandomSampler",
        num_iters=cfg.num_train_iter,
        num_workers=1,
        distributed=False,
    )
    # dataloader for test data
    loader_dict["eval"] = get_data_loader(
        dset_dict["eval"], cfg.eval_batch_size, num_workers=1
    )

    return loader_dict, cfg
