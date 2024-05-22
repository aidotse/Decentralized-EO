import sys
import os
import warnings
from pathlib import Path
import argparse
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import toml
from dotmap import DotMap
from mpi4py import MPI
import pykep as pk

# Import additional necessary modules
from create_plots import create_plots
from init_paseos import init_paseos
from actor_logic import constraint_func, decide_on_activity, perform_activity
from utils import get_savepath_str

# From fine_tune.py
import os
import pickle
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything.utils.transforms import ResizeLongestSide
import sys
sys.path.append(os.path.expanduser('~/Decentralized-EO'))
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append("../")  # needed until paseos is properly installed
import paseos

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

def custom_collate_fn(batch):
    satellite_tiles, bboxes, ground_truth_tiles = zip(*batch)
    satellite_tiles = torch.stack([torch.from_numpy(tile) for tile in satellite_tiles])
    ground_truth_tiles = torch.stack([torch.from_numpy(tile) for tile in ground_truth_tiles])
    return satellite_tiles, list(bboxes), ground_truth_tiles

def initialize_model(device):
    model_type = "vit_t"
    sam_checkpoint = "../weights/mobile_sam.pt"
    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.train()
    return mobile_sam

def main(cfg):
    # Init
    rank = 0  # compute index of this node
    assert (
        cfg.time_per_batch < 30
    ), "For a high time per batch you may miss comms windows?"
    assert cfg.time_for_comms > 0, "Time for comms must be positive"
    time_in_standby = 0
    time_since_last_update = 0
    total_simulation_time = 0
    standby_period = 900  # how long to standby if necessary
    MPI_sync_period = 600  # After how many seconds we wait synchronize instance clocks
    cfg.save_path = get_savepath_str(cfg)

    plot = True
    local_time_at_test = []

    def constraint_function():
        return constraint_func(paseos_instance, groundstations)

    paseos.set_log_level("INFO")
    device = "cuda:" + str(rank) if cfg.cuda and torch.cuda.is_available() else "cpu"

    # Init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    other_ranks = [x for x in range(comm.Get_size()) if x != rank]
    print(f"Started rank {rank}, other ranks are {other_ranks}")
    sys.stdout.flush()

    # Remove any prior lock
    if rank == 0 and os.path.exists(".mpi_lock"):
        print("Removing old lock...")
        os.remove(".mpi_lock")

    if rank == 0 and os.path.exists(cfg.save_path + ".pth.tar"):
        print("Removing old model...")
        os.remove(cfg.save_path + ".pth.tar")

    print("Loading dataset...", flush=True)

    print(f"Rank {rank} - Init training", flush=True)
    sys.stdout.flush()

    # Init paseos
    paseos_instance, local_actor, groundstations = init_paseos(rank, comm.Get_size())
    time_of_last_sync = local_actor.local_time.mjd2000 * pk.DAY2SEC
    print(f"Rank {rank} - Init PASEOS", flush=True)

    if plot and rank == 0:
        plotter = paseos.plot(paseos_instance, paseos.PlotType.SpacePlot)
    
    # Initialize model and data loaders for fine-tuning
    mobile_sam = initialize_model(device)
    transform = ResizeLongestSide(mobile_sam.image_encoder.img_size)

    train_dataset = SatelliteTileDataset(data_dir='../data_preparation/tiles/train', tile_list_file='train_tile_list.pkl')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    optimizer = Adam(mobile_sam.mask_decoder.parameters(), lr=1e-4, weight_decay=0)
    loss_fn = nn.MSELoss()

    num_epochs = 1

    # Simulation loop
    batch_idx = 0
    while total_simulation_time < cfg.simulation_time:

        print(f"Total Simulation Time: {total_simulation_time}", flush = True)
        ################################################################################
        # Sync time between ranks to minimize divergence
        if (
            local_actor.local_time.mjd2000 * pk.DAY2SEC - time_of_last_sync
        ) > MPI_sync_period:
            print(
                f"Rank {rank} waiting for sync at t={local_actor.local_time}",
                end=" ",
                flush=True,
            )
            _ = comm.allreduce(1, op=MPI.SUM)  # send one to indicate still running
            comm.Barrier()
            print(f"Rank {rank} synced.", flush=True)
            time_of_last_sync = local_actor.local_time.mjd2000 * pk.DAY2SEC

        if batch_idx % 100 == 0:
            print(
                f"Rank {rank} - {str(paseos_instance.local_actor.local_time)} - Temperature[C]: "
                + f"{local_actor.temperature_in_K - 273.15:.2f},"
                + f"Battery SoC: {local_actor.state_of_charge:.2f}"
            )
            sys.stdout.flush()
            # print(f"PASEOS advancing time by {cfg.time_per_batch}s.")

        activity, power_consumption, time_in_standby = decide_on_activity(
            paseos_instance,
            cfg.time_per_batch,
            cfg.time_for_comms,
            time_in_standby,
            standby_period,
            time_since_last_update,
        )

        ################################################################################
        # Perform  what was the decided on first in paseos and than on the rank, either
        # A) Exchange model with ground
        # B) Traing model on a batch
        # C) Standby to cool down / recharge
        if activity == "Model_update":
            print(
                f"Rank {rank} will update with GS "
                + str(list(paseos_instance.known_actors.items())[0][0])
                + " at "
                + str(paseos_instance.local_actor.local_time)
            )
            # 1) Model comms in PASEOS (already know there is a window from decide on activity)
            perform_activity(
                activity,
                power_consumption,
                paseos_instance,
                cfg.time_for_comms,
                constraint_function,
            )
            # 2) Evaluate test set before exchanging models
            print(f"Rank {rank} - Pre-aggregation test.")
            #loss, is_best, best_loss = eval_test_set(
            #    rank,
            #    optimizer,
            #    batch_idx,
            #    net,
            #    criterion,
            #    test_losses,
            #    test_dataloader,
            #    local_time_at_test,
            #    paseos_instance,
            #    lr_scheduler,
            #    best_loss,
            #)

            # 3) Exchange models with the ground stations
            #update_central_model(
            #    rank,
            #    device,
            #    batch_idx,
            #    net,
            #    loss,
            #    best_loss,
            #    paseos_instance._state.time,
            #    cfg,
            #)
            time_since_last_update = 0

            # 4) Evaluate test set after exchanging models
            print(f"Rank {rank} - Post-aggregation test.")
            #loss, is_best, best_loss = eval_test_set(
            #    rank,
            #    optimizer,
            #    batch_idx,
            #    net,
            #    criterion,
            #    test_losses,
            #    test_dataloader,
            #    local_time_at_test,
            #    paseos_instance,
            #    lr_scheduler,
            #    best_loss,
            #)
            # Push the time of last step slightly beyond to be distinguishable in plots
            local_time_at_test[-1] += 10
           
        elif activity == "Training":
            # 1) Model training cost in PASEOS
            perform_activity(
                activity,
                power_consumption,
                paseos_instance,
                cfg.time_per_batch,
                constraint_function,
            )
            time_since_last_update += cfg.time_per_batch
            batch_idx += 1            
        
            # Fine-tune the model
            train_dataset.tile_list = train_dataset.load_tiles() 

            # Training loop
            for satellite_tile_hw_batch, bbox_batch, ground_truth_tile_batch in train_loader:
                for j in range(len(satellite_tile_hw_batch)):
                    satellite_tile_hw = satellite_tile_hw_batch[j].numpy()
                    bbox = bbox_batch[j]
                    ground_truth_tile = ground_truth_tile_batch[j].numpy()

                    input_image = transform.apply_image(satellite_tile_hw)            
                    input_image_torch = torch.as_tensor(input_image, device=device).permute(2, 0, 1).contiguous().unsqueeze(0)
                    input_image = mobile_sam.preprocess(input_image_torch)
                    original_image_size = satellite_tile_hw.shape[:2]
                    input_size = tuple(input_image_torch.shape[2:4])

                    bbox_np = np.array(bbox)
                    if bbox_np.size == 4:
                        bbox_np = bbox_np.reshape(1, 4)

                    box = transform.apply_boxes(bbox_np, original_image_size)
                    box_torch = torch.tensor(box, dtype=torch.float, device=device).unsqueeze(0)

                    with torch.no_grad():
                        image_embedding = mobile_sam.image_encoder(input_image)
                        sparse_embeddings, dense_embeddings = mobile_sam.prompt_encoder(
                            points=None,
                            boxes=box_torch,
                            masks=None,
                        )

                    low_res_masks, _ = mobile_sam.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=mobile_sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                    )

                    upscaled_masks = mobile_sam.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
                    binary_mask = torch.sigmoid(upscaled_masks)

                    gt_mask_resized = torch.from_numpy(np.resize(ground_truth_tile, (1, 1, ground_truth_tile.shape[0], ground_truth_tile.shape[1]))).to(device)
                    gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32, device=device)

                    loss = loss_fn(binary_mask, gt_binary_mask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
    
            print(f'EPOCH: {num_epochs}, Mean training loss: {loss.item()}')

        else:
            # 1) Model standby in PASEOS and do nothing :)
            perform_activity(
                activity,
                power_consumption,
                paseos_instance,
                cfg.time_for_comms,
                constraint_function,
            )
            time_since_last_update += cfg.time_per_batch
            print(
                f"Rank {rank} standing by - Temperature[C]: "
                + f"{local_actor.temperature_in_K - 273.15:.2f},"
                + f"Battery SoC: {local_actor.state_of_charge:.2f}"
            )
        
        if plot and batch_idx % 10 == 0 and rank == 0:
            plotter.update(paseos_instance)

        total_simulation_time = (
            paseos_instance._state.time - paseos_instance._cfg.sim.start_time
        )        

    Path(cfg.save_path + "/").mkdir(parents=True, exist_ok=True)
    paseos_instance.save_status_log_csv(cfg.save_path + "/" + str(rank) + ".csv")
    create_plots(paseos_instances=[paseos_instance], cfg=cfg, rank=rank)  

    toml.dump(cfg, open(cfg.save_path + "/cfg.toml", "w"))

    print(f"Rank {rank} waiting to finish.")

    # Send 0 as sign that we are finished
    # Wait until all ranks are finished
    while comm.allreduce(0, op=MPI.SUM) > 0:
        print(f"Rank {rank} standing by...")
        comm.Barrier()

    print(f"Rank {rank} finished.")  

        #total_simulation_time += cfg.time_per_batch

if __name__ == "__main__":
    if len(sys.argv) < 2:
        warnings.warn("Please pass the path to a cfg file. Using default cfg")
        #path = "../cfg/simulation_without_training_cfg.toml"
        path = "../cfg/default_cfg.toml"
    else:
        path = sys.argv[1]
    if not os.path.exists(path):
        raise Exception(f"No cfg file found at {path}.")
    print(f"Loading cfg from {path}")
    with open(path) as cfg:
        # dynamic=False inhibits automatic generation of non-existing keys
        cfg = DotMap(toml.load(cfg), _dynamic=False)
    print(cfg)
    main(cfg)
