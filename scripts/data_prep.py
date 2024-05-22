""" 
PART 1a): Data download and pre-processing (train and val files)

Description: This script downloads and processes Worldfloods data from Hugging Face.
- Loads files from the train and val folders
- Generates tiles of 256x256
- Removes any tiles containing clouds
- Removes any tiles that do not contain water
- Binarises the ground truth masks
- Calculates NDWI in the satellite data and adds this as an additional band
- Normalises the satellite data, using the pixel statistics
- Extracts bounding box coordinates of water bodies
- Saves a specified number of satellite and GT tile pairs, and their bounding boxes, in a dictionary
- Saves a specified number of satellite and GT tile pairs on disk
"""

import os
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.measure import label, regionprops
from huggingface_hub import list_repo_files, hf_hub_download, scan_cache_dir
import pickle
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

# Command-line arguments
parser = argparse.ArgumentParser(description="Process WorldFloods data")
parser.add_argument('--dst_path', type=str, default='../data_preparation/tiles', help='Root directory to save processed tiles')
parser.add_argument('--tile_size', type=int, default=256, help='Size of each tile (pixels)')
parser.add_argument('--dataset', type=str, default='train', help='Dataset to process from Hugging Face: train or val')
parser.add_argument('--sample_size', type=str, default=1, help='Maximum number of tiles to extract from each input satellite image')
parser.add_argument('--visualise', action='store_true', help='Visualise the tiles and bounding box')

args = parser.parse_args()

tile_list = []

def download_and_process_data(dataset_type: str):
    repo_id = "isp-uv-es/WorldFloodsv2"
    file_list = list_repo_files(repo_id, repo_type="dataset")
    
    path_pairs = {}
    
    for file in file_list:
        if file.startswith(f"{dataset_type}/gt/") and file.endswith('.tif'):
            base_name = os.path.basename(file)
            s2_equivalent = file.replace("/gt/", "/S2/")
            path_pairs[base_name] = {'gt': file, 's2': s2_equivalent}
            if len(path_pairs) == 5:  # Limit to the first 5 pairs
                break
    
    # Information: count the number of GT and S2 files
    gt_count = len({paths['gt'] for paths in path_pairs.values() if 'gt' in paths})
    s2_count = len({paths['s2'] for paths in path_pairs.values() if 's2' in paths})
    print(f"Number of GT files: {gt_count}")
    print(f"Number of S2 files: {s2_count}")

    # Download the files from Hugging Face to dst_path
    for key, paths in path_pairs.items():
        gt_path = hf_hub_download(repo_id=repo_id, filename=paths['gt'], repo_type="dataset")
        s2_path = hf_hub_download(repo_id=repo_id, filename=paths['s2'], repo_type="dataset")
        process_tile_data(s2_path, gt_path, os.path.join(args.dst_path, dataset_type))
        # The total volume of WorldFloods data is >300 GB. 
        # Delete the downloaded files after processing...
        os.remove(gt_path)
        os.remove(s2_path)
        clear_huggingface_cache()

def clear_huggingface_cache():
    # Clear Hugging Face cache after processing
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if os.path.exists(cache_dir):
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                os.remove(os.path.join(root, file))

def process_tile_data(s2_path, gt_path, dst_path):
    base_filename = os.path.splitext(os.path.basename(gt_path))[0]
    
    skipped_tiles = 0
    processed_tiles = 0
    saved_tiles = 0

    with rasterio.open(s2_path) as s2_dataset, rasterio.open(gt_path) as gt_dataset:
        tiles_across = gt_dataset.width // args.tile_size
        tiles_down = gt_dataset.height // args.tile_size

        for y in range(tiles_down):
            for x in range(tiles_across):
                tile_name = f"Tile_{base_filename}_{x}_{y}.tif"
                window = Window(x * args.tile_size, y * args.tile_size, args.tile_size, args.tile_size)
                gt_tile_band1 = gt_dataset.read(1, window=window) # Band 1 contains values of 2 for cloud
                gt_tile_band2 = gt_dataset.read(2, window=window) # Band 2 contains values of 2 for water

                # Combined check: Skip tiles which DO contain cloud or DO NOT contain water
                if np.any(gt_tile_band1 == 2) or not np.any(gt_tile_band2 == 2):
                    skipped_tiles += 1
                    continue

                ## Extract bounding boxes
                # Convert the GT tile to binary (With a value of 1 for water and 0 for everything else)
                binary_gt_tile = np.where(gt_tile_band2 == 2, 1, 0)
                label_image = label(binary_gt_tile)
                regions = list(regionprops(label_image))
                # Filter out regions with bounding boxes smaller than 20x20 pixels
                filtered_regions = [region for region in regions if (region.bbox[2] - region.bbox[0] >= 20) and (region.bbox[3] - region.bbox[1] >= 20)]
                # Only use tiles which contain exactly one bounding box (one water body)
                if len(filtered_regions) == 1:
                    sample_size = args.sample_size
                    if saved_tiles < sample_size: # Saving only X tiles from each input satellite image (to ensure the training dataset is sufficiently small to load on the satellite)
                        region = filtered_regions[0]
                        minr, minc, maxr, maxc = region.bbox
                        bbox_coords = (minc, minr, maxc, maxr)

                        ## Satellite data processing
                        satellite_tile = s2_dataset.read(window=window)
                        green_band = satellite_tile[2]
                        nir_band = satellite_tile[7]
                        ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-10)

                        print("Shape of satellite data:", satellite_tile.shape)

                        # Add NDWI as an additional band
                        satellite_tile = np.vstack([satellite_tile, ndwi[np.newaxis, :, :]])

                        # Normalise bands
                        rgb_indices = [1, 2, 3, 15]  # Correct indices for bands 2 (Blue), 3 (Green), and 4 (Red)
                        normalized_bands = np.empty((len(rgb_indices), tile_size, tile_size), dtype=np.uint8)
                        for i, band_index in enumerate(rgb_indices):
                            band = satellite_tile[band_index, :, :]  # Access the band using the correct zero-based index
                            p2, p98 = np.percentile(band, (2, 98))   # Compute the 2nd and 98th percentiles
                            normalized_band = np.clip((band - p2) / (p98 - p2), 0, 1) * 255  # Normalize and scale to 0-255
                            normalized_bands[i] = normalized_band.astype(np.uint8)  # Convert to uint8 for image processing

                        # Update the original satellite tile with normalized RGB bands
                        satellite_tile[1] = normalized_bands[0]  # Update Blue band
                        satellite_tile[2] = normalized_bands[1]  # Update Green band
                        satellite_tile[3] = normalized_bands[2]  # Update Red band
                        satellite_tile[15] = normalized_bands[3] # Update NDWI band

                        rgb_image = np.transpose(normalized_bands[:3], (1, 2, 0))
                        test_image_array_hw = np.transpose(normalized_bands, (1, 2, 0))

                        # Save the S2 and GT tiles to dst_path
                        s2_tile_filename = os.path.join(dst_path, f"S2_tile_{base_filename}_{x}_{y}.tif")
                        gt_tile_filename = os.path.join(dst_path, f"GT_tile_{base_filename}_{x}_{y}.tif")
                        
                        print("Shape of satellite data:", satellite_tile.shape)
                        with rasterio.open(s2_tile_filename, 'w', driver='GTiff', 
                                        height=satellite_tile.shape[1], width=satellite_tile.shape[2],
                                        count=satellite_tile.shape[0], dtype=np.uint8) as dst:
                            for i in range(satellite_tile.shape[0]):
                                dst.write(satellite_tile[i, :, :], i + 1)

                        with rasterio.open(gt_tile_filename, 'w', driver='GTiff', 
                                        height=gt_tile_band2.shape[0], width=gt_tile_band2.shape[1],
                                        count=1, dtype=np.uint8) as dst:
                            dst.write(gt_tile_band2.astype(np.uint8), 1)

                        # Append the tile combination and bounding box coordinates to the tile_list dictionary
                        tile_list.append({
                            "satellite_tile": satellite_tile,
                            "ground_truth_tile": binary_gt_tile,
                            "bounding_box": bbox_coords,
                            "bbox_index": region.label
                        })

                        processed_tiles += 1
                        saved_tiles += 1

                        if args.visualise:
                            # Save the visualization
                            visualization_filename = os.path.join(dst_path, f"Visualization_{base_filename}_{x}_{y}.png")
                            save_tile_visualization(rgb_image, binary_gt_tile, bbox_coords, visualization_filename)

    # Print summary statistics after all tiles have been checked
    print(f"Total processed tiles for {base_filename}: {processed_tiles}")
    print(f"Total skipped tiles {base_filename}: {skipped_tiles}")

def save_tile_visualization(rgb_image, binary_gt_tile, bbox_coords, visualization_filename):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    # Display the satellite image tile with a bounding box
    ax[0].imshow(rgb_image)
    ax[0].set_title("Satellite Tile")
    minc, minr, maxc, maxr = bbox_coords
    rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr, linewidth=1, edgecolor='red', facecolor='none')
    ax[0].add_patch(rect)
    ax[0].axis('off')

    ax[1].imshow(binary_gt_tile)
    ax[1].set_title("Ground Truth Tile")
    rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr, linewidth=1, edgecolor='red', facecolor='none')
    ax[1].add_patch(rect)
    ax[1].axis('off')

    plt.savefig(visualization_filename)
    plt.close()

if __name__ == "__main__":
    tile_size = args.tile_size
    dataset_subdir = os.path.join(args.dst_path, args.dataset)
    os.makedirs(dataset_subdir, exist_ok=True)
    download_and_process_data(args.dataset)

    # Save the tile_list dictionary
    with open(os.path.join(dataset_subdir, f"{args.dataset}_tile_list.pkl"), 'wb') as f:
        pickle.dump(tile_list, f)
