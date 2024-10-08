import logging
from typing import Dict
import numpy as np

from rt_utils import RTStruct


def get_all_roi_names(rtstruct: RTStruct) -> Dict:
    """
    Get all ROI names from RTStruct object
    """
    roi_names = rtstruct.get_roi_names()
    return roi_names

def fetch_all_rois(rtstruct: RTStruct, aux_structures_list_ch = {2:['External', '*Skull']}) -> np.ndarray:
    "use fetch_mapped_rois, but first automatically create a structure_map of all structures present in the rtstruct"
    
    roi_names = rtstruct.get_roi_names()
    # structure_map = {idx: [name] for idx, name in enumerate(roi_names)}
    structure_map = {1:[], **aux_structures_list_ch}
    for name in roi_names:
        # append name to list in structure_map[1]
        structure_map[1].append(name)
        
    return fetch_mapped_rois(rtstruct, structure_map)


def fetch_mapped_rois(rtstruct: RTStruct, structure_map: dict) -> np.ndarray:

    masks = {}
    roi_names = rtstruct.get_roi_names()
    logging.info(f'roi_names = {roi_names}')

    for roi_idx, roi_name in enumerate(roi_names):

        for structure_idx, structures in structure_map.items():

            mask_idx = int(structure_idx)

            if roi_name.lower() not in (_.lower() for _ in structures):
                logging.warning(f"Structure: {roi_name} not in structure map for index {structure_idx}")
                continue

            logging.debug(f"\t-- Converting structure: {roi_name}")

            try:
                mask = rtstruct.get_roi_mask_by_name(roi_name)

                number_voxel = np.count_nonzero(mask)
                logging.debug(f"\tCounted number of voxels: {number_voxel}")

                if number_voxel == 0:
                    continue

                if mask_idx in masks:
                    mask = np.logical_or(mask, masks[mask_idx])

                masks[mask_idx] = mask

                break
            
            except AttributeError as e:
                logging.error(f"AttributeError processing ROI {roi_name}: {e}", exc_info=True)
                break
            except Exception as e:
                logging.error(f"Error processing ROI {roi_name}: {e}", exc_info=True)
                break

    if len(masks) == 0:
        logging.info('found no structure masks for this image')
        return None

    shape = masks[list(masks.keys())[0]].shape
    shape = shape + (len(structure_map) + 1,)
    stacked_mask = np.zeros(shape, dtype=np.uint8)

    for idx, mask in masks.items():
        stacked_mask[:, :, :, idx] = mask.astype(np.uint8)

    # Set background
    background = np.sum(stacked_mask, axis=-1) == 0
    stacked_mask[..., 0] = background

    return stacked_mask


def fetch_rtstruct_roi_masks(
    rtstruct: RTStruct,
    structure_map: dict = None,
) -> np.ndarray:
    """
    Default structure list start from 1
    """

    if structure_map is not None:
        masks = fetch_mapped_rois(rtstruct, structure_map)

    else:
        masks = fetch_all_rois(rtstruct)

    return masks