import numpy as np
import h5py
import os
from PIL import Image
from tqdm import tqdm

def convert_zenodo_hdf5_to_tif(file, save_folder):
    """Converts an HDF5 file (containing stacked low- and high-count data) to individual TIF files.

    The HDF5 files for training, validation, and test set can be found at:
    https://zenodo.org/records/8237173

    Parameters
    ----------
    file : str
        HDF5 filename
    save_folder : str
        Folder where to store the TIF files

    Returns
    -------
    Within save_folder two sub-folders "LC" and "HC" are created where
    the individual TIF files are saved in sorted order.
    """
    import time
    st = time.time()
    # Create low- and high-count subfolders
    lc_folder = os.path.join(save_folder, "LC")
    hc_folder = os.path.join(save_folder, "HC")

    if not os.path.exists(lc_folder):
        os.makedirs(lc_folder)

    if not os.path.exists(hc_folder):
        os.makedirs(hc_folder)

    # Load raw data into RAM
    with h5py.File(file, 'r') as fid:
        last = fid['low_count']['data'].shape[-1]-1
    lc_check_path = os.path.join(lc_folder, f'{last:05d}_lc.tif')
    hc_check_path = os.path.join(hc_folder, f'{last:05d}_hc.tif')
    if(os.path.exists(lc_check_path)==False or os.path.exists(hc_check_path)==False):
        print('tif files do not exist, start to generate')
        with h5py.File(file, 'r') as fid:
            lc = fid['low_count']['data'][:,:,:]
            hc = fid['high_count']['data'][:,:,:]
        print(f'data load done, take {time.time()-st:.1}s')
        # Conversion to TIF files
        for i in tqdm(range(lc.shape[-1])):
            Image.fromarray(lc[:,:,i].astype(np.int32)).save(os.path.join(lc_folder, f"{i:05d}_lc.tif"))
            Image.fromarray(hc[:,:,i].astype(np.int32)).save(os.path.join(hc_folder, f"{i:05d}_hc.tif"))
    else:
        print('tif file already exists!')

if __name__ == '__main__':
    file = '../example_data/training_data.hdf5'
    save_folder = '../example_data/training/'
    convert_zenodo_hdf5_to_tif(file, save_folder)

    file = '../example_data/validation_data.hdf5'
    save_folder = '../example_data/validation/'
    convert_zenodo_hdf5_to_tif(file, save_folder)    

