import h5py
import numpy as np
import os
from scipy import sparse

def load_sparse_npz(file_path):
    """加载.npz文件中的稀疏矩阵"""
    with np.load(file_path) as data:
        return sparse.csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])

def convert_sparse_to_dense(sparse_matrix):
    """将稀疏矩阵转换为密集矩阵"""
    return sparse_matrix.toarray().reshape(256,256,-1)

def save_to_hdf5(dense_data, dataset_name, output_file):
    """将密集矩阵保存到HDF5文件"""
    with h5py.File(output_file, 'a') as fid:
        grp = fid.create_group(dataset_name)
        dset = grp.create_dataset('data', data=dense_data)

def convert_npz_to_hdf5(npz_file, hdf5_file, dataset_name):
    """将.npz文件中的稀疏矩阵转换为HDF5文件中的密集矩阵"""
    sparse_matrix = load_sparse_npz(npz_file)
    dense_data = convert_sparse_to_dense(sparse_matrix)
    save_to_hdf5(dense_data, dataset_name, hdf5_file)

def compare_datasets(original_file, converted_file, dataset_name):
    """
    比较原始HDF5文件和转换后的HDF5文件中的数据集是否相同。
    
    :param original_file: 原始HDF5文件的路径。
    :param converted_file: 转换后的HDF5文件的路径。
    :param dataset_name: 要比较的数据集名称。
    """
    # 打开原始HDF5文件和转换后的HDF5文件
    with h5py.File(original_file, 'r') as orig_fid, h5py.File(converted_file, 'r') as conv_fid:
        # 加载原始数据集和转换后的数据集
        orig_data = orig_fid[dataset_name]['data'][:]
        conv_data = conv_fid[dataset_name]['data'][:]

        new = orig_data - conv_data
        if(new.max() == new.min() and new.max()==0):
            print('the 2 hdf5 files are the same')
        else:
            print('problem happened, please check the 2 hdf5 files to see if they are the same') 

def convert(input_npz_files, hdf5_output_file):
    datasets = ['high_count', 'low_count']  # 数据集名称列表

    if os.path.exists(hdf5_output_file):
        os.remove(hdf5_output_file)
    for npz_file, dataset in zip(input_npz_files, datasets):
        convert_npz_to_hdf5(npz_file, hdf5_output_file, dataset)
        print(f'Converted {npz_file} to {hdf5_output_file} as {dataset}')

if __name__ == '__main__':
    path = '../example_data'
    train_npz = [f"{path}/training_data_high_count.npz", f'{path}/training_data_low_count.npz']
    train_out = f'{path}/converted_training_data.hdf5'
    val_npz   = [f'{path}/validation_data_high_count.npz', f'{path}/validation_data_low_count.npz']
    val_out   = f'{path}/converted_validation_data.hdf5'
    convert(train_npz, train_out)
    convert(val_npz, val_out)

    datasets = ['high_count', 'low_count'] 
    original_hdf5_file = f'{path}/training_data.hdf5'
    for dataset in datasets:
        compare_datasets(original_hdf5_file, train_out, dataset)    

    original_hdf5_file = f'{path}/validation_data.hdf5'
    for dataset in datasets:
        compare_datasets(original_hdf5_file, val_out, dataset)
