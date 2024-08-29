import h5py
import sys, os, time
from scipy import sparse
from multiprocessing import Pool

def match(raw, sp):
    maxi = (raw - sp.toarray().reshape(raw.shape)).max()
    mini = (raw - sp.toarray().reshape(raw.shape)).min()
    if(maxi==0 and mini==0):
        return 0
    else:
        return 1

def process_data(inputs, dataset_name):
    st = time.time()
    with h5py.File(inputs, 'r') as fid:
        data = fid[dataset_name]['data'][:,:,:]
        data_sp = sparse.csr_matrix(data.reshape(-1, data.shape[2]))
        print(f'{dataset_name} convert to sparse format doen, time used: {time.time()-st:.1f}s')
        st = time.time()
        if(match(data, data_sp) == 0):            
            sparse.save_npz(f'{inputs[:-5]}_{dataset_name}.npz', data_sp)
    print(f'{dataset_name} save to the disk done, time used: {time.time()-st:.1f}s')

def preprocess(inputs):
    print(inputs)

    datasets = ['low_count', 'high_count']

    #multi process
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.starmap(process_data, [(inputs, dataset) for dataset in datasets])


    #single process    
    #with h5py.File(inputs, 'r') as fid:
    #    lc = fid['low_count']['data'][:,:,:]
    #    hc = fid['high_count']['data'][:,:,:]
    #
    #lc_sp = sparse.csr_matrix(lc.reshape(-1, lc.shape[2]))
    #if(match(lc, lc_sp)==0):
    #    sparse.save_npz(f'{inputs}_lc.npz',lc_sp)
    #
    #hc_sp = sparse.csr_matrix(hc.reshape(-1, hc.shape[2]))
    #if(match(hc, hc_sp)==0):
    #    sparse.save_npz(f'{inputs}_hc.npz',hc_sp)
    
    
    raw_size = os.path.getsize(inputs)/1024/1024 #MB
    sp_size  = (os.path.getsize(f'{inputs[:-5]}_{datasets[0]}.npz') + os.path.getsize(f'{inputs[:-5]}_{datasets[1]}.npz') ) / 1024 / 1024 #MB
    print(f'after the sparse process, the size of the file reduces about {raw_size/sp_size:.1f} times ({raw_size:.0f}MB ===> {sp_size:.0f}MB)')

if __name__ == '__main__':
    preprocess('../example_data/training_data.hdf5')
    preprocess('../example_data/validation_data.hdf5')
