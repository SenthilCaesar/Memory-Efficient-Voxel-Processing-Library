
from VoxelProcessing import VoxelProcessing
import numpy as np
import scipy
import datetime
import multiprocessing as mp


if __name__ == '__main__':

    # Parameters
    dimension = [400, 400, 400]
    filename = '400_400_400_den10.npy'
    structure = np.ones((3,3,3))
    operation = 'grey_dilation'
    n_block = 400
    fakeGhost = 1

    # Main
    start_t = datetime.datetime.now()
    data = VoxelProcessing(filename, dimension, n_block, structure, fakeGhost)
    arr_2d = data.convert_to_2d()
    data.compressed_storage(arr_2d)
    CRS = data.load_compressed()
    data.get_CRS_mem_size(CRS)
    print()
    
    # Run Without MP module
    # data.Morphology(CRS, operation)
    
    # Run with MP module
    cpu = mp.Pool(processes=7)
    cpu.apply(data.Morphology, args=(CRS, operation))
    
    data.merge_blocks()
    end_t = datetime.datetime.now()
    total_t = end_t - start_t
    print("Time taken = ", total_t.seconds, "sec")
    data.print_memory_usage()
