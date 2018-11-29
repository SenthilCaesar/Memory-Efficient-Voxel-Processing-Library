
from VoxelProcessing import VoxelProcessing
import numpy as np
import scipy
import datetime

if __name__ == '__main__':

    # Parameters
    dimension = [2000, 1500, 1500]
    filename = '2000_1500_1500_den10.npy'
    #a = np.random.uniform(size=(3,9))
    structure = np.ones((3,3,3))
    operation = 'grey_dilation'
    n_block = 2
    fakeGhost = 1

    # Main
    start_t = datetime.datetime.now()
    data = VoxelProcessing(filename, dimension, n_block, structure, fakeGhost)
    arr_2d = data.convert_to_2d()
    data.compressed_storage(arr_2d)
    CRS = data.load_compressed()
    data.get_CRS_mem_size(CRS)
    data.Morphology(CRS, operation)
    data.merge_blocks()
    end_t = datetime.datetime.now()
    total_t = end_t - start_t
    print("Time taken = ", total_t.seconds, "sec")
    data.print_memory_usage()
