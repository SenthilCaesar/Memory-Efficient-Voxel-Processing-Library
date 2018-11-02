from VoxelProcessing import VoxelProcessing
import numpy as np
import scipy
import datetime

if __name__ == '__main__':

    # Parameters
    #filename = 'gyroidUniform.npy'
    #one = np.random.uniform(size=(2000,8000000))
   # data = scipy.sparse.random(500, 560000, density=0.20, dtype='float64')
   # data = data.toarray()
    #one = data.reshape(500,700,800)
    #np.save('500_700_800_den20.npy', one)
    dimension = [500, 700, 800]
    filename = '500_700_800_den20.npy'
    #a = np.random.uniform(size=(3,9))
    structure = np.ones((3,3,3))
    operation = 'dilation'

    # Main
    start_t = datetime.datetime.now()
    data = VoxelProcessing(filename, dimension, structure)
    arr_2d = data.convert_to_2d()
    no_of_blocks = data.get_no_of_blocks(arr_2d)
    data.compressed_storage(arr_2d)
    CRS = data.load_compressed()
    data.get_CRS_mem_size(CRS)
    data.Morphology(CRS, no_of_blocks, operation)
    data.merge_blocks()
    end_t = datetime.datetime.now()
    total_t = end_t - start_t
    print("Time taken = ", total_t.seconds)