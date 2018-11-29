
from VoxelProcessing import VoxelProcessing
import os.path
import scipy
from scipy import sparse
import numpy as np
import datetime
import time
import gc

def runIt(density, dim, blocks, files, structure, fakeGhost, operation):
    for i in range(0, len(density)):
        filename = str(dim[0]) + "_" + str(dim[1]) + "_" + str(dim[2]) + "_den" + str(int(density[i]*100)) + ".npy"
        files.append(filename)
        if not os.path.isfile(filename):
            print('Creating', filename)
            data = scipy.sparse.random(dim[0], dim[1]*dim[2], density=density[i], dtype='float64')
            data = data.toarray()
            array3d = data.reshape(dim[0],dim[1],dim[2])
            np.save(filename, array3d)
            del data, array3d
            time.sleep(120)

    for m in range(0, len(files)):
        for n in range(0, len(blocks)):
            print("---------------------------------------")
            print("Filename = ", files[m])
            start_t = datetime.datetime.now()
            data = VoxelProcessing(files[m], dim, blocks[n], structure, fakeGhost)
            arr_2d = data.convert_to_2d()
            data.compressed_storage(arr_2d)
            CRS = data.load_compressed()
            data.get_CRS_mem_size(CRS)
            data.Morphology(CRS, operation)
            data.merge_blocks()
            end_t = datetime.datetime.now()
            total_t = end_t - start_t
            print("Time taken = ", total_t.seconds)
            data.print_memory_usage()
            del data, CRS, arr_2d
            time.sleep(120)


if __name__ == '__main__':

    density = [0.10]
    dim = [2000,1500,1500]
    blocks =  [2, 4, 5, 8, 10, 100, 200, 400, 500, 1000]
    files = []
    structure = np.ones((3,3,3))
    fakeGhost = 1
    operation = 'grey_dilation'

    runIt(density, dim, blocks, files, structure, fakeGhost, operation)
