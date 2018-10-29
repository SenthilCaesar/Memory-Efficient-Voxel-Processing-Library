from VoxelProcessing import VoxelProcessing
import numpy as np

if __name__ == '__main__':
    
    # Parameters
    #filename = 'gyroidUniform.npy'    
    #one = np.random.uniform(size=(175,10000))
    #one = one.reshape(175,100,100)
    #np.save('175_100_100.npy', one)
    dimension = [200, 200, 200]
    filename = 'gyroidUniform.npy'
    #a = np.random.uniform(size=(3,9))
    structure = np.ones((3,3,3))
    operation = 'dilation'
    
    # Main
    data = VoxelProcessing(filename, dimension, structure)
    arr_2d = data.convert_to_2d()
    no_of_blocks = data.get_no_of_blocks(arr_2d)
    data.compressed_storage(arr_2d)    
    CRS = data.load_compressed()    
    data.Morphology(CRS, no_of_blocks, operation)
    data.merge_blocks()
    
