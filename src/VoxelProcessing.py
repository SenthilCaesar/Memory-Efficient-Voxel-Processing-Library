import numpy as np
import os
from scipy import sparse
from scipy import ndimage

class VoxelProcessing:
    
    def __init__(self, filename, dimension, structure):
        self.x_dim = dimension[0]
        self.y_dim = dimension[1]
        self.z_dim = dimension[2]
        self.struct_element = structure
        self.arr_map = np.load(filename, mmap_mode="r")
        
    def get_CRS_mem_size(self, CRS):
        total = CRS.data.nbytes + CRS.indptr.nbytes + CRS.indices.nbytes
        mem =  total / 1000000000
        print("CRS Memory size = ", mem, "Gb")
        
    def convert_to_2d(self):
        arr_2d = self.arr_map.reshape((self.arr_map.shape[0]*
                             self.arr_map.shape[1]), self.arr_map.shape[2])
        arr_2d = arr_2d.transpose()
        return arr_2d
    
    def compressed_storage(self, arr_2d):
        directory = 'compressed'
        output = 'output'
        CRS = sparse.csr_matrix(arr_2d, dtype='float32')
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(output):
            os.makedirs(output)
        sparse.save_npz("compressed/CRS.npz", CRS)
   
    '''Returns sparse matrix with Non-zero stored 
    elements in Compressed Sparse Row format'''
    
    def load_compressed(self):
        filename = "compressed/CRS.npz"
        try:
            CRS_RAM = open(filename, 'r')
        except:
            print('Cannot open', filename)
            return 0
        else:
            CRS_RAM = sparse.load_npz(filename)
            return CRS_RAM
            
    def get_no_of_blocks(self, arr_2d):
        arr_int = []
        for i in range(1, 11):
            block_size = arr_2d.shape[0] / i
            if float(block_size).is_integer():
                arr_int.append(i)
        block_size = arr_int[-1]
        print("No of blocks = ", block_size)
        print()
        return block_size

    def Morphology(self, CRS, n_blocks, operation):
        start_index = 0
        fake_ghost = 1
        splits = (self.arr_map.shape[0]*
                    self.arr_map.shape[1])//(self.y_dim*10)
        x = CRS.shape[1] // (splits*10)
        end_index = int(CRS.shape[1] / n_blocks) + (x * fake_ghost)
        jump = int(CRS.shape[1] / n_blocks)
        self.f_handle = open('output/binary', 'wb')
        for i in range(0, n_blocks):
            print("Start Index = ",start_index)
            print("End Index = ", end_index)
            block_2d = CRS[:,start_index:end_index].toarray()
            print("Block Shape = ", block_2d.shape)
            # print("Block_2d Memory size = ", getsizeof(block_2d))
            # print("Block_2d Type = ", type(block_2d))
            block_2d = block_2d.T
            start_index = end_index - (x * fake_ghost * 2)
            if i == n_blocks -2:
               end_index = end_index + jump - x
            else:
                end_index = end_index + jump
            self.convert_to_3d(i, block_2d, operation, n_blocks, fake_ghost)
        self.f_handle.close()
            
    def convert_to_3d(self, i, block_2d, operation, n_blocks, fake_ghost):
      
        #n_splits = (self.arr_map.shape[0]*self.arr_map.shape[1])//(self.y_dim*10)
        
        n_splits = self.x_dim // n_blocks

        if i == 0 or i == n_blocks -1:
            n_splits += 1 * fake_ghost
        else:
            n_splits += 2 * fake_ghost
            
        print(n_splits, block_2d.shape)
        mylist = np.split(block_2d, n_splits)
        block_3d = np.dstack(mylist)
        block_3d = np.rollaxis(block_3d,-1)
        # print("Block_3d Memory size = ", getsizeof(block_3d))
        # print("Block_3d Type = ", type(block_3d))
        if operation == 'dilation':
            print("Performing Dilation on block ", i)
            self.block_dilation(block_3d, i, self.struct_element, 
                                n_blocks, n_splits, fake_ghost)              
        if operation == 'erosion':
            print("Performing Erosion on block ", i)
            self.block_erosion(block_3d, self.struct_element)
    
    def block_dilation(self, block_3d, i, struct_element, 
                       n_blocks, n_splits, fake_ghost):

        dilated = ndimage.grey_dilation(block_3d, structure=struct_element)
        trimmed = self.trim_ghostCells(dilated, i, n_blocks, n_splits, fake_ghost)            
        trimmed.tofile(self.f_handle)        
        print("Block Shape = ", trimmed.shape)
        
    def block_erosion(self, block_3d, struct_element):
        eroded = ndimage.grey_erosion(block_3d, structure=struct_element)
        eroded.tofile(self.f_handle)
        print("Block Shape = ", eroded.shape)
        
    def trim_ghostCells(self, block_3d, i, n_blocks, n_splits, fake_ghost):
        if i == 0:
            block_3d = block_3d[0:n_splits-1*fake_ghost,:,:]
        elif i == n_blocks - 1:
            block_3d = block_3d[1*fake_ghost:n_splits,:,:]
        else:
            block_3d = block_3d[1*fake_ghost:n_splits-1*fake_ghost,:,:]
        return block_3d 
        
    def merge_blocks(self):
        filename = "output/binary"
        output_filename = "output/Merged.npy"
        try:
            merging = open(filename, 'r')
        except:
            print('Cannot open', filename)
            return 0
        else:
            merging = np.memmap(filename, dtype=np.float32, 
                                mode='r', shape=(self.x_dim,self.y_dim,self.z_dim))
            
            np.save(output_filename, merging)
            print()
            print("Blocks successfully merged to ", output_filename)