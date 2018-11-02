import numpy as np
from scipy import ndimage
import datetime

filename='500_700_800_den20.npy'
structure_element = np.ones((3,3,3))
merged='output/Merged.npy'

start_t = datetime.datetime.now()
a = np.load(filename)
a_float32 = np.array(a, dtype='float32')

b = ndimage.grey_dilation(a_float32, structure=structure_element)
end_t = datetime.datetime.now()

total_t = end_t - start_t
print("Time taken = ", total_t.seconds)

Merged = np.load(merged)

print((b==Merged).all())
