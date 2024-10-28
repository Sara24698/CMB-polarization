import h5py
import numpy as np


    
validation_file = h5py.File('./Validation_U.h5', 'r')

inputs_validation = validation_file["M"][:,:,:].astype(np.float32)
labels_validation = validation_file["M0"][:,:,:].astype(np.float32)

    
with h5py.File('./Outputs_CENN_U.h5', 'w') as f:       
	f['net'] = inputs_validation
	f['sim'] = labels_validation
