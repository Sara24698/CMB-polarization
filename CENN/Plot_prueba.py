import matplotlib.pyplot as plt
import numpy as np
import h5py
 


train_file_Q = h5py.File('./Validation_Q.h5', 'r')
train_file_U = h5py.File('./Validation_U.h5', 'r')
    

inputs_train_Q = train_file_Q["M"][:,:,:].astype(np.float32)
labels_train_Q = train_file_Q["M0"][:,:,:].astype(np.float32)
inputs_train_U = train_file_U["M"][:,:,:].astype(np.float32)
labels_train_U = train_file_U["M0"][:,:,:].astype(np.float32)

Filtro='Q'


fig, ([ax0, ax1, ax2, ax3]) = plt.subplots(1, 4, figsize=(16,6)) 
fig.tight_layout(pad=4.0)
            
ax0.title.set_text('Input 100 GHz ('+Filtro+')')
ax1.title.set_text('Input 143 GHz ('+Filtro+')')
ax2.title.set_text('Input 217 GHz ('+Filtro+')')
ax3.title.set_text('CMB label ('+Filtro+')')

for i in range(0, 1):

	fig0=ax0.imshow(inputs_train_Q[i,:,:,0])
	fig.colorbar(fig0, ax=ax0, fraction=0.046, pad=0.04)
            
	fig1=ax1.imshow(inputs_train_Q[i,:,:,1])
	fig.colorbar(fig1, ax=ax1, fraction=0.046, pad=0.04)
            
	fig2=ax2.imshow(inputs_train_Q[i,:,:,2])
	fig.colorbar(fig2, ax=ax2, fraction=0.046, pad=0.04)

	fig3=ax3.imshow(labels_train_Q[i,:,:,0])
	fig.colorbar(fig3, ax=ax3, fraction=0.046, pad=0.04)
            
	plt.savefig('Simulacion_'+str(i)+'_'+Filtro+'.pdf')
