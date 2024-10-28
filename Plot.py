from classy import Class
import numpy as np
import matplotlib.pyplot as plt
import healpy as hl
import pandas as pd

"""
total_cls = pd.read_csv("./r_0_2.csv", sep = ";")
total_cls=np.array(total_cls)
l_camb=total_cls[:, 0]


tt=total_cls[:, 1]#*2*np.pi/(l_camb*(l_camb+1))
ee=total_cls[:, 2]#*2*np.pi/(l_camb*(l_camb+1))
bb=total_cls[:, 3]#*2*np.pi/(l_camb*(l_camb+1))
te=total_cls[:, 4]#*2*np.pi/(l_camb*(l_camb+1))


cls_camb=[tt, ee, bb, te]


cosmo = Class()
cosmo.set({'output':'tCl ,pCl ,lCl','lensing':'yes','modes':'s,t','r':'0.2', 'l_max_scalars': '7000', 'l_max_tensors': '7000', 'h':'67.33', 'omega_cdm': '0.119', 'omega_b': '0.02226',  'Omega_fld': '0.723926', 'N_ur': '2.046', 'N_ncdm':'1', 'omega_ncdm': '0.0006451439', 'A_s': '2.1e-09', 'n_s': '0.9667'})
cosmo.compute()

l_class = np.array(range(0 ,6144))
factor = l_class*(l_class+1) /(2* np.pi)
cl = cosmo.lensed_cl(6143)

cls_class = [cl['tt']*7.43e12, cl['ee']*7.43e12, cl['bb']*7.43e12, cl['te']*7.43e12]
"""

inputmap = hl.fitsfunc.read_map('./Planck_map.fits', field=0, memmap = True)










plt.loglog(l_camb, cls_camb[0], label='tt_camb')
plt.loglog(l_camb, cls_camb[1], label='ee_camb')
plt.loglog(l_camb, cls_camb[2], label='bb_camb')
#plt.loglog(l_class, cls_class[0], label='tt_class')
#plt.loglog(l_class, cls_class[1], label='ee_class')
#plt.loglog(l_class, cls_class[2], label='bb_class')
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell(\ell +1) /(2\ pi) C_l^{BB}$")
plt.tight_layout()
plt.legend()
plt.savefig("Cls.png")
