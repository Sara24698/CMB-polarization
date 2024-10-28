# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#from classy import Class
import healpy as hl
import pandas as pd


total_cls = pd.read_csv("./r_0_05.csv", sep = ";")
total_cls=np.array(total_cls)
l=total_cls[:, 0]


tt=total_cls[:, 1]*2*np.pi/(l*(l+1))*7.43
ee=total_cls[:, 2]*2*np.pi/(l*(l+1))*7.43
bb=total_cls[:, 3]*2*np.pi/(l*(l+1))*7.43
te=total_cls[:, 4]*2*np.pi/(l*(l+1))*7.43


cls=[tt, ee, bb, te]
frecuencias_arcmin = np.array([9.66/60/180*np.pi, 7.22/60/180*np.pi, 4.90/60/180*np.pi])
frecuencias_GHz = np.array([100, 143, 217])


	
for i in range(len(frecuencias_arcmin)):
	map = hl.sphtfunc.synfast(cls, nside=2048, pol=True, fwhm=frecuencias_arcmin[i], new=True)
	#hl.write_map('Mapa_total_'+str(frecuencias_GHz[i])+'_GHz'+'.fits',map)
	hl.write_map('Mapa_Q_'+str(frecuencias_GHz[i])+'_GHz'+'.fits',map[1])
	hl.write_map('Mapa_U_'+str(frecuencias_GHz[i])+'_GHz'+'.fits',map[2])




