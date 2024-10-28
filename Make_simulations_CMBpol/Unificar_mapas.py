import numpy as np
import healpy as hp
import matplotlib.pyplot as plt



def Smooth_FullSkyMap(in_map, out_map, in_fwhm, out_fwhm):
   # Smooth_FullSkyMap("../CMB_maps/Mapa_Q_r0_2_217_GHz.fits","../CMB_maps/Mapa_Q_r0_2_100_GHz", 294.0,579.6)
   

   # fwhm in and out in arcsec

   sigma_smo = np.sqrt((out_fwhm**2.) - (in_fwhm**2.))# fwhm in and out in arcsec
   sigma_smo = (sigma_smo / (3600. * 180.)) * np.pi


   mappa = hp.read_map(in_map[1])
   #smo_map = hp.smoothing(mappa, fwhm = sigma_smo, nest = False)
   #CMB_Smoothed = hp.sphtfunc.smoothing(smo_map, fwhm = 0.00872665)

   

   hp.write_map(out_map+'.fits', mappa, nest = False )
   hp.mollview(mappa)
   plt.savefig(out_map+'.png')
   pass

Smooth_FullSkyMap("./CMB_143GHZ_SimPlanck.fits","CMB_143GHZ_SimPlanck.fits", 433.2, 433.2)
#Smooth_FullSkyMap("./R_0_1/Mapa_total_217_GHz.fits","Mapa_total_143_GHz_R_0_1", 294.0, 433.2)
#Smooth_FullSkyMap("./R_0_05/Mapa_total_217_GHz.fits","Mapa_total_143_GHz_R_0_05", 294.0, 433.2)
#Smooth_FullSkyMap("./R_0_01/Mapa_total_217_GHz.fits","Mapa_total_143_GHz_R_0_01", 294.0, 433.2)
#Smooth_FullSkyMap("./R_0_004/Mapa_total_217_GHz.fits","Mapa_total_143_GHz_R_0_004", 294.0, 433.2)
#Smooth_FullSkyMap("./R_0_001/Mapa_total_217_GHz.fits","Mapa_total_143_GHz_R_0_001", 294.0, 433.2)
#Smooth_FullSkyMap("./R_0_1/Mapa_U_217_GHz.fits","Mapa_U_217_GHz", 294.0,294.0)
#Smooth_FullSkyMap("./R_0_05/Mapa_Q_217_GHz.fits","Mapa_Q_100_GHz", 294.0,579.6)
#Smooth_FullSkyMap("./R_0_05/Mapa_U_217_GHz.fits","Mapa_U_100_GHz", 294.0,579.6)
#Smooth_FullSkyMap("./R_0_2/Mapa_Q_217_GHz.fits","Mapa_Q_143_GHz", 294.0,433.2)
#Smooth_FullSkyMap("./R_0_2/Mapa_U_217_GHz.fits","Mapa_U_143_GHz", 294.0,433.2)


