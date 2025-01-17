#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# from InputSimulation import reading_the_data
# from Simulation import Simulation

import h5py

# Import the NaMaster python wrapper
import pymaster as nmt

# Simulation_Parameters = reading_the_data()

Images_Directory = './figures/draft/'


data_Q = ['./Outputs_CENN_Q.h5', './Outputs_CENN_simu_Q.h5']
data_U = ['./Outputs_CENN_U.h5', './Outputs_CENN_simu_U.h5']

color_Input = ['tab:green']
color_Output = ['tab:red']
color_Error = ['black']
color_Residuals = ['black']

label_Input = ['Input Planck']
label_Output = ['Output']
label_Error = ['r = 0.2']
label_Residuals = ['R = 0.2']

lmax = [600]
aposcale = [0.8]


# Comparison trained at 25-trained at 30 and tested at 25

# data_Q = ['./Output/25arcmin/Output_Q_Train_30_Test25.h5', './Output/25arcmin/Output_Q_Train_25_Test25.h5']
# data_U = ['./Output/25arcmin/Output_U_Train_30_Test25.h5', './Output/25arcmin/Output_U_Train_25_Test25.h5']

# color_Input = ['tab:blue', 'tab:green']
# color_Output = ['tab:red', 'tab:orange']
# color_Error = ['black', 'tab:brown']
# color_Residuals = ['black', 'tab:brown']

# label_Input = ['Input-30 arcmin', 'Input-25 arcmin']
# label_Output = ['Recovered-Trained 30', 'Recovered-Trained 25']
# label_Error = ['Trained 30', 'Trained 25']
# label_Residuals = ['Residuals-Trained 30', 'Residuals-Trained 25']

# lmax = [700, 700]
# aposcale = [1., 1.]

# Comparison 5 arcmin d1s1 d2s2

# data_Q = ['./Output/5arcmin/Outputs_CENN_Q_5arcmin.h5', './Output/5arcmin/Outputs_CENN_Q_5arcmin_d2s2.h5']
# data_U = ['./Output/5arcmin/Outputs_CENN_U_5arcmin.h5', './Output/5arcmin/Outputs_CENN_U_5arcmin_d2s2.h5']

# color_Input = ['tab:blue', 'tab:green']
# color_Output = ['tab:red', 'tab:orange']
# color_Error = ['black', 'tab:brown']
# color_Residuals = ['black', 'tab:brown']

# label_Input = ['Input-train fg. model', 'Input-d4s2']
# label_Output = ['Recovered-d6s1', 'Recovered-d4s2']
# label_Error = ['train fg. model', 'd4s2']
# label_Residuals = ['Residuals-d6s1', 'Residuals-d4s2']

# lmax = [1500, 1500]
# aposcale = [1.5, 1.5]

# Comparison trained at 20-25-30 and tested at 20

# data_Q = ['./Output/20arcmin/Output_Q_Train_30_Test20.h5', './Output/20arcmin/Output_Q_Train_25_Test20.h5', './Output/20arcmin/Output_Q_Train_20_Test20.h5']
# data_U = ['./Output/20arcmin/Output_U_Train_30_Test20.h5', './Output/20arcmin/Output_U_Train_25_Test20.h5', './Output/20arcmin/Output_U_Train_20_Test20.h5']

# color_Input = ['tab:blue', 'tab:green']
# color_Output = ['tab:red', 'tab:orange', 'magenta']
# color_Error = ['black', 'tab:brown', 'yellow']
# color_Residuals = ['black', 'tab:brown', 'yellow']

# label_Input = ['Input-20 arcmin', 'Input-20 arcmin']
# label_Output = ['Recovered-Trained 30', 'Recovered-Trained 25', 'Recovered-Trained 20']
# label_Error = ['Trained 30', 'Trained 25', 'Trained 20']
# label_Residuals = ['Residuals-Trained 30', 'Residuals-Trained 25', 'Residuals-Trained 20']

# lmax = [800, 800, 800]
# aposcale = [1.2, 1.2, 1.2]

class Read():
    
    def reading_noise_patches():
        
        Number_Of_Initial_Pixels = 256
        Number_Of_Cutted_Pixels = 8
        Number_Of_Pixels = 240
        
        data_Q = ['./Parches_ruido_Q.h5']
        data_U = ['./Parches_ruido_U.h5']
        
        data_Q = h5py.File(data_Q[0], 'r')
        data_U = h5py.File(data_U[0], 'r')
                
        noise_Q= data_Q['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        noise_U = data_U['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        noise_Q = noise_Q[:,:,:, 1]
        noise_U = noise_U[:,:,:, 1]

        print(np.shape(noise_Q))

        return noise_Q, noise_U

    def reading_dust_patches():
        
        Number_Of_Initial_Pixels = 256
        Number_Of_Cutted_Pixels = 8
        Number_Of_Pixels = 240
        
        data_Q = ['./Validation_Patches_D_Q_30arcmin.h5', './Validation_Patches_D_Q_25arcmin.h5', './Validation_Patches_D_Q_20arcmin.h5', './Validation_Patches_D_Q_5arcmin_d6s1.h5', './Validation_Patches_D_Q_5arcmin_d4s2.h5']
        data_U = ['./Validation_Patches_D_U_30arcmin.h5', './Validation_Patches_D_Q_25arcmin.h5', './Validation_Patches_D_U_20arcmin.h5', './Validation_Patches_D_U_5arcmin_d6s1.h5', './Validation_Patches_D_U_5arcmin_d4s2.h5']
        
        data_Q_30arcmin = h5py.File(data_Q[0], 'r')
        data_U_30arcmin = h5py.File(data_U[0], 'r')
        
        data_Q_25arcmin = h5py.File(data_Q[1], 'r')
        data_U_25arcmin = h5py.File(data_U[1], 'r')
        
        data_Q_20arcmin = h5py.File(data_Q[2], 'r')
        data_U_20arcmin = h5py.File(data_U[2], 'r')
        
        data_Q_5arcmin_d6s1 = h5py.File(data_Q[3], 'r')
        data_U_5arcmin_d6s1 = h5py.File(data_U[3], 'r')
        
        data_Q_5arcmin_d4s2 = h5py.File(data_Q[4], 'r')
        data_U_5arcmin_d4s2 = h5py.File(data_U[4], 'r')
        
        noise_Q_30arcmin = data_Q_30arcmin['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        noise_U_30arcmin = data_U_30arcmin['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)

        noise_Q_25arcmin = data_Q_25arcmin['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        noise_U_25arcmin = data_U_25arcmin['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)

        noise_Q_20arcmin = data_Q_20arcmin['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        noise_U_20arcmin = data_U_20arcmin['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)

        dust_Q_5arcmin_d6s1 = data_Q_5arcmin_d6s1['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        dust_U_5arcmin_d6s1 = data_U_5arcmin_d6s1['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)

        dust_Q_5arcmin_d4s2 = data_Q_5arcmin_d4s2['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        dust_U_5arcmin_d4s2 = data_U_5arcmin_d4s2['M'][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)


        return noise_Q_30arcmin, noise_U_30arcmin, noise_Q_25arcmin, noise_U_25arcmin, noise_Q_20arcmin, noise_U_20arcmin, dust_Q_5arcmin_d6s1, dust_U_5arcmin_d6s1, dust_Q_5arcmin_d4s2, dust_U_5arcmin_d4s2
    
class Estimate_E_B_Power_Spectrum():
    
    def estimating_theoretical_power_spectrum_all_sky(Archivo_fits):
        
        CMB = CMB = hp.read_map(Archivo_fits, (0, 1, 2))
        
        # fwhm 0.5 deg in rad
        
        #CMB_Smoothed = hp.sphtfunc.smoothing(CMB, fwhm = 0.00872665)
        
        Cls = hp.sphtfunc.anafast(CMB, lmax = 2500)
        
        l = np.arange(0, 2501)
        
        # maps in muK^2
        
        EE = Cls[1]*1e12
        BB = Cls[2]*1e12
        
        # change to Cl l(l+1)/2pi
        
        EE = EE*(l*(l+1))/(2*np.pi)
        BB = BB*(l*(l+1))/(2*np.pi)
        
        return l, EE, BB

    def estimating_E_B_from_CENN(data_Q, data_U, flag):
        
        Number_Of_Initial_Pixels = 256
        Number_Of_Cutted_Pixels = 8
        Number_Of_Pixels = 240
        # reading data
        
        # validation_Q = './Validation_Q.h5'
        # validation_U = './Validation_U.h5'
        
        #data_Q = './Outputs_CENN_Q.h5'
        #data_U = './Outputs_CENN_U.h5'
        
        # validation_Q = h5py.File(validation_Q, 'r')
        # validation_U = h5py.File(validation_U, 'r')
        
        data_Q = h5py.File(data_Q, 'r')
        data_U = h5py.File(data_U, 'r')
        
        # Q
        
        outputs_Q = data_Q["net"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        # outputs_Q = outputs_Q
        inputs_Q = data_Q["sim"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        
        # U
            
        outputs_U = data_U["net"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
        # outputs_U = outputs_U
        inputs_U = data_U["sim"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)

        # outputs_Q = data_Q["net"][:,:Number_Of_Pixels,:Number_Of_Pixels].astype(np.float32)
        # inputs_Q = data_Q["sim"][:,:Number_Of_Pixels,:Number_Of_Pixels].astype(np.float32)
        
        # # # U
            
        # outputs_U = data_U["net"][:,:Number_Of_Pixels,:Number_Of_Pixels].astype(np.float32)
        # inputs_U = data_U["sim"][:,:Number_Of_Pixels,:Number_Of_Pixels].astype(np.float32)
        
        # test with 5 patches
        
        # outputs_Q = outputs_Q[0:5]
        # inputs_Q = inputs_Q[0:5]
        # outputs_U = outputs_U[0:5]
        # inputs_U = inputs_U[0:5]
        
        # Residuals
        
        Input_CMB_lista_ee = []
        Input_CMB_lista_bb = []
        Output_CMB_lista_ee = []
        Output_CMB_lista_bb = []
        Residuals_CMB_lista_ee = []
        Residuals_CMB_lista_bb = []
        mean_ee = []
        std_ee = []
        mean_bb = []
        std_bb = []
        
        for i in range(len(inputs_Q)):
            
            # namaster code starts here
    
            Lx = ((Number_Of_Pixels*90)/3600) * np.pi/180
            Ly = ((Number_Of_Pixels*90)/3600) * np.pi/180
    
            Nx = Number_Of_Pixels
            Ny = Number_Of_Pixels
    
            #l, cl_tt, cl_ee_th, cl_bb_th, cl_te = np.loadtxt('cls.txt', unpack=True)
            #beam = np.exp(-((0.5)* np.pi/180 * l)**2)
            #cl_tt *= beam
            #cl_ee_th *= beam
            #cl_bb_th *= beam
            #cl_te *= beam
            #mpt, mpq, mpu = nmt.synfast_flat(Nx, Ny, Lx, Ly,
                                             #np.array([cl_tt, cl_te, 0 * cl_tt,
                                                       #cl_ee_th, 0 * cl_ee_th, cl_bb_th]),
                                             #[0, 2])
    
            #mask = np.ones_like(mpt).flatten()
            mask = np.ones_like(inputs_Q[0,:]).flatten()
            xarr = np.ones(Ny)[:, None] * np.arange(Nx)[None, :] * Lx/Nx
            yarr = np.ones(Nx)[None, :] * np.arange(Ny)[:, None] * Ly/Ny
            
            """            
            # First we dig a couple of holes
            def dig_hole(x, y, r):
                rad = (np.sqrt((xarr - x)**2 + (yarr - y)**2)).flatten()
                return np.where(rad < r)[0]
            
            
            mask[dig_hole(0.3 * Lx, 0.6 * Ly, 0.05 * np.sqrt(Lx * Ly))] = 0.
            mask[dig_hole(0.7 * Lx, 0.12 * Ly, 0.07 * np.sqrt(Lx * Ly))] = 0.
            mask[dig_hole(0.7 * Lx, 0.8 * Ly, 0.03 * np.sqrt(Lx * Ly))] = 0.
            """

            apo_fac=16    
            mask[np.where(xarr.flatten() < Lx / apo_fac)] = 0
            mask[np.where(xarr.flatten() > (apo_fac-1.) * Lx / apo_fac)] = 0
            mask[np.where(yarr.flatten() < Lx / apo_fac)] = 0
            mask[np.where(yarr.flatten() > (apo_fac-1.) * Lx / apo_fac)] = 0
            mask = mask.reshape([Nx, Ny])

            mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=1., apotype="C1")
    
            #mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=aposcale[flag], apotype="C1")
            # mask2 = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=1., apotype="C1")
            # x 1e6 to muK
    
            f2_inputs = nmt.NmtFieldFlat(Lx, Ly, mask, [inputs_Q[i][:,:,0], inputs_U[i][:,:,0]], purify_b=True)
            f2_outputs = nmt.NmtFieldFlat(Lx, Ly, mask, [outputs_Q[i][:,:,0], outputs_U[i][:,:,0]], purify_b=True)            
            #f2_residuals = nmt.NmtFieldFlat(Lx, Ly, mask, [(inputs_Q[i][:,:,0]-outputs_Q[i][:,:,0]), (
             #   inputs_U[i][:,:]-outputs_U[i][:,:])], purify_b=True)
            
            # l0_bins = 10**np.arange(1.5, 3.5, 0.1)
            # lf_bins = 10**np.arange(1.5, 3.5, 0.1)
            
            #l0_bins = np.arange(50, lmax[flag], 25)
            #lf_bins = np.arange(60, lmax[flag], 25)
            bins = np.arange(0,2525,35)
            b = nmt.NmtBinFlat(bins[:-1], bins[1:])
            #b = nmt.NmtBinFlat(l0_bins, lf_bins)
    
            ells_uncoupled = b.get_effective_ells()
            
            # True CMB from Q

            w22 = nmt.NmtWorkspaceFlat()
            w22.compute_coupling_matrix(f2_inputs, f2_inputs, b)

            w33 = nmt.NmtWorkspaceFlat()
            w33.compute_coupling_matrix(f2_outputs, f2_outputs, b)

            #w44 = nmt.NmtWorkspaceFlat()
            #w44.compute_coupling_matrix(f2_residuals, f2_residuals, b)

            
            Input_CMB = nmt.compute_coupled_cell_flat(f2_inputs, f2_inputs, b)
            Output_CMB = nmt.compute_coupled_cell_flat(f2_outputs, f2_outputs, b)
            #Residuals_CMB = nmt.compute_coupled_cell_flat(f2_residuals, f2_residuals, b)

            Input_CMB = w22.decouple_cell(Input_CMB)
            Output_CMB = w33.decouple_cell(Output_CMB)
            #Residuals_CMB = w44.decouple_cell(Residuals_CMB)

    
            Input_CMB_lista_ee.append(Input_CMB[0]*1e12)
            Input_CMB_lista_bb.append(Input_CMB[3]*1e12)
            
            Output_CMB_lista_ee.append(Output_CMB[0]*1e12)
            Output_CMB_lista_bb.append(Output_CMB[3]*1e12)
            
            #Residuals_CMB_lista_ee.append(Residuals_CMB[0]*1e12)
            #Residuals_CMB_lista_bb.append(Residuals_CMB[3]*1e12)
            
        # estimating average power spectrum from all the patches    
            
        # Cls Input
        
        cl22_coupled_dummy_ee = 0
        cl22_coupled_dummy_bb = 0
        suma_std_ee = 0
        suma_std_bb = 0
            
        for i in range(len(Input_CMB_lista_ee)):
            
            cl22_coupled_dummy_ee += Input_CMB_lista_ee[i]
            cl22_coupled_dummy_bb += Input_CMB_lista_bb[i]

        cl_ee = cl22_coupled_dummy_ee/len(Input_CMB_lista_ee)
        cl_bb = cl22_coupled_dummy_bb/len(Input_CMB_lista_bb)

        for i in range(len(Input_CMB_lista_ee)):
            
            suma_std_dummy_ee = (Input_CMB_lista_ee[i] - cl_ee)**2
            suma_std_ee += suma_std_dummy_ee
            
            suma_std_dummy_bb = (Input_CMB_lista_bb[i] - cl_bb)**2
            suma_std_bb += suma_std_dummy_bb
            
        cl_ee_std = np.sqrt(suma_std_ee/len(Input_CMB_lista_ee))
        cl_bb_std = np.sqrt(suma_std_bb/len(Input_CMB_lista_bb))
        
        mean_ee.append(cl_ee)
        std_ee.append(cl_ee_std)
        mean_bb.append(cl_bb)
        std_bb.append(cl_bb_std)
        
        # Cls Output
             
        cl22_coupled_dummy_ee = 0
        cl22_coupled_dummy_bb = 0
        suma_std_ee = 0
        suma_std_bb = 0
            
        for i in range(len(Input_CMB_lista_ee)):
            
            cl22_coupled_dummy_ee += Output_CMB_lista_ee[i]
            cl22_coupled_dummy_bb += Output_CMB_lista_bb[i]

        cl_ee = cl22_coupled_dummy_ee/len(Output_CMB_lista_ee)
        cl_bb = cl22_coupled_dummy_bb/len(Output_CMB_lista_bb)
        
        for i in range(len(Input_CMB_lista_ee)):
            
            suma_std_dummy_ee = (Output_CMB_lista_ee[i] - cl_ee)**2
            suma_std_ee += suma_std_dummy_ee
            
            suma_std_dummy_bb = (Output_CMB_lista_bb[i] - cl_bb)**2
            suma_std_bb += suma_std_dummy_bb
            
        cl_ee_std = np.sqrt(suma_std_ee/len(Output_CMB_lista_ee))
        cl_bb_std = np.sqrt(suma_std_bb/len(Output_CMB_lista_bb))
        
        mean_ee.append(cl_ee)
        std_ee.append(cl_ee_std)
        mean_bb.append(cl_bb)
        std_bb.append(cl_bb_std)
        
        # Cls Residuals
        """
        cl22_coupled_dummy_ee = 0
        cl22_coupled_dummy_bb = 0
        suma_std_ee = 0
        suma_std_bb = 0
            
        for i in range(len(Input_CMB_lista_ee)):
            
            cl22_coupled_dummy_ee += Residuals_CMB_lista_ee[i]
            cl22_coupled_dummy_bb += Residuals_CMB_lista_bb[i]

        cl_ee = cl22_coupled_dummy_ee/len(Residuals_CMB_lista_ee)
        cl_bb = cl22_coupled_dummy_bb/len(Residuals_CMB_lista_bb)
        
        for i in range(len(Input_CMB_lista_ee)):
            
            suma_std_dummy_ee = (Residuals_CMB_lista_ee[i] - cl_ee)**2
            suma_std_ee += suma_std_dummy_ee
            
            suma_std_dummy_bb = (Residuals_CMB_lista_bb[i] - cl_bb)**2
            suma_std_bb += suma_std_dummy_bb
            
        cl_ee_std = np.sqrt(suma_std_ee/len(Residuals_CMB_lista_ee))
        cl_bb_std = np.sqrt(suma_std_bb/len(Residuals_CMB_lista_bb))
        
        mean_ee.append(cl_ee)
        std_ee.append(cl_ee_std)
        mean_bb.append(cl_bb)
        std_bb.append(cl_bb_std)
        """
        return mean_ee, mean_bb, std_ee, std_bb, ells_uncoupled
    
    def estimating_noise_power_spectra():
        
        # flag 0 = 30arcmin, flag 1 = 25 arcmin, flag 2 = 20 arcmin
        
        aposcale = [0.8]
        lmax = [800]
    
        Number_Of_Pixels = 240
            
        noise_Q, noise_U = Read.reading_noise_patches()
        
        #noise_Q = [noise_Q_30arcmin, noise_Q_25arcmin, noise_Q_20arcmin]
        #noise_U = [noise_U_30arcmin, noise_U_25arcmin, noise_U_20arcmin]
        
        #noise_Q = noise_Q[resolution]
        #noise_U = noise_U[resolution]
        
        #noise_Q = noise_Q[:,:,:,:]
        #noise_U = noise_U[:,:,:,:]        
        
        Input_CMB_lista_ee = []
        Input_CMB_lista_bb = []
        mean_ee = []
        std_ee = []
        mean_bb = []
        std_bb = []
        
        for i in range(len(noise_Q)):
            
             # namaster code starts here
    
            Lx = ((Number_Of_Pixels*90)/3600) * np.pi/180
            Ly = ((Number_Of_Pixels*90)/3600) * np.pi/180
    
            Nx = Number_Of_Pixels
            Ny = Number_Of_Pixels
    
            #l, cl_tt, cl_ee_th, cl_bb_th, cl_te = np.loadtxt('cls.txt', unpack=True)
            #beam = np.exp(-((0.5)* np.pi/180 * l)**2)
            #cl_tt *= beam
            #cl_ee_th *= beam
            #cl_bb_th *= beam
            #cl_te *= beam
            #mpt, mpq, mpu = nmt.synfast_flat(Nx, Ny, Lx, Ly,
                                             #np.array([cl_tt, cl_te, 0 * cl_tt,
                                                       #cl_ee_th, 0 * cl_ee_th, cl_bb_th]),
                                             #[0, 2])
    
            #mask = np.ones_like(mpt).flatten()
            mask = np.ones_like(noise_Q[0,:]).flatten()
            xarr = np.ones(Ny)[:, None] * np.arange(Nx)[None, :] * Lx/Nx
            yarr = np.ones(Nx)[None, :] * np.arange(Ny)[:, None] * Ly/Ny
            
            """            
            # First we dig a couple of holes
            def dig_hole(x, y, r):
                rad = (np.sqrt((xarr - x)**2 + (yarr - y)**2)).flatten()
                return np.where(rad < r)[0]
            
            
            mask[dig_hole(0.3 * Lx, 0.6 * Ly, 0.05 * np.sqrt(Lx * Ly))] = 0.
            mask[dig_hole(0.7 * Lx, 0.12 * Ly, 0.07 * np.sqrt(Lx * Ly))] = 0.
            mask[dig_hole(0.7 * Lx, 0.8 * Ly, 0.03 * np.sqrt(Lx * Ly))] = 0.
            """

            apo_fac=16    
            mask[np.where(xarr.flatten() < Lx / apo_fac)] = 0
            mask[np.where(xarr.flatten() > (apo_fac-1.) * Lx / apo_fac)] = 0
            mask[np.where(yarr.flatten() < Lx / apo_fac)] = 0
            mask[np.where(yarr.flatten() > (apo_fac-1.) * Lx / apo_fac)] = 0
            mask = mask.reshape([Nx, Ny])

            mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=1., apotype="C1")

    
            #mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=aposcale[flag], apotype="C1")
            # mask2 = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=1., apotype="C1")
            # x 1e6 to muK
    
            f2_inputs = nmt.NmtFieldFlat(Lx, Ly, mask, [noise_Q[i][:,:], noise_U[i][:,:]], purify_b=True)      
            
            # l0_bins = 10**np.arange(1.5, 3.5, 0.1)
            # lf_bins = 10**np.arange(1.5, 3.5, 0.1)
            
            #l0_bins = np.arange(50, lmax[flag], 25)
            #lf_bins = np.arange(60, lmax[flag], 25)
            bins = np.arange(0,2025,35)
            b = nmt.NmtBinFlat(bins[:-1], bins[1:])
            #b = nmt.NmtBinFlat(l0_bins, lf_bins)
    
            ells_uncoupled = b.get_effective_ells()
            
            # True CMB from Q

            w22 = nmt.NmtWorkspaceFlat()
            w22.compute_coupling_matrix(f2_inputs, f2_inputs, b)

            
            Input_CMB = nmt.compute_coupled_cell_flat(f2_inputs, f2_inputs, b)

            Input_CMB = w22.decouple_cell(Input_CMB)

    
            Input_CMB_lista_ee.append(Input_CMB[0]*1e12)
            Input_CMB_lista_bb.append(Input_CMB[3]*1e12)
            
        # estimating average power spectrum from all the patches    
            
        # Cls Input
        
        cl22_coupled_dummy_ee = 0
        cl22_coupled_dummy_bb = 0
        suma_std_ee = 0
        suma_std_bb = 0
           
        for i in range(len(Input_CMB_lista_ee)):
            
            cl22_coupled_dummy_ee += Input_CMB_lista_ee[i]
            cl22_coupled_dummy_bb += Input_CMB_lista_bb[i]
    
        cl_ee = cl22_coupled_dummy_ee/len(Input_CMB_lista_ee)
        cl_bb = cl22_coupled_dummy_bb/len(Input_CMB_lista_bb)
        
        for i in range(len(Input_CMB_lista_ee)):
            
            suma_std_dummy_ee = (Input_CMB_lista_ee[i] - cl_ee)**2
            suma_std_ee += suma_std_dummy_ee
            
            suma_std_dummy_bb = (Input_CMB_lista_bb[i] - cl_bb)**2
            suma_std_bb += suma_std_dummy_bb
            
        cl_ee_std = np.sqrt(suma_std_ee/len(Input_CMB_lista_ee))
        cl_bb_std = np.sqrt(suma_std_bb/len(Input_CMB_lista_bb))
        
        mean_ee.append(cl_ee)
        std_ee.append(cl_ee_std)
        mean_bb.append(cl_bb)
        std_bb.append(cl_bb_std)
        
        return mean_ee, mean_bb, std_ee, std_bb, ells_uncoupled

    def estimating_dust_power_spectra(resolution, channel):
        
        # flag 0 = 30arcmin, flag 1 = 25 arcmin, flag 2 = 20 arcmin
        
        aposcale = [0.8, 1., 1.2, 1.5, 1.5]
        lmax = [600, 700, 800, 1500, 1500]
    
        Number_Of_Pixels = 240
            
        noise_Q_30arcmin, noise_U_30arcmin, noise_Q_25arcmin, noise_U_25arcmin, noise_Q_20arcmin, noise_U_20arcmin, dust_Q_5arcmin_d6s1, dust_U_5arcmin_d6s1, dust_Q_5arcmin_d4s2, dust_U_5arcmin_d4s2 = Read.reading_dust_patches()
    
        noise_Q = [noise_Q_30arcmin, noise_Q_25arcmin, noise_Q_20arcmin, dust_Q_5arcmin_d6s1, dust_Q_5arcmin_d4s2]
        noise_U = [noise_U_30arcmin, noise_U_25arcmin, noise_U_20arcmin, dust_U_5arcmin_d6s1, dust_U_5arcmin_d4s2]
        
        noise_Q = noise_Q[resolution]
        noise_U = noise_U[resolution]  
        
        noise_Q = noise_Q[:,:,:,channel]
        noise_U = noise_U[:,:,:,channel]  
        
        Input_CMB_lista_ee = []
        Input_CMB_lista_bb = []
        mean_ee = []
        std_ee = []
        mean_bb = []
        std_bb = []
        
        for i in range(len(noise_Q_30arcmin)):
            
            # namaster code starts here
    
            Lx = ((Number_Of_Pixels*90)/3600) * np.pi/180
            Ly = ((Number_Of_Pixels*90)/3600) * np.pi/180
    
            Nx = Number_Of_Pixels
            Ny = Number_Of_Pixels
    
            l, cl_tt, cl_ee_th, cl_bb_th, cl_te = np.loadtxt('cls.txt', unpack=True)
            beam = np.exp(-((0.5)* np.pi/180 * l)**2)
            cl_tt *= beam
            cl_ee_th *= beam
            cl_bb_th *= beam
            cl_te *= beam
            mpt, mpq, mpu = nmt.synfast_flat(Nx, Ny, Lx, Ly,
                                             np.array([cl_tt, cl_te, 0 * cl_tt,
                                                       cl_ee_th, 0 * cl_ee_th, cl_bb_th]),
                                             [0, 2])
    
            mask = np.ones_like(mpt).flatten()
            xarr = np.ones(Ny)[:, None] * np.arange(Nx)[None, :] * Lx/Nx
            yarr = np.ones(Nx)[None, :] * np.arange(Ny)[:, None] * Ly/Ny
            
            
            # First we dig a couple of holes
            def dig_hole(x, y, r):
                rad = (np.sqrt((xarr - x)**2 + (yarr - y)**2)).flatten()
                return np.where(rad < r)[0]
            
            
            mask[dig_hole(0.3 * Lx, 0.6 * Ly, 0.05 * np.sqrt(Lx * Ly))] = 0.
            mask[dig_hole(0.7 * Lx, 0.12 * Ly, 0.07 * np.sqrt(Lx * Ly))] = 0.
            mask[dig_hole(0.7 * Lx, 0.8 * Ly, 0.03 * np.sqrt(Lx * Ly))] = 0.
    
            mask[np.where(xarr.flatten() < Lx / 16.)] = 0
            mask[np.where(xarr.flatten() > 15 * Lx / 16.)] = 0
            mask[np.where(yarr.flatten() < Ly / 16.)] = 0
            mask[np.where(yarr.flatten() > 15 * Ly / 16.)] = 0
            mask = mask.reshape([Ny, Nx])
    
            mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=aposcale[resolution], apotype="C1")
            # mask2 = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=1., apotype="C1")
            # x 1e6 to muK
    
            f2_inputs = nmt.NmtFieldFlat(Lx, Ly, mask, [noise_Q[i][:,:]*1e6, noise_U[i][:,:]*1e6], purify_b=True)
            
            # l0_bins = 10**np.arange(1.5, 3.5, 0.1)
            # lf_bins = 10**np.arange(1.5, 3.5, 0.1)
            
            l0_bins = np.arange(50, lmax[resolution], 25)
            lf_bins = np.arange(60, lmax[resolution], 25)
            
            
            b = nmt.NmtBinFlat(l0_bins, lf_bins)
    
            ells_uncoupled = b.get_effective_ells()
            
            # True CMB from Q
            
            Input_CMB = nmt.compute_coupled_cell_flat(f2_inputs, f2_inputs, b)
    
            Input_CMB_lista_ee.append(Input_CMB[0])
            Input_CMB_lista_bb.append(Input_CMB[3])
            
        # estimating average power spectrum from all the patches    
            
        # Cls Input
        
        cl22_coupled_dummy_ee = 0
        cl22_coupled_dummy_bb = 0
        suma_std_ee = 0
        suma_std_bb = 0
            
        for i in range(len(Input_CMB_lista_ee)):
            
            cl22_coupled_dummy_ee += Input_CMB_lista_ee[i]
            cl22_coupled_dummy_bb += Input_CMB_lista_bb[i]
    
        cl_ee = cl22_coupled_dummy_ee/len(Input_CMB_lista_ee)
        cl_bb = cl22_coupled_dummy_bb/len(Input_CMB_lista_bb)
        
        for i in range(len(Input_CMB_lista_ee)):
            
            suma_std_dummy_ee = (Input_CMB_lista_ee[i] - cl_ee)**2
            suma_std_ee += suma_std_dummy_ee
            
            suma_std_dummy_bb = (Input_CMB_lista_bb[i] - cl_bb)**2
            suma_std_bb += suma_std_dummy_bb
            
        cl_ee_std = np.sqrt(suma_std_ee/len(Input_CMB_lista_ee))
        cl_bb_std = np.sqrt(suma_std_bb/len(Input_CMB_lista_bb))
        
        mean_ee.append(cl_ee)
        std_ee.append(cl_ee_std)
        mean_bb.append(cl_bb)
        std_bb.append(cl_bb_std)
        
        return mean_ee, mean_bb, std_ee, std_bb, ells_uncoupled
    

class Plot_E_B_Power_Spectrum():
    
    def reading_noise_spectra():
        
        # resolution 0 is 30 arcmin, 1 is 25 and 2 is 20
        
        EE = []
        BB = []
        EE_Uncertainty = []
        BB_Uncertainty = []
        l = []
        
        for i in range(3):
        
            cl_ee, cl_bb, cl_ee_std, cl_bb_std, ells_uncoupled = Estimate_E_B_Power_Spectrum.estimating_noise_power_spectra()
            
            EE.append(cl_ee)
            BB.append(cl_bb)
            EE_Uncertainty.append(cl_ee_std)
            BB_Uncertainty.append(cl_bb_std)
            l.append(ells_uncoupled)
            
        return EE, BB, EE_Uncertainty, BB_Uncertainty, l
    
    def reading_dust_spectra(resolution):
        
        # resolution 0 is 30 arcmin, 1 is 25 and 2 is 20
        
        EE = []
        BB = []
        EE_Uncertainty = []
        BB_Uncertainty = []
        l = []
        
        for i in range(2):
        
            cl_ee, cl_bb, cl_ee_std, cl_bb_std, ells_uncoupled = Estimate_E_B_Power_Spectrum.estimating_dust_power_spectra(resolution=resolution, channel=i)
            
            EE.append(cl_ee)
            BB.append(cl_bb)
            EE_Uncertainty.append(cl_ee_std)
            BB_Uncertainty.append(cl_bb_std)
            l.append(ells_uncoupled)
            
        return EE, BB, EE_Uncertainty, BB_Uncertainty, l
    
    def reading_the_spectra():
        
        EE = []
        BB = []
        EE_Uncertainty = []
        BB_Uncertainty = []
        l = []
        
        for i in range(len(data_Q)):
        
            cl_ee, cl_bb, cl_ee_std, cl_bb_std, ells_uncoupled = Estimate_E_B_Power_Spectrum.estimating_E_B_from_CENN(data_Q[i], data_U[i], flag=i)
            
            EE.append(cl_ee)
            BB.append(cl_bb)
            EE_Uncertainty.append(cl_ee_std)
            BB_Uncertainty.append(cl_bb_std)
            l.append(ells_uncoupled)
            
        return EE, BB, EE_Uncertainty, BB_Uncertainty, l    


    def plotting_E_B_Power_Spectrum_from_CENN():
        
        EE, BB, EE_Uncertainty, BB_Uncertainty, l = Plot_E_B_Power_Spectrum.reading_the_spectra()
        #EE_noise, BB_noise, EE_Uncertainty_noise, BB_Uncertainty_noise, l_noise = Plot_E_B_Power_Spectrum.reading_noise_spectra()
        # EE_dust, BB_dust, EE_Uncertainty_dust, BB_Uncertainty_dust, l_dust = Plot_E_B_Power_Spectrum.reading_dust_spectra(resolution=3)
        # EE_dust_d4s2, BB_dust_d4s2, EE_Uncertainty_dust_d4s2, BB_Uncertainty_dust_d4s2, l_dust_d4s2 = Plot_E_B_Power_Spectrum.reading_dust_spectra(resolution=4)
        # EE_dust, BB_dust, EE_Uncertainty_dust, BB_Uncertainty_dust, l_dust = Plot_E_B_Power_Spectrum.reading_dust_spectra()
              
        l_th_all_sky_Planck, EE_th_all_sky_Planck, BB_th_all_sky_Planck = Estimate_E_B_Power_Spectrum.estimating_theoretical_power_spectrum_all_sky('./CMB_143GHZ_SimPlanck.fits')
        l_th_all_sky_Planck_ruido, EE_th_all_sky_Planck_ruido, BB_th_all_sky_Planck_ruido = Estimate_E_B_Power_Spectrum.estimating_theoretical_power_spectrum_all_sky('./Ruido_Planck_143.fits')
        #l_th_all_sky_0_2, EE_th_all_sky_0_2, BB_th_all_sky_0_2 = Estimate_E_B_Power_Spectrum.estimating_theoretical_power_spectrum_all_sky('./Mapa_total_143_GHz_R_0_2.fits')
        #l_th_all_sky_0_1, EE_th_all_sky_0_1, BB_th_all_sky_0_1 = Estimate_E_B_Power_Spectrum.estimating_theoretical_power_spectrum_all_sky('./Mapa_total_143_GHz_R_0_1.fits')
        #l_th_all_sky_0_05, EE_th_all_sky_0_05, BB_th_all_sky_0_05 = Estimate_E_B_Power_Spectrum.estimating_theoretical_power_spectrum_all_sky('./Mapa_total_143_GHz_R_0_05.fits')
        #l_th_all_sky_0_01, EE_th_all_sky_0_01, BB_th_all_sky_0_01 = Estimate_E_B_Power_Spectrum.estimating_theoretical_power_spectrum_all_sky('./Mapa_total_143_GHz_R_0_01.fits')
        #l_th_all_sky_0_004, EE_th_all_sky_0_004, BB_th_all_sky_0_004 = Estimate_E_B_Power_Spectrum.estimating_theoretical_power_spectrum_all_sky('./Mapa_total_143_GHz_R_0_004.fits')
        #l_th_all_sky_0_001, EE_th_all_sky_0_001, BB_th_all_sky_0_001 = Estimate_E_B_Power_Spectrum.estimating_theoretical_power_spectrum_all_sky('./Mapa_total_143_GHz_R_0_001.fits')


        
        
        # EE
        
                  
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
        
        #a0.plot(l[0], ((EE[0][0]*(l[0]*(l[0]+1))/(2*np.pi))), color=color_Input[0], label='Input')
        #a0.fill_between(l[0], ((EE[0][0]*(l[0]*(l[0]+1))/(2*np.pi))) - (
            #(EE_Uncertainty[0][0]*(l[0]*(l[0]+1))/(2*np.pi))), ((EE[0][0]*(l[0]*(
                #l[0]+1))/(2*np.pi))) + ((EE_Uncertainty[0][0]*(l[0]*(l[0]+1))/(
                    #2*np.pi))), color=color_Input[0], alpha=0.2)
                        
        
        for i in range(len(data_Q)):
        
            

            if i ==0:
                a0.plot(l_th_all_sky_Planck, EE_th_all_sky_Planck, 'black', linestyle='dashed', color='black', label='EE Planck')
                a0.plot(l_th_all_sky_Planck_ruido, EE_th_all_sky_Planck_ruido, linestyle='dashed', color='grey', label='Planck noise')
                #a0.plot(l[i], ((EE[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), color='purple', label=label_Input[i])
                #a0.plot(l_noise[i], ((EE_noise[i][0]*(l_noise[i]*(l_noise[i]+1))/(2*np.pi))), linestyle='dashed', color='grey', label='Patch noise')
                #a0.fill_between(l[i], ((EE[i][0]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                     #(EE_Uncertainty[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), ((EE[i][0]*(l[i]*(
                         #l[i]+1))/(2*np.pi))) + ((EE_Uncertainty[i][0]*(l[i]*(l[i]+1))/(
                             #2*np.pi))), alpha=0.2)
                
                a0.plot(l[i], ((EE[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), color='orange', label=label_Output[i])
                a0.fill_between(l[i], ((EE[i][1]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                    (EE_Uncertainty[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), ((EE[i][1]*(l[i]*(
                        l[i]+1))/(2*np.pi))) + ((EE_Uncertainty[i][1]*(l[i]*(l[i]+1))/(
                            2*np.pi))), alpha=0.2)

                #a0.plot(2000, ((EE[i][1][56]*(l[i][56]*(l[i][56]+1))/(2*np.pi))), color='red', marker = "*", label='l = 2000')
                
                a1.plot(l[i], (((EE[i][0]-EE[i][1])*(l[i]*(l[i]+1))/(2*np.pi))), color=color_Error[i], label = label_Error[i])
                a1.fill_between(l[i], (((EE[i][0]-EE[i][1])*(l[i]*(l[i]+1))/(2*np.pi))) - ((
                    EE_Uncertainty[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), (((EE[i][0]-EE[i][1])*(l[i]*(
                        l[i]+1))/(2*np.pi))) + ((EE_Uncertainty[i][1]*(l[i]*(l[i]+1))/(
                            2*np.pi))), color=color_Error[i], alpha=0.2)
                
            else:
                 err=(((EE[i][0][56]-EE[i][1][56])*(l[i][56]*(l[i][56]+1))/(2*np.pi)))
                 print(err)
                 a0.errorbar(2000, (EE[0][1][56]*(l[0][56]*(l[0][56]+1))/(2*np.pi)), yerr=err,  fmt='*',  color='red',  label='l = 2000', capsize=5, capthick=2)
                 #a0.plot(l[i], ((EE[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), color='blue', label=label_Input[i])
                 #a0.plot(l[i], ((EE[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), color='orange', label=label_Output[i])
                 #a1.plot(l[i], (((EE[i][0]-EE[i][1])*(l[i]*(l[i]+1))/(2*np.pi))), color=color_Error[i], label = label_Error[i])

            a1.axhline(y = 0, color = 'tab:red', linestyle = 'dashed',linewidth=1)
        
            a0.set_ylabel(r'$\mathcal{D}_\mathcal{l}^{EE}$'+'[$\u03bcK^{2}$]')
            a1.set_ylabel(r'$\Delta\mathcal{D}_\mathcal{l}^{EE}$'+'[$\u03bcK^{2}$]')
            f.tight_layout()
            a0.set_yscale('log')
            a0.set_xscale('log')
            a1.set_xscale('log')
            a0.set_xlabel('$\mathcal{l}$')
            a0.legend(loc='lower right', fontsize=7)
            #a1.legend(loc='upper right', fontsize=6)
            a0.set_xlim(10, 2501)
            a1.set_xlim(10, 2501)
            plt.ylim([0.00001,1201])
            a1.set_ylim([-40,100])
        
        
        plt.savefig('EE_Planck.pdf')
        
                       
# BB
        """
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

        #a0.plot(l[0], ((BB[0][0]*(l[0]*(l[0]+1))/(2*np.pi))), color=color_Input[0], label='Input')
        #a0.fill_between(l[0], ((BB[0][0]*(l[0]*(l[0]+1))/(2*np.pi))) - (
         #   (BB_Uncertainty[0][0]*(l[0]*(l[0]+1))/(2*np.pi))), ((BB[0][0]*(l[0]*(
          #      l[0]+1))/(2*np.pi))) + ((BB_Uncertainty[0][0]*(l[0]*(l[0]+1))/(
           #         2*np.pi))), color=color_Input[0], alpha=0.2)


        a0.plot(l_th_all_sky_Planck, BB_th_all_sky_Planck, color='black', linestyle='dashed', label='BB Planck')
        a0.plot(l_th_all_sky_Planck_ruido, BB_th_all_sky_Planck_ruido, linestyle='dashed', color='purple', label='Planck noise')
        #a0.plot(l_th_all_sky_Planck2, BB_th_all_sky_Planck2, label='BB Planck antes')
        #a0.plot(l_th_all_sky_0_2, BB_th_all_sky_0_2, label='R = 0.2')
        #plt.plot(l_th_all_sky_0_05, BB_th_all_sky_0_05, label='R = 0.05')
        #a0.plot(l_th_all_sky_0_01, BB_th_all_sky_0_01, color = 'purple', label='Mapa Camb')
        #a0.plot(l_th_all_sky_0_004, BB_th_all_sky_0_004, label='R = 0.004')
        #plt.plot(l_th_all_sky_0_001, BB_th_all_sky_0_001, label='R = 0.001')


        #a0.plot(30, BB_th_all_sky_0_2[30], color='orange', marker = "*", label='r = 0.2 all sky')
        #a0.plot(80, BB_th_all_sky_0_2[80], color = 'orange', marker = "*")
        #a0.plot(100, BB_th_all_sky_0_2[100], color = 'orange', marker = "*")

        #a0.plot(30, BB_th_all_sky_0_1[30], color='red', marker = "+", label='r = 0.1 all sky')
        #a0.plot(80, BB_th_all_sky_0_1[80], color = 'red', marker = "+")
        #a0.plot(100, BB_th_all_sky_0_1[100], color = 'red', marker = "+")

        #a0.plot(30, BB_th_all_sky_0_05[30], color='brown', marker = "x", label='r = 0.05 all sky')
        #a0.plot(80, BB_th_all_sky_0_05[80], color = 'brown', marker = "x")
        #a0.plot(100,BB_th_all_sky_0_05[100], color = 'brown', marker = "x")
        
        for i in range(len(data_U)):
            if i ==0:
                a0.plot(l[i], ((BB[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), color='blue', label=label_Input[i])
                a0.plot(l[i], ((BB_noise[i][0]*(l_noise[i]*(l_noise[i]+1))/(2*np.pi))), color='grey', linestyle='dashed', label='Patch noise')
                a0.fill_between(l[i], ((BB[i][0]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                     (BB_Uncertainty[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), ((BB[i][0]*(l[i]*(
                         l[i]+1))/(2*np.pi))) + ((BB_Uncertainty[i][0]*(l[i]*(l[i]+1))/(
                             2*np.pi))), alpha=0.2)
                  
                a0.plot(l[i], ((BB[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), color='orange', label=label_Output[i])
                a0.fill_between(l[i], ((BB[i][1]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                    (BB_Uncertainty[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), ((BB[i][1]*(l[i]*(
                        l[i]+1))/(2*np.pi))) + ((BB_Uncertainty[i][1]*(l[i]*(l[i]+1))/(
                            2*np.pi))), alpha=0.2)
                
                a1.plot(l[i], (((BB[i][0]-BB[i][1])*(l[i]*(l[i]+1))/(2*np.pi))), color=color_Error[i], label = label_Error[i])
                a1.fill_between(l[i], (((BB[i][0]-BB[i][1])*(l[i]*(l[i]+1))/(2*np.pi))) - ((
                    BB_Uncertainty[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), (((BB[i][0]-BB[i][1])*(l[i]*(
                       l[i]+1))/(2*np.pi))) + ((BB_Uncertainty[i][1]*(l[i]*(l[i]+1))/(
                           2*np.pi))), color=color_Error[i], alpha=0.2)
               
            else: 
                a0.plot(l[i], ((BB[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), color='blue', label=label_Input[i])
                a0.plot(l[i], ((BB[i][1]*(l[i]*(l[i]+1))/(2*np.pi))), color='orange', label=label_Output[i])
                a1.plot(l[i], (((BB[i][0]-BB[i][1])*(l[i]*(l[i]+1))/(2*np.pi))), color=color_Error[i], label = label_Error[i])


                  
            a1.axhline(y = 0, color = 'tab:red', linestyle = 'dashed',linewidth=1)
                        
            a0.set_ylabel(r'$\mathcal{D}_\mathcal{l}^{BB}$'+'[$\u03bcK^{2}$]')
            a1.set_ylabel(r'$\Delta\mathcal{D}_\mathcal{l}^{BB}$'+'[$\u03bcK^{2}$]')
            f.tight_layout()
            a0.set_yscale('log')
            a0.set_xscale('log')
            a1.set_xscale('log')
            a0.set_xlabel('$\mathcal{l}$')
            a0.set_xlim(10, 1001)
            a1.set_xlim(10, 1001)
            #a0.set_ylim(0.00001, 10)
            a0.legend(loc='lower right', fontsize=6)
            #a1.legend(loc='upper right', fontsize=6)
            a1.set_ylim([-10, 5])
        
        plt.savefig('BB_raw_noise.pdf')
        """        
"""        
# Residuals

        plt.plot(l[0], ((EE[0][0]*(l[0]*(l[0]+1))/(2*np.pi))), color=color_Input[0], label='Input CMB')
        plt.fill_between(l[0], ((EE[0][0]*(l[0]*(l[0]+1))/(2*np.pi))) - (
            (EE_Uncertainty[0][0]*(l[0]*(l[0]+1))/(2*np.pi))), ((EE[0][0]*(l[0]*(
                l[0]+1))/(2*np.pi))) + ((EE_Uncertainty[0][0]*(l[0]*(l[0]+1))/(
                    2*np.pi))), color=color_Input[0], alpha=0.2)
                    
        
        for i in range(len(data_Q)):
            # plt.plot(l[i], ((EE[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), color=color_Input[i], label=label_Input[i])
            # plt.fill_between(l[i], ((EE[i][0]*(l[i]*(l[i]+1))/(2*np.pi))) - (
            #     (EE_Uncertainty[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), ((EE[i][0]*(l[i]*(
            #         l[i]+1))/(2*np.pi))) + ((EE_Uncertainty[i][0]*(l[i]*(l[i]+1))/(
            #             2*np.pi))), color=color_Input[i], alpha=0.2)            
            
            # plt.plot(l_noise[i], ((EE_noise[i][0]*(l_noise[i]*(l_noise[i]+1))/(2*np.pi))), color='grey', linestyle='--', label='Noise')
            
            plt.plot(l[i], ((EE[i][2]*(l[i]*(l[i]+1))/(2*np.pi))), color=color_Residuals[i], label=label_Residuals[i])
            plt.fill_between(l[i], ((EE[i][2]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                (EE_Uncertainty[i][2]*(l[i]*(l[i]+1))/(2*np.pi))), ((EE[i][2]*(l[i]*(
                    l[i]+1))/(2*np.pi))) + ((EE_Uncertainty[i][2]*(l[i]*(l[i]+1))/(
                        2*np.pi))), color=color_Residuals[i], alpha=0.2)
                        
            plt.ylabel(r'$\mathcal{D}_\mathcal{l}^{EE}$'+'[$\u03bcK^{2}$]')
            plt.tight_layout()
            plt.yscale('log')
            plt.yscale('log')
            plt.xlabel('$\mathcal{l}$')
        
        # plt.plot(l_noise[0], ((EE_noise[0][0]*(l_noise[0]*(l_noise[0]+1))/(2*np.pi))), color='tab:red', linestyle='dotted', label='Noise max')
        # plt.plot(l_noise[1], ((EE_noise[1][0]*(l_noise[1]*(l_noise[1]+1))/(2*np.pi))), color='tab:red', linestyle='dashed', label='Noise min') 

        # plt.plot(l_dust[1], ((EE_dust[1][0]*(l_dust[1]*(l_dust[1]+1))/(2*np.pi))), color='grey', linestyle='dashed', label='Dust d6 max')        
        # plt.plot(l_dust[0], ((EE_dust[0][0]*(l_dust[0]*(l_dust[0]+1))/(2*np.pi))), color='grey', linestyle='dotted', label='Dust d6 min')
       
        # plt.plot(l_dust_d4s2[1], ((EE_dust_d4s2[1][0]*(l_dust_d4s2[1]*(l_dust_d4s2[1]+1))/(2*np.pi))), color='tab:red', linestyle='dashed', label='Dust d4 max')
        # plt.plot(l_dust_d4s2[0], ((EE_dust_d4s2[0][0]*(l_dust_d4s2[0]*(l_dust_d4s2[0]+1))/(2*np.pi))), color='tab:red', linestyle='dotted', label='Dust d4 min')
                                       
        
        plt.legend(fontsize='small')    
        plt.ylim([1e-3,10])  
        #plt.savefig(Images_Directory+'Residuals_EE_Comparison_5arcmin'+'.pdf')
        
        
        
        
        plt.plot(l[0], ((BB[0][0]*(l[0]*(l[0]+1))/(2*np.pi))), color=color_Input[0], label='Input CMB')
        plt.fill_between(l[0], ((BB[0][0]*(l[0]*(l[0]+1))/(2*np.pi))) - (
            (BB_Uncertainty[0][0]*(l[0]*(l[0]+1))/(2*np.pi))), ((BB[0][0]*(l[0]*(
                l[0]+1))/(2*np.pi))) + ((BB_Uncertainty[0][0]*(l[0]*(l[0]+1))/(
                    2*np.pi))), color=color_Input[0], alpha=0.2)
        
        for i in range(len(data_Q)):   

            # plt.plot(l[i], ((BB[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), color=color_Input[i], label=label_Input[i])
            # plt.fill_between(l[i], ((BB[i][0]*(l[i]*(l[i]+1))/(2*np.pi))) - (
            #     (BB_Uncertainty[i][0]*(l[i]*(l[i]+1))/(2*np.pi))), ((BB[i][0]*(l[i]*(
            #         l[i]+1))/(2*np.pi))) + ((BB_Uncertainty[i][0]*(l[i]*(l[i]+1))/(
            #             2*np.pi))), color=color_Input[i], alpha=0.2)
            
            # plt.plot(l_noise[i], ((BB_noise[i][0]*(l_noise[i]*(l_noise[i]+1))/(2*np.pi))), color='grey', linestyle='--', label='Noise')
                        
            plt.plot(l[i], ((BB[i][2]*(l[i]*(l[i]+1))/(2*np.pi))), color=color_Residuals[i], label=label_Residuals[i])
            plt.fill_between(l[i], ((BB[i][2]*(l[i]*(l[i]+1))/(2*np.pi))) - (
                (BB_Uncertainty[i][2]*(l[i]*(l[i]+1))/(2*np.pi))), ((BB[i][2]*(l[i]*(
                    l[i]+1))/(2*np.pi))) + ((BB_Uncertainty[i][2]*(l[i]*(l[i]+1))/(
                        2*np.pi))), color=color_Residuals[i], alpha=0.2)
                        
            plt.ylabel(r'$\mathcal{D}_\mathcal{l}^{BB}$'+'[$\u03bcK^{2}$]')
            plt.tight_layout()
            plt.yscale('log')
            plt.yscale('log')
            plt.xlabel('$\mathcal{l}$')
        
            
        # plt.plot(l_noise[0], ((BB_noise[0][0]*(l_noise[0]*(l_noise[0]+1))/(2*np.pi))), color='tab:red', linestyle='dotted', label='Noise max')
        # plt.plot(l_noise[1], ((BB_noise[1][0]*(l_noise[1]*(l_noise[1]+1))/(2*np.pi))), color='tab:red', linestyle='dashed', label='Noise min') 

        # plt.plot(l_dust[1], ((BB_dust[1][0]*(l_dust[1]*(l_dust[1]+1))/(2*np.pi))), color='grey', linestyle='dashed', label='Dust d6 max')        
        # plt.plot(l_dust[0], ((BB_dust[0][0]*(l_dust[0]*(l_dust[0]+1))/(2*np.pi))), color='grey', linestyle='dotted', label='Dust d6 min')
       
        # plt.plot(l_dust_d4s2[1], ((BB_dust_d4s2[1][0]*(l_dust_d4s2[1]*(l_dust_d4s2[1]+1))/(2*np.pi))), color='tab:red', linestyle='dashed', label='Dust d4 max')
        # plt.plot(l_dust_d4s2[0], ((BB_dust_d4s2[0][0]*(l_dust_d4s2[0]*(l_dust_d4s2[0]+1))/(2*np.pi))), color='tab:red', linestyle='dotted', label='Dust d4 min')
                 
        
        plt.legend(fontsize='small')
        plt.ylim([1e-4, 10])
        #plt.savefig(Images_Directory+'Residuals_BB_Comparison_5arcmin'+'.pdf')
"""
# class Plot_Patches():
    
#     data_Q = './Output/Outputs_CENN_Q_30arcmin.h5'
#     data_U = './Output/Outputs_CENN_U_30arcmin.h5'
    
#     Number_Of_Initial_Pixels = 256
#     Number_Of_Cutted_Pixels = 8
#     Number_Of_Pixels = 240
    
#     validation_Q = './Output/Validation_Q_30arcmin.h5'
#     validation_U = './Output/Validation_U_30arcmin.h5'
    

    
#     validation_Q = h5py.File(validation_Q, 'r')
#     validation_U = h5py.File(validation_U, 'r')
    
#     data_Q = h5py.File(data_Q, 'r')
#     data_U = h5py.File(data_U, 'r')

#     total_Q = validation_Q["M"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
#     outputs_Q = data_Q["net"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
#     inputs_Q = data_Q["sim"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)

#     total_U = validation_U["M"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
#     outputs_U = data_U["net"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)
#     inputs_U = data_U["sim"][:,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels,Number_Of_Cutted_Pixels:Number_Of_Initial_Pixels-Number_Of_Cutted_Pixels].astype(np.float32)


#     def plotting_the_patches():

#         fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,6))
#         fig.tight_layout(pad=4.0)
        
#         # fig.title.set_text(Channels[i])
    
#         ax1.title.set_text('Input Total (Q)')
#         ax2.title.set_text('CMB (Q)')
#         ax3.title.set_text('Recovered CMB (Q)')
#         ax4.title.set_text('Residuals (Q)')
    
#         fig1=ax1.imshow(total_Q[1,:,:,1]*1e6)
#         fig.colorbar(fig1, ax=ax1, fraction=0.046, pad=0.04)
        
#         fig2=ax2.imshow(inputs_Q[1]*1e6)
#         fig.colorbar(fig2, ax=ax2, fraction=0.046, pad=0.04)
    
#         fig3=ax3.imshow(outputs_Q[1]*1e6)
#         fig.colorbar(fig3, ax=ax3, fraction=0.046, pad=0.04)
        
#         fig4=ax4.imshow(inputs_Q[1]*1e6-outputs_Q[1]*1e6)
#         fig.colorbar(fig4, ax=ax4, fraction=0.046, pad=0.04)
        
#         plt.savefig(Images_Directory+'Patch_Q'+'.pdf')
#         plt.show()
        
        
                
#         fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,6))
#         fig.tight_layout(pad=4.0)
        
#         # fig.title.set_text(Channels[i])
    
#         ax1.title.set_text('Input Total (U)')
#         ax2.title.set_text('CMB (U)')
#         ax3.title.set_text('Recovered CMB (U)')
#         ax4.title.set_text('Residuals (U)')
    
#         fig1=ax1.imshow(total_U[1,:,:,1]*1e6)
#         fig.colorbar(fig1, ax=ax1, fraction=0.046, pad=0.04)
        
#         fig2=ax2.imshow(inputs_U[1]*1e6)
#         fig.colorbar(fig2, ax=ax2, fraction=0.046, pad=0.04)
    
#         fig3=ax3.imshow(outputs_U[1]*1e6)
#         fig.colorbar(fig3, ax=ax3, fraction=0.046, pad=0.04)
        
#         fig4=ax4.imshow(inputs_U[1]*1e6-outputs_U[1]*1e6)
#         fig.colorbar(fig4, ax=ax4, fraction=0.046, pad=0.04)
        
#         plt.savefig(Images_Directory+'Patch_U'+'.pdf')
#         plt.show()
                
Plot_E_B_Power_Spectrum.plotting_E_B_Power_Spectrum_from_CENN()

