import h5py
import datetime
import numpy as np
import glob
import matplotlib.pyplot as plt
import xarray as xr
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from skimage.transform import downscale_local_mean



class translator():

    def __init__(self):
        pass

    ### file loader(s) ###
    def load_npy(self,filename,downscale=1):
        '''loads .npy files and extracts q and chi from filename.
            Based on Salleo group naming convention'''
        data = np.load(filename)
        # string parsing -- example of filename: '...ss=10nm 128x128...__q 0.50 3.00 0.02 a -90.00 90.00 2.00.npy'
        ss_idx = filename.find('ss=')
        if ss_idx == -1:
            ss_idx = filename.find('ss')
            ss = float(filename[(ss_idx+2):(ss_idx+4)])
        else:
            ss = float(filename[(ss_idx+3):(ss_idx+5)])
        qchi_string = filename.split('__')[1][:-4].split(' ')
        qmin = float(qchi_string[1])
        qmax = float(qchi_string[2])
        qinc = float(qchi_string[3])
        chi_min = float(qchi_string[5])
        chi_max = float(qchi_string[6])
        chi_inc = float(qchi_string[7])
        q = np.arange(qmin, qmax, qinc*downscale)
        chi = np.arange(chi_min, chi_max, chi_inc*downscale)

        #optionally downscale image by some integer factor
        if downscale > 1:
            data = downscale_local_mean(data,(1,downscale,downscale))
        return ss, q, chi, data

    ### reconstruction ###
    def dark_field(self,ss, data):
        '''Reconstructs dark field and returns as an xarray DataArray

        TO DO: implement option to select annular region of q'''

        reshape_factor = int(np.sqrt(len(data)))
        x = y = np.arange(0,ss*reshape_factor,ss)
        df_data = np.sum(data,axis=(1,2))
        df_data = np.reshape(df_data,(reshape_factor,reshape_factor))
        df_data_array = xr.DataArray(df_data,dims=['x','y'],coords=dict(x=x,y=y))
        return df_data_array

    def implot(self,array_1d,**kwargs):
        reshape_val = int(np.sqrt(len(array_1d)))
        plt.imshow(np.reshape(array_1d,(reshape_val,reshape_val)),**kwargs)


    ### fitting functions ###
    def powerlaw(self,x, a, b, c):
        return a*x**(-b) + c

    def gaussian(self,x, a, b, c,d):
        return a*np.exp(-(x-b)**2/c**2) + d

    def _gaussian(self,x,*args):
        arr = np.zeros(len(x))
        for i in range(len(args)//4):
            arr += self.gaussian(x,*args[i*4:i*4+4])
        return arr



    ### fitting routines ###
    def subtract_powerlaw(self, q, data, qmin_exclude=1, qmax_exclude=2.2):
        '''subtracts powerlaw from every pixel'''
        data_sub = np.zeros(data.shape)
    #     max_angle = np.zeros((len(data),4))
        qidx = (q < qmin_exclude) | (q > qmax_exclude)
        for i in np.arange(0,len(data)):
            yy = np.mean(data[i,:,:],axis=1)
            popt, pcov = curve_fit(self.powerlaw,q[qidx],yy[qidx])
            data_sub[i,:,:] = data[i,:,:] - self.powerlaw(q,*popt)[:,None]
        return data_sub


    def peak_finder(self,q, chi, data,qmin_exclude=1.5, qmax_exclude=2, prominence=10,distance=np.pi/4):
        '''estimates how many gaussian peaks to fit
            and initial guesses for their starting position in chi'''
        peaks = []
        num_peaks = np.zeros(len(data))
        data_1d = np.empty((len(data),len(chi)))
        q_idx = (q > qmin_exclude) & (q < qmax_exclude)
        for i in np.arange(0,len(data)):
            data_1d[i,:] = np.nanmean(data[i,q_idx,:],axis=0)
            peak_pos, _ = find_peaks(data_1d[i,:],prominence=prominence,distance=distance)
            peaks.append(chi[peak_pos])
            num_peaks[i] = len(peak_pos)
        return peaks, num_peaks, data_1d


    def fit_peaks(self,chi,data_1d,peaks,num_peaks):
        opt_params = []
        num_peaks = num_peaks.astype(int)
        for i in range(len(data_1d)):
            if num_peaks[i] > 0:
                y = data_1d[i,:]
                guess_prms = np.zeros(4*num_peaks[i])
                low_bound = guess_prms.copy()
                high_bound = guess_prms.copy()

                for j in range(num_peaks[i]):
                    ''' gaussian = a*np.exp(-(x-b)**2/c**2) + d '''

                    guess_prms[j*4] = 200           # a
                    guess_prms[j*4+1] = peaks[i][j] # b
                    guess_prms[j*4+2] = 10          # c
                    guess_prms[j*4+3] = 10          # d

                    low_bound[j*4] = 0              # a
                    low_bound[j*4+2] = 0            # c
                    low_bound[j*4+3] = 0          # d

                    high_bound[j*4] = 5000          # a
                    high_bound[j*4+2] = 40          # c
                    high_bound[j*4+3] = 50          # d

                    # set bounds on peak position
                    if peaks[i][j] > 0:
                        low_bound[j*4+1] = peaks[i][j]*0.75
                        high_bound[j*4+1] = min(peaks[i][j]*1.25,90)
                    elif peaks[i][j] < 0:
                        low_bound[j*4+1] = max(peaks[i][j]*1.25,-90)
                        high_bound[j*4+1] = peaks[i][j]*0.75
                    elif peaks[i][j] == 0:
                        low_bound[j*4+1] = -20
                        high_bound[j*4+1] = 20

                popt, _ = curve_fit(self._gaussian,chi,y,guess_prms,bounds=(low_bound,high_bound),maxfev=5000)
                opt_params.append(popt)
            else:
                opt_params.append([])

        return opt_params

    def list_to_array(self, param_list,num_peaks=None):
        ''' Converts list of fit parameters into 4 NxNxM arrays where

            N : image dimensions
            M : max number of peaks found in image
            4 arrays are peak position, intensity, fwhm, and background '''

        if num_peaks is None:
            num_peaks = np.zeros(len(param_list))
            for i, pixel in enumerate(param_list):
                num_peaks[i] = len(pixel)

        depth_val = int(np.amax(num_peaks))

        num_peaks = num_peaks.astype(int)
        peak_pos = np.empty((len(param_list),depth_val))
        peak_pos[:] = np.nan
        peak_int = peak_pos.copy()
        peak_fwhm = peak_pos.copy()
        bckgrnd = peak_pos.copy()

        for i,pixel in enumerate(param_list):
            if num_peaks[i] > 0:
                pixel = np.reshape(pixel,(-1,4))
                peak_int[i,:num_peaks[i]] = pixel[:,0]
                peak_pos[i,:num_peaks[i]] = pixel[:,1]
                peak_fwhm[i,:num_peaks[i]] = pixel[:,2]
                bckgrnd[i,:num_peaks[i]] = pixel[:,3]

        return peak_int, peak_pos,peak_fwhm, bckgrnd
