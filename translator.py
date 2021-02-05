import h5py
import datetime
import numpy as np
import glob
import matplotlib.pyplot as plt
import xarray as xr
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from skimage.transform import downscale_local_mean

### file loader(s) ###
def load_npy(filename,downscale=1):
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
def dark_field(ss, data):
    '''Reconstructs dark field and returns as an xarray DataArray'''

    reshape_factor = int(np.sqrt(len(data)))
    x = y = np.arange(0,ss*reshape_factor,ss)
    df_data = np.sum(data,axis=(1,2))
    df_data = np.reshape(df_data,(reshape_factor,reshape_factor))
    df_data_array = xr.DataArray(df_data,dims=['x','y'],coords=dict(x=x,y=y))
    return df_data_array

### fitting functions ###
def powerlaw(x, a, b, c):
    return a*x**(-b) + c

def powerlaw_gauss(x, a1, a2, a3, b1, b2, b3):
    return a1*x**(-a2) + a3 + b1*np.exp(-(x-b2)**2/b3**2)

def gauss_1(x,a,b,c,d):
    return a*np.exp(-(x-b)**2/c**2) + d

def gauss_2(x, a1, b1, c1, a2, b2, c2, d):
    return (a1*np.exp(-(x-b1)**2/c1**2) +
            a2*np.exp(-(x-b2)**2/c2**2) + d)

def gauss_3(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, d):
    return (a1*np.exp(-(x-b1)**2/c1**2) +
            a2*np.exp(-(x-b2)**2/c2**2) +
            a3*np.exp(-(x-b3)**2/c3**2) + d)


### 2D fitting ###
# def _gauss_2d(M, a, b_x, b_y, c_x, c_y, d):
#     x, y = M
#     g = a*np.exp(-(x-b_x)**2/c_x**2 - (y-b_y)**2/c_y**2) + d
#     return g.ravel()

# def gauss_2d(x, y , a, b_x, b_y, c_x, c_y, d):
#     g = a*np.exp(-(x-b_x)**2/c_x**2 - (y-b_y)**2/c_y**2) + d
#     return g

# ydata = data_sub[(70+120*128),5:,:]
# X, Y = np.meshgrid(chi,q[5:])
# xdata = np.vstack((X.ravel(), Y.ravel()))
# init_guess = [20,0,1.7,5,5,0]
# init_guess[1] = peaks[70+120*128][0]
# bounds = ([0,-90,1.5,0,0,-10],[80,90,2,20,5,10])
# popt, pcov = curve_fit(_gauss_2d, xdata, ydata.ravel(), p0=init_guess,bounds=bounds)


### fitting routines ###
def subtract_powerlaw(q, data, qmin_exclude=1, qmax_exclude=2.2):
    '''subtracts powerlaw from every pixel'''
    data_sub = np.zeros(data.shape)
#     max_angle = np.zeros((len(data),4))
    qidx = (q < qmin_exclude) | (q > qmax_exclude)
    for i in np.arange(0,len(data)):
        yy = np.mean(data[i,:,:],axis=1)
        popt, pcov = curve_fit(powerlaw,q[qidx],yy[qidx])
        data_sub[i,:,:] = data[i,:,:] - powerlaw(q,*popt)[:,None]
    return data_sub


def peak_finder(q, chi, data,qmin_exclude=1.5, qmax_exclude=2, prominence=10):
    '''estimates how many gaussian peaks to fit
        and initial guesses for their starting position in chi'''
    peaks = []
    num_peaks = np.zeros(len(data))
    data_1d = np.empty((len(data),len(chi)))
    q_idx = (q > qmin_exclude) & (q < qmax_exclude)
    for i in np.arange(0,len(data)):
        data_1d[i,:] = np.mean(data[i,q_idx,:],axis=0)
        peak_pos, _ = find_peaks(data_1d[i,:],prominence=prominence)
        peaks.append(chi[peak_pos])
        num_peaks[i] = len(peak_pos)
    return peaks, num_peaks, data_1d


def fit_peaks(chi, data_1d, peaks, num_peaks):
    '''fits 1, 2, or 3 gaussians to intensity vs. chi. 
        returns optimized params as a list of variable length arrays'''
    gauss_params = []
    for i in np.arange(0,len(data_1d)):
        ydata = data_1d[i,:]
        if num_peaks[i] == 0:
            p0 = (10,-90,10,10)
            bounds = ([0,-90,0,0],[50,90,30,50])
            try:
                popt, _ = curve_fit(gauss_1,chi,ydata,p0=p0,bounds=bounds)
                gauss_params.append(popt)
            except RuntimeError:
                gauss_params.append([])
        elif num_peaks[i] == 1:
            p0 = (10,peaks[i][0],10,10)
            bounds = ([0,-90,0,0],[50,90,30,50])
            try:
                popt, _ = curve_fit(gauss_1,chi,ydata,p0=p0,bounds=bounds)
                gauss_params.append(popt)
            except RuntimeError:
                gauss_params.append([])
        elif num_peaks[i] == 2:
            p0 = (10,peaks[i][0],10,10,peaks[i][1],10,2)
            bounds = ([0,-90,0,0,-90,0,-10],[50,90,30,50,90,30,50])
            try:
                popt, _ = curve_fit(gauss_2,chi,ydata,p0=p0,bounds=bounds)
                gauss_params.append(popt)
            except RuntimeError:
                gauss_params.append([])
        elif num_peaks[i] == 3:
            p0 = (10,peaks[i][0],10,10,peaks[i][1],10,10,peaks[i][2],10,2)
            bounds = ([0,-90,0,0,-90,0,0,-90,0,-10],[50,90,30,50,90,30,50,90,30,50])
            try:
                popt, _ = curve_fit(gauss_3,chi,ydata,p0=p0,bounds=bounds)
                gauss_params.append(popt)
            except RuntimeError:
                gauss_params.append([])
        else:
            print('More than 3 peaks, adjust prominence value')
            gauss_params.append([])

    return gauss_params

def list_to_array(data_list):
    peak_pos = np.empty((len(data_list),3))
    peak_int = np.empty((len(data_list),3))
    peak_fwhm = np.empty((len(data_list),3))
    bckgrnd = np.empty(len(data_list))
    peak_pos[:] = np.nan
    peak_int[:] = np.nan
    peak_fwhm[:] = np.nan
    bckgrnd[:] = np.nan
    for i,array in enumerate(data_list):
        if len(array) == 4:
            peak_int[i,0] = array[0]
            peak_pos[i,0] = array[1]
            peak_fwhm[i,0] = array[2]
            bckgrnd[i] = array[3]
        elif len(array) == 7:
            peak_int[i,0] = array[0]
            peak_pos[i,0] = array[1]
            peak_fwhm[i,0] = array[2]
            peak_int[i,1] = array[3]
            peak_pos[i,1] = array[4]
            peak_fwhm[i,1] = array[5]
            bckgrnd[i] = array[6]
        elif len(array) == 10:
            peak_int[i,0] = array[0]
            peak_pos[i,0] = array[1]
            peak_fwhm[i,0] = array[2]
            peak_int[i,1] = array[3]
            peak_pos[i,1] = array[4]
            peak_fwhm[i,1] = array[5]
            peak_int[i,2] = array[3]
            peak_pos[i,2] = array[4]
            peak_fwhm[i,2] = array[5]
            bckgrnd[i] = array[6]
    return peak_pos, peak_int, peak_fwhm, bckgrnd

def implot(array_1d,**kwargs):
    reshape_val = int(np.sqrt(len(array_1d)))
    plt.imshow(np.reshape(array_1d,(reshape_val,reshape_val)),**kwargs)




### HDF5 Writer ###

def write_orientation_hdf5(orientation_array,PhysSize,fname,author='PJD',polymer='PBTTT'):
    NumXY = orientation_array.shape[0]
    xvec = np.cos(orientation_array)
    yvec = np.sin(orientation_array)
    s1 = np.zeros((1,NumXY,NumXY,3))
    s2 = s1.copy()
    s1[0,:,:,1] = xvec
    s1[0,:,:,2] = yvec
    phi1 = 1 - np.sqrt(xvec**2+yvec**2)
    phi1_out = phi1[np.newaxis,:,:]
    phi2_out = np.zeros((1,NumXY,NumXY))

    print(f'--> Marking {fname}')
    with h5py.File(fname,'w') as f:
            f.create_dataset("igor_parameters/igormaterialnum",data=2.0)
            f.create_dataset("vector_morphology/Mat_1_alignment",data=s1,compression='gzip',compression_opts=9)
            f.create_dataset("vector_morphology/Mat_2_alignment",data=s2,compression='gzip',compression_opts=9)
            f.create_dataset("vector_morphology/Mat_1_unaligned",data=phi1_out,compression='gzip',compression_opts=9)
            f.create_dataset("vector_morphology/Mat_2_unaligned",data=phi2_out,compression='gzip',compression_opts=9)

            f.create_dataset('morphology_variables/creation_date', data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            f.create_dataset('morphology_variables/film_normal', data=[1,0,0])
            f.create_dataset('morphology_variables/morphology_creator', data=author)
            f.create_dataset('morphology_variables/name', data=author)
            f.create_dataset('morphology_variables/version', data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            f.create_dataset('morphology_variables/voxel_size_nm', data=PhysSize)

            f.create_dataset('igor_parameters/igorefield', data="0,1")
            f.create_dataset('igor_parameters/igormaterials', data=f"{polymer},vac")
            f.create_dataset('igor_parameters/igormodelname', data="4DSTEM")
            f.create_dataset('igor_parameters/igormovie', data=0)
            f.create_dataset('igor_parameters/igorname', data="perp001")
            f.create_dataset('igor_parameters/igornum', data=0)
            f.create_dataset('igor_parameters/igorparamstring', data="n/a")
            f.create_dataset('igor_parameters/igorpath', data="n/a")
            f.create_dataset('igor_parameters/igorrotation', data=0)
            f.create_dataset('igor_parameters/igorthickness', data=1)
            f.create_dataset('igor_parameters/igorvoxelsize', data=1)
    return fname
