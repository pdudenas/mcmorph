import numpy as np
import matplotlib.pyplot as plt
import h5py
import pathlib
import numba as nb


@nb.njit()
def create_Rz_parallel_nb(angles):
    rz = np.zeros((*angles.shape,3,3))
    tmp_cos = np.cos(angles)
    tmp_sin = np.sin(angles)
    rz[...,0,0] = tmp_cos
    rz[...,1,1] = tmp_cos
    rz[...,1,0] = tmp_sin
    rz[...,0,1] = -tmp_sin
    rz[...,2,2] = 1
    return rz

@nb.njit()
def create_Ry_parallel_nb(angles):
    ry = np.zeros((*angles.shape,3,3))
    tmp_cos = np.cos(angles)
    tmp_sin = np.sin(angles)
    ry[...,0,0] = tmp_cos
    ry[...,2,2] = tmp_cos
    ry[...,2,0] = -tmp_sin
    ry[...,0,2] = tmp_sin
    ry[...,1,1] = 1
    return ry

@nb.njit()
def create_Rx_parallel_nb(angles):
    rx = np.zeros((*angles.shape,3,3))
    tmp_cos = np.cos(angles)
    tmp_sin = np.sin(angles)
    rx[...,0,0] = 1
    rx[...,1,1] = tmp_cos
    rx[...,2,2] = tmp_cos
    rx[...,2,1] = tmp_sin
    rx[...,1,2] = -tmp_sin
    return rx

#'...ij,...jk->...ik'
@nb.njit(parallel=True)
def nb_einsum(A,B):
    
    shape = A.shape
    A = np.reshape(A,(-1,3,3))
    B = np.reshape(B,(-1,3,3))
    res = np.empty(A.shape)
    
    for s in nb.prange(A.shape[0]):
        for i in range(3):
            for k in range(3):
                acc = 0
                for j in range(3):
                    acc += A[s,i,j]*B[s,j,k]
                res[s,i,k] = acc
                
    res = np.reshape(res,shape)
    return res

@nb.njit(parallel=True)
def nb_einsum2(A,B):
    #A is the 3x3 index of refraction
    #B is the transposed rotation matrix
    shape = B.shape
    B = np.reshape(B,(-1,3,3))
    res = np.empty(B.shape)
    
    for s in nb.prange(B.shape[0]):
        for i in range(3):
            for k in range(3):
                acc = 0
                for j in range(3):
                    acc += A[i,j]*B[s,j,k]
                res[s,i,k] = acc
                
    res = np.reshape(res,shape)
    return res


@nb.njit()
def create_Rzyz_parallel(phi, theta, psi):
    rz1 = create_Rz_parallel_nb(phi)
    ry1 = create_Ry_parallel_nb(theta)
    rz2 = create_Rz_parallel_nb(psi)
    r_zyz = nb_einsum(rz2,nb_einsum(ry1,rz1))
    return r_zyz

@nb.njit()
def rotate_n(n,rotmat):
    ### R@n@R.T
    
    shape = rotmat.shape
    rotmat = np.reshape(rotmat,(-1,3,3))
    res1 = np.empty(shape).astype(np.complex128)
    res2 = np.empty(shape).astype(np.complex128)
    
    for s in nb.prange(rotmat.shape[0]):
        for i in range(3):
            for k in range(3):
                acc = 0
                for j in range(3):
                    acc += n[i,j]*rotmat[s,k,j]
                res1[s,i,k] = acc
                
    for s in nb.prange(rotmat.shape[0]):
        for i in range(3):
            for k in range(3):
                acc = 0
                for j in range(3):
                    acc += rotmat[s,i,j]*res1[s,j,k]
                res2[s,i,k] = acc
                
    res2 = np.reshape(res2,shape)
    return res2


def calc_polarization(n, n_rotated, S, Evec):
    n_iso = np.mean(n)
    tensor_iso = np.zeros((3,3),dtype=complex)
    np.fill_diagonal(tensor_iso,n_iso)
    p_iso = (1-S)*1/4/np.pi*(tensor_iso@tensor_iso-np.identity(3))@Evec
    p_align =  S*1/4/np.pi*(n_rotated@n_rotated-np.identity(3))@Evec
    return p_iso + p_align


def check_NumMat(f, morphology_type):
    morphology_num = f['Morphology_Parameters/NumMaterial'][()]
    
    if morphology_type == 0:
        num_mat = 0
        while f'Euler_Angles/Mat_{num_mat + 1}_Vfrac' in f.keys():
            num_mat +=1
    elif morphology_type == 1:
        num_mat = 0
        while f'Vector_Morphology/Mat_{num_mat + 1}_unaligned' in f.keys():
            num_mat +=1
    
    assert morphology_num==num_mat, 'Number of materials does not match manual count of materials. Recheck hdf5'
    
    return num_mat

def readH5_Euler(filename):
    # read in vector morphology
    with h5py.File(filename,'r') as f:
        num_mat = check_NumMat(f,morphology_type=0)
        
        ds = f['Euler_Angles/Mat_1_Vfrac'][()]
        PhysSize = f['Morphology Parameters/PhysSize'][()]
        
        Vfrac = np.zeros((num_mat,*ds.shape))
        S = Vfrac.copy()
        theta = Vfrac.copy()
        psi = Vfrac.copy()

        #'Mat_1_Psi', 'Mat_1_S', 'Mat_1_Theta', 'Mat_1_Vfrac'

        for i in range(0, num_mat):
            Vfrac[i,:,:,:] = f[f'Euler_Angles/Mat_{i+1}_Vfrac']
            S[i,:,:,:] = f[f'Euler_Angles/Mat_{i+1}_S']
            theta[i,:,:,:] = f[f'Euler_Angles/Mat_{i+1}_Theta']
            psi[i,:,:,:] = f[f'Euler_Angles/Mat_{i+1}_Psi']
    
    return Vfrac, S, theta, psi, PhysSize

def calc_n(file):
    with open(file,'r') as f:
        E = []
        beta_perp = []
        beta_para = []
        delta_perp = []
        delta_para = []
        for line in f:
            if line.startswith('Energy = '):
                E.append(float(line.split(' = ')[1][:-2]))
            if line.startswith('BetaPerp'):
                beta_perp.append(float(line.split(' = ')[1][:-2]))
            if line.startswith('BetaPara'):
                beta_para.append(float(line.split(' = ')[1][:-2]))
            if line.startswith('DeltaPara'):
                delta_para.append(float(line.split(' = ')[1][:-2]))
            if line.startswith('DeltaPerp'):
                delta_perp.append(float(line.split(' = ')[1][:-2]))
    E = np.array(E)
    n = np.zeros((len(E),3,3),dtype=complex)
    for i in range(len(E)):
        n[i,0,0] = np.complex(1-delta_para[i],beta_para[i])
        n[i,1,1] = np.complex(1-delta_perp[i],beta_perp[i])
        n[i,2,2] = np.complex(1-delta_perp[i],beta_perp[i])
    return E, n


def read_materials(material_path):
    try:
        material_path.glob('Mat*.txt')
    except AttributeError:
        pathlib_path = pathlib.Path(material_path)

    file_list = sorted(list(pathlib_path.glob('Mat*.txt')))
    all_n = []
    all_E = []
    for file in file_list:
        E, n = calc_n(file)
        all_n.append(n)
        all_E.append(E)

    return all_E, all_n
