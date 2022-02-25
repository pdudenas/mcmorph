import numpy as np
import matplotlib.pyplot as plt
import h5py
import pathlib

def create_rotmat(array):
    ''' Creates a rotation matrix that rotates the unit vector [1,0,0]
    into array. Only necessary for vector morphologies in Cy-RSoXS '''
    rotmat = np.zeros((3,3),dtype=complex)
    sx = array[0]
    sy = array[1]
    sz = array[2]
    mag = np.sqrt(sx**2 + sy**2 + sz**2)

    rotmat[0,0] = sx/mag
    rotmat[0,1] = -sy/mag
    rotmat[0,2] = -sz/mag

    rotmat[1,0] = sy/mag
    rotmat[1,1] = (sz**2 + sx*sy**2/mag)/(sy**2 + sz**2)
    rotmat[1,2] = (sy*sz*(-1 + sx/mag))/(sy**2+sz**2)

    rotmat[2,0] = sz/mag
    rotmat[2,1] = sy*sz*(-1 + sx/mag)/(sy**2 + sz**2)
    rotmat[2,2] = (sy**2 + sx*sz**2/mag)/(sy**2 + sz**2)

    return rotmat


def rotate_n(n, rotmat):
    return np.dot(rotmat,np.dot(n,rotmat.T))


def calc_polarization(n_rotated, Evec):
    return 1/4/np.pi*(n_rotated@n_rotated-np.identity(3))@Evec



def read_hdf5(filename):
    with h5py.file(filename,'r') as h5:
        num_mat = int(h5['igor_parameters/igormaterialnum'][()])
        voxel_size = h5['morphology_variables/voxel_size_nm'][()]
        mat1_s = h5['vector_morphology/Mat_1_alignment'][()]
        shape = mat1_s.shape
        s_matrix = np.zeros((num_mat,*shape))
        for i in range(num_mat):
            s_matrix[i,:,:,:,:] = h5[f'vector_morphology/Mat_{i+i}_alignment'][()]
    return s_matrix, voxel_size

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
