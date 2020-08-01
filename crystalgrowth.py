import numpy as np
from skimage.filters import sobel


def binarize(array):
    binary_array = np.zeros((array.shape))
    binary_array[~np.isnan(array)] = 1
    return binary_array

def nucleate(blank_array, nucleation_sites):
    positions = np.random.randint(0,len(blank_array),size=(nucleation_sites,2))
    orientations = np.random.uniform(-np.pi/2,np.pi/2,size=(nucleation_sites))
    blank_array[positions[:,0],positions[:,1]] = orientations
    return blank_array, positions, orientations

def fill_nan_localmean(array,nan_idx):
#     nan_idx = np.column_stack(np.where(np.isnan(array)))
    for pos in nan_idx:
        ylow = max(pos[0] - 1,0)
        yhigh = min(pos[0] + 2,array.shape[0])
        xlow = max(pos[1] - 1,0)
        xhigh = min(pos[1] +2,array.shape[1])
        neighbors = array[ylow:yhigh,xlow:xhigh]
        local_mean = np.mean(neighbors[~np.isnan(neighbors)])
        array[pos[0],pos[1]] = local_mean
    return array

def grow_complete(nucleated_array,growth_anisotropy=2,pi_mutate=np.deg2rad(5),c_mutate=np.deg2rad(5),time=0):
    '''Takes nucleated array and grows crystal until the array is completely filled. 
        Uses a sobel gradient to find crystal edges for more efficient iterating (compared to interating over all not-nans)'''
    grow_array = nucleated_array.copy()
    while np.any(np.isnan(grow_array)==True):
        if time < 2:
            crystal_idx = np.column_stack(np.where(~np.isnan(grow_array)))
        else:
            crystal_binary = binarize(grow_array)
            crystal_sobel = sobel(crystal_binary)
            edges = np.ceil(crystal_binary*crystal_sobel)
            crystal_idx = np.column_stack(np.where(edges==1))
        
        # sobel edge does not work on single pixels on edge of array, use this to fill in last few values
        if len(np.column_stack(np.where(np.isnan(grow_array)))) < 5: 
            remaining_nans = np.column_stack(np.where(np.isnan(grow_array)))
            grow_array = fill_nan_localmean(grow_array,remaining_nans)
            print('almost done')
        for pos in crystal_idx:
            ## crystallization along pi-pi stacking
            director_x = int(round(np.abs(np.cos(grow_array[pos[0],pos[1]]))))
            director_y = int(round(np.abs(np.sin(grow_array[pos[0],pos[1]]))))
            director_xup = (pos[1]+director_x)%len(grow_array)
            director_yup = (pos[0]+director_y)%len(grow_array)
            director_xdn = (pos[1]-director_x)%len(grow_array)
            director_ydn = (pos[0]-director_y)%len(grow_array)
            
            if np.isnan(grow_array[director_yup,director_xup]):
                grow_array[director_yup,director_xup] = grow_array[pos[0],pos[1]] + np.random.uniform(-pi_mutate,pi_mutate)
            if np.isnan(grow_array[director_ydn,director_xdn]):
                grow_array[director_ydn,director_xdn] = grow_array[pos[0],pos[1]] + np.random.uniform(-pi_mutate,pi_mutate)
                
            ## lateral crystal thickening, switch directors
            if time%growth_anisotropy == 0:
                ## for diagonal pi-pi stacking
                if (np.abs(director_x) + np.abs(director_y)) == 2:
                    ## fill in checkerboard pattern
                    if np.isnan(grow_array[director_yup,pos[1]]):
                        grow_array[director_yup,pos[1]] = grow_array[pos[0],pos[1]] + np.random.uniform(-c_mutate,c_mutate)
                    if np.isnan(grow_array[director_ydn,pos[1]]):
                        grow_array[director_ydn,pos[1]] = grow_array[pos[0],pos[1]] + np.random.uniform(-c_mutate,c_mutate)
                    if np.isnan(grow_array[pos[0],director_xup]):
                        grow_array[pos[0],director_xup] = grow_array[pos[0],pos[1]] + np.random.uniform(-c_mutate,c_mutate)
                    if np.isnan(grow_array[pos[0],director_xdn]):
                        grow_array[pos[0],director_xdn] = grow_array[pos[0],pos[1]] + np.random.uniform(-c_mutate,c_mutate)
                        
                ## for pi-pi direction vertical or horizontal
                else:
                    director_xup = (pos[1]+director_y)%len(grow_array)
                    director_yup = (pos[0]+director_x)%len(grow_array)
                    director_xdn = (pos[1]-director_y)%len(grow_array)
                    director_ydn = (pos[0]-director_x)%len(grow_array)
                    if np.isnan(grow_array[director_yup,director_xdn]):
                        grow_array[director_yup,director_xdn] = grow_array[pos[0],pos[1]] + np.random.uniform(-c_mutate,c_mutate)
                    if np.isnan(grow_array[director_ydn,director_xup]):
                        grow_array[director_ydn,director_xup] = grow_array[pos[0],pos[1]] + np.random.uniform(-c_mutate,c_mutate)
            ## constrain orientation to +/- 90 degrees
            grow_array[grow_array > np.pi/2] %= -np.pi/2
            grow_array[grow_array < -np.pi/2] %= np.pi/2
        time += 1
    return grow_array

def grow_partial(nucleated_array, timesteps,growth_anisotropy=2,pi_mutate=np.deg2rad(5),c_mutate=np.deg2rad(5)):
    '''Grows nucleated array for given number of timesteps. Same algorithm as 'grow_complete'
        Useful if you want to nucleate and then grow small crystals surrounded by amorphous matrix'''
    grow_array = nucleated_array.copy()
    for time in np.arange(timesteps):
        if time < 2:
            crystal_idx = np.column_stack(np.where(~np.isnan(grow_array)))
        else:
            crystal_binary = binarize(grow_array)
            crystal_sobel = sobel(crystal_binary)
            edges = np.ceil(crystal_binary*crystal_sobel)
            crystal_idx = np.column_stack(np.where(edges==1))
            
        for pos in crystal_idx:
            ## crystallization along pi-pi stacking
            director_x = int(round(np.abs(np.cos(grow_array[pos[0],pos[1]]))))
            director_y = int(round(np.abs(np.sin(grow_array[pos[0],pos[1]]))))
            director_xup = (pos[1]+director_x)%len(grow_array)
            director_yup = (pos[0]+director_y)%len(grow_array)
            director_xdn = (pos[1]-director_x)%len(grow_array)
            director_ydn = (pos[0]-director_y)%len(grow_array)
            
            if np.isnan(grow_array[director_yup,director_xup]):
                grow_array[director_yup,director_xup] = grow_array[pos[0],pos[1]] + np.random.uniform(-pi_mutate,pi_mutate)
            if np.isnan(grow_array[director_ydn,director_xdn]):
                grow_array[director_ydn,director_xdn] = grow_array[pos[0],pos[1]] + np.random.uniform(-pi_mutate,pi_mutate)
                
            ## lateral crystal thickening, switch directors
            if time%growth_anisotropy == 0:
                ## for diagonal pi-pi stacking
                if (np.abs(director_x) + np.abs(director_y)) == 2:
                    ## fill in checkerboard pattern
                    if np.isnan(grow_array[director_yup,pos[1]]):
                        grow_array[director_yup,pos[1]] = grow_array[pos[0],pos[1]] + np.random.uniform(-c_mutate,c_mutate)
                    if np.isnan(grow_array[director_ydn,pos[1]]):
                        grow_array[director_ydn,pos[1]] = grow_array[pos[0],pos[1]] + np.random.uniform(-c_mutate,c_mutate)
                    if np.isnan(grow_array[pos[0],director_xup]):
                        grow_array[pos[0],director_xup] = grow_array[pos[0],pos[1]] + np.random.uniform(-c_mutate,c_mutate)
                    if np.isnan(grow_array[pos[0],director_xdn]):
                        grow_array[pos[0],director_xdn] = grow_array[pos[0],pos[1]] + np.random.uniform(-c_mutate,c_mutate)
                        
                ## for pi-pi direction vertical or horizontal
                else:
                    director_xup = (pos[1]+director_y)%len(grow_array)
                    director_yup = (pos[0]+director_x)%len(grow_array)
                    director_xdn = (pos[1]-director_y)%len(grow_array)
                    director_ydn = (pos[0]-director_x)%len(grow_array)
                    if np.isnan(grow_array[director_yup,director_xdn]):
                        grow_array[director_yup,director_xdn] = grow_array[pos[0],pos[1]] + np.random.uniform(-np.deg2rad(5),np.deg2rad(5))
                    if np.isnan(grow_array[director_ydn,director_xup]):
                        grow_array[director_ydn,director_xup] = grow_array[pos[0],pos[1]] + np.random.uniform(-np.deg2rad(5),np.deg2rad(5))
            ## constrain orientation to +/- 90 degrees
            grow_array[grow_array > np.pi/2] %= -np.pi/2
            grow_array[grow_array < -np.pi/2] %= np.pi/2
    return grow_array