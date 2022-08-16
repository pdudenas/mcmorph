import numpy as np
import numba as nb
from numba.typed import List
from numba import prange

@nb.njit()
def grow_fiber_core(fiber_length,mu,sigma,fiberspace_size):
    ''' Creates a fiber core with a
    uniform distribution about an overall director

    mu - overall director direction
    sigma - standard deviation of normal distribution
    TO DO: code in other distributions'''

    xylist = np.empty((fiber_length,2))

    ypos = np.random.randint(0,fiberspace_size)
    xpos = np.random.randint(0,fiberspace_size)
    xylist[0,0] = ypos
    xylist[0,1] = xpos
    # grow fiber for given amount of steps
    for i in range(1,fiber_length):
        # perturb fiber growth direction
        director_perturb = np.random.normal(mu, sigma)
        xstep = np.cos(director_perturb)
        ystep = np.sin(director_perturb)
        xpos = (xpos + xstep)
        ypos = (ypos + ystep)

        xylist[i,1] = xpos
        xylist[i,0] = ypos

    return xylist

@nb.njit()
def grow_sino_fiber_core(fiber_length,mu,sigma,amplitude,period,fiberspace_size,amplitude_sigma = 0.1, period_sigma=0.1):
    ''' Uses a list based method to propagate a fiber with a
    uniform distribution about an overall director

    mu - overall director direction
    sigma - standard deviation of normal distribution

    TO DO: code in other distributions'''
    xylist = np.empty((fiber_length,2))

    ypos = np.random.randint(0,fiberspace_size)
    xpos = np.random.randint(0,fiberspace_size)
    xylist[0,0] = ypos
    xylist[0,1] = xpos

    amplitude += np.random.normal(0,amplitude*amplitude_sigma)
    period += np.random.normal(0,period*period_sigma)

    # grow fiber for given amount of steps
    for i in range(1,fiber_length):
        # perturb fiber growth direction
        director_perturb = amplitude*(np.sin(i*np.pi/period) + np.random.normal(mu,sigma))
        director = director_perturb
        if director > np.pi/2+mu:
            director = np.pi/2 + mu
        elif director < -np.pi/2 + mu:
            director = -np.pi/2 + mu
        xstep = np.cos(director)
        ystep = np.sin(director)
        xpos = (xpos + xstep)
        ypos = (ypos + ystep)


        xylist[i,1] = xpos
        xylist[i,0] = ypos

    return xylist

@nb.njit()
def smooth_fiber_core(xylist, avg_width):
    ''' Smooths out grow_fiber_core results using a
    moving window average of width avg_width.
    Used for axial director calculation '''

    xysmooth = np.zeros(xylist.shape)
    iter_val = xylist.shape[0]
    
    for j in range(0,avg_width):
        xysmooth[j,0] = np.mean(xylist[0:(2*j+1),0])
        xysmooth[j,1] = np.mean(xylist[0:(2*j+1),1])
    for j in range(avg_width,iter_val-avg_width):
        min_idx = j-avg_width
        max_idx = j+avg_width+1
        xysmooth[j,0] = np.mean(xylist[min_idx:max_idx,0])
        xysmooth[j,1] = np.mean(xylist[min_idx:max_idx,1])
    for j in range(iter_val-avg_width,iter_val):
        xysmooth[j,0] = np.mean(xylist[(j-avg_width):,0])
        xysmooth[j,1] = np.mean(xylist[(j-avg_width):,1])

    return xysmooth

@nb.njit()
def axial_director(xylist):
    ''' Calculates axial director of fiber core at each point '''
    theta = np.zeros(xylist.shape[0])
    diff_y = np.diff(xylist[:,0].ravel())
    diff_x = np.diff(xylist[:,1].ravel())
    theta[:-1] = np.arctan(diff_y/diff_x)
    theta[-1] = theta[-2]

    return theta


@nb.njit()
def fill_loop(tuple1,fiber_orientation):
    for i in range(len(tuple1[0])):
        y = tuple1[0][i]
        x = tuple1[1][i]
        fiber_orientation[y,x] = np.nanmean(fiber_orientation[(y-1):(y+2),(x-1):(x+2)])
        
    return fiber_orientation

@nb.njit()
def create_disk_coords(radius):
    """Generates a flat, disk-shaped footprint.
    A pixel is within the neighborhood if the Euclidean distance between
    it and the origin is no greater than radius.
    Parameters
    ----------
    radius : int
        The radius of the disk-shaped footprint.
    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.
    Returns
    -------
    footprint : ndarray
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
    """
    L = np.arange(-radius, radius + 1)
    # disk_out = np.zeros(shape=(L.size,L.size),dtype=dtype)
    disk_coords = List()
    for j in range(L.size):
        for i in range(L.size):
            if (L[j]**2 + L[i]**2 <= radius**2):
                disk_coords.append((L[j],L[i]))
                
    return disk_coords

@nb.njit()
def select_dilate_nb(xylist, coords, size):
    output_array = np.zeros((size,size),dtype=np.uint8)
    for i in range(xylist.shape[0]):
        for val_pair in coords:
            ycoord = int((val_pair[0]+round(xylist[i,0]))%size)
            xcoord = int((val_pair[1]+round(xylist[i,1]))%size)
            output_array[ycoord,xcoord] = 1
    return output_array

@nb.njit()
def fill_nans(xylist, fiber_orientation, width, theta):
    # fill in missing holes using fiber from disk structuring element
    disk_coords = create_disk_coords(width)
    fiber_center = select_dilate_nb(xylist,disk_coords,fiber_orientation.shape[0])
    fill1 = np.nonzero((fiber_center==1) & (np.isnan(fiber_orientation)))

    fiber_orientation = fill_loop(fill1,fiber_orientation)
    
    # get the first bit of the fiber that wasn't caught with nanmean
    fill2_y, fill2_x = np.nonzero((fiber_center==1) & (np.isnan(fiber_orientation)))
    for i in range(len(fill2_x)):
        fiber_orientation[fill2_y[i],fill2_x[i]] = theta[0]

    # replace nans with zeros
    fill_nan_y, fill_nan_x = np.nonzero(np.isnan(fiber_orientation))
    for i in range(len(fill_nan_y)):
        fiber_orientation[fill_nan_y[i],fill_nan_x[i]] = 0
    
    
    return fiber_center, fiber_orientation

@nb.njit()
def expand_loop(xylist, theta, r, fiberspace_size, fiber_orientation):
    fiber_count = np.zeros(fiber_orientation.shape)
    for i in range(xylist.shape[0]):
        tmp_dx = xylist[i,1] + r*np.cos(np.pi/2+theta[i])
        tmp_dy = xylist[i,0] + r*np.sin(np.pi/2+theta[i])
        for j in range(len(tmp_dx)):
            dx = round(tmp_dx[j])%fiberspace_size
            dy = round(tmp_dy[j])%fiberspace_size
            fiber_count[dy,dx] += 1.0
            fiber_orientation[dy,dx] += theta[i]
    
    fiber_orientation /= fiber_count
    
    return fiber_orientation

@nb.njit()
def expand_fiber(xylist,theta,width,fiberspace_size):
        ''' assign orientation orthogonal to fiber core out to some width '''
        
        # create fiber_count, fiber_orientation, and fiber_center
        fiber_orientation = np.zeros((fiberspace_size,fiberspace_size))
        
        width = max(width,1.0)
        r = np.linspace(-width,width,5*round(width))

        fiber_orientation = expand_loop(xylist, theta, r, fiberspace_size, fiber_orientation)
        
        fiber_center, fiber_orientation = fill_nans(xylist,
                                                    fiber_orientation,
                                                    width,theta)

        return fiber_center, fiber_orientation

@nb.njit()
def grow_fibers(fiber_number,fiber_length,director,sigma1,sigma2,fiber_width,avg_width,
                    fiberspace_size,fiber_width_sigma=0):
    ''' inputs-
        fiber_number: number of fibers to grow
        director: overall alignment direction of fibers
        sigma1: standard deviation on the normal distribution that determines
                each individual fibers direction
        sigma2: standard deviation on normal distribution that perturbs an
                invidual fibers direction as it grows
        fiber_width: radius of fiber short axis in pixels
        avg_width: smoothing window size
        fiberspace_size: size of the square array
        fiber_width_sigma: fiber width normal distribution standard deviation
        fiber_length: number of steps to grow fiber. if None, defaults to
                        fiberspace_size
    '''
    fiberspace = np.zeros((fiberspace_size,fiberspace_size),dtype=np.float64)
    alignment_space = np.zeros((fiberspace_size,fiberspace_size),dtype=np.float64)

    all_cores = List()
    for i in prange(fiber_number):
        mu = np.random.normal(director,sigma1)

        xylist = grow_fiber_core(fiber_length,mu,sigma2,fiberspace_size)
        all_cores.append(xylist)
        xysmooth = smooth_fiber_core(xylist,avg_width)
        theta = axial_director(xysmooth)
        fiber_count, fiber_orientation = expand_fiber(xylist,theta,np.random.normal(fiber_width,fiber_width_sigma),fiberspace_size)
        fiberspace += fiber_count
        alignment_space += fiber_orientation

    alignment_space /= fiberspace

    return fiberspace, alignment_space, all_cores

@nb.njit()
def grow_sino_fibers(fiber_number,fiber_length,director,sigma1,sigma2,fiber_width,avg_width,
                fiberspace_size,amplitude, period, fiber_width_sigma=0,amplitude_sigma=0.1, period_sigma=0.1):
    ''' inputs-
        fiber_number: number of fibers to grow
        director: overall alignment direction of fibers
        sigma1: standard deviation on the normal distribution that determines
                each individual fibers direction
        sigma2: standard deviation on normal distribution that perturbs an
                invidual fibers direction as it grows
        fiber_width: radius of fiber short axis in pixels
        avg_width: smoothing window size
        fiberspace_size: size of the square array
        fiber_width_sigma: fiber width normal distribution standard deviation
        fiber_length: number of steps to grow fiber. if None, defaults to
                        fiberspace_size
    '''
    fiberspace = np.zeros((fiberspace_size,fiberspace_size),dtype=np.float64)
    alignment_space = np.zeros((fiberspace_size,fiberspace_size),dtype=np.float64)

    all_cores = List()
    for i in prange(fiber_number):
        mu = np.random.normal(director,sigma1)

        xylist = grow_sino_fiber_core(fiber_length, mu, sigma2,
                                        amplitude, period, fiberspace_size,
                                        amplitude_sigma=amplitude_sigma, period_sigma=period_sigma)
        all_cores.append(xylist)
        xysmooth = smooth_fiber_core(xylist,avg_width)
        theta = axial_director(xysmooth)
        fiber_count, fiber_orientation = expand_fiber(xylist,theta,np.random.normal(fiber_width,fiber_width_sigma),fiberspace_size)
        fiberspace += fiber_count
        alignment_space += fiber_orientation

    alignment_space /= fiberspace

    return fiberspace, alignment_space, all_cores

class fibergrowth():

    def __init__(self):
        pass

    
    def grow_fiber_drfield(self,fiber_length,director_field,sigma):
        ''' Uses a list based method to propagate a fiber through a
        pre-generated director field
        director_field - pre-generated director field
        sigma - standard deviation of normal distribution

        TO DO: code in other distributions'''
        ymax, xmax = director_field.shape
        xlist = []
        ylist = []
        init_pos = self.rng.integers(0,ymax,2)
        xpos = init_pos[1]
        ypos = init_pos[0]
        ylist.append(init_pos[0])
        xlist.append(init_pos[1])
        # grow fiber for given amount of steps
        for i in np.arange(0,fiber_length):
            # perturb fiber growth direction
            director_perturb = director_field[int(ypos)%ymax,int(xpos)%xmax] + self.rng.normal(0, sigma)
            director = director_perturb
            xstep = np.cos(director)
            ystep = np.sin(director)
            xpos = (xpos + xstep)
            ypos = (ypos + ystep)


            xlist.append(xpos)
            ylist.append(ypos)

        return xlist, ylist


    def grow_fibers(self,fiber_number,director,sigma1,sigma2,fiber_width,avg_width,
                    fiberspace_size,fiber_width_sigma=0, fiber_length=None):
        ''' inputs-
            fiber_number: number of fibers to grow
            director: overall alignment direction of fibers
            sigma1: standard deviation on the normal distribution that determines
                    each individual fibers direction
            sigma2: standard deviation on normal distribution that perturbs an
                    invidual fibers direction as it grows
            fiber_width: radius of fiber short axis in pixels
            avg_width: smoothing window size
            fiberspace_size: size of the square array
            fiber_width_sigma: fiber width normal distribution standard deviation
            fiber_length: number of steps to grow fiber. if None, defaults to
                          fiberspace_size
        '''
        fiberspace = np.zeros((fiberspace_size,fiberspace_size))
        alignment_space = fiberspace.copy()

        # pre-allocate here once, instead of every loop in expand_fiber
        fiber_count = fiberspace.copy()
        fiber_orientation = fiberspace.copy()
        fiber_center = fiberspace.copy()



        if fiber_length is None:
            fiber_length = fiberspace_size


        for i in range(fiber_number):
            mu = self.rng.normal(director,sigma1)

            xlist, ylist = self.grow_fiber_core(fiber_length,mu,sigma2,
                            fiberspace_size)

            xsmooth, ysmooth = self.smooth_fiber_core(xlist,ylist,avg_width)
            theta = self.axial_director(xsmooth,ysmooth)
            fiber_count, fiber_orientation = self.expand_fiber(xlist,
                                                               ylist,
                                                               theta,
                                                               self.rng.normal(fiber_width,fiber_width_sigma),
                                                               fiberspace,
                                                               fiber_count,
                                                               fiber_orientation,
                                                               fiber_center)
            fiberspace += fiber_count
            alignment_space += fiber_orientation

        alignment_space /= fiberspace

        return fiberspace, alignment_space


    def grow_fibers_field(self,fiber_number,director_field,sigma,fiber_width,avg_width,fiber_width_sigma=0, fiber_length=None):
        ''' inputs-
            fiber_number: number of fibers to grow
            director: overall alignment direction of fibers
            sigma1: standard deviation on the normal distribution that determines
                    each individual fibers direction
            sigma2: standard deviation on normal distribution that perturbs an
                    invidual fibers direction as it grows
            fiber_width: radius of fiber short axis in pixels
            avg_width: smoothing window size
            fiberspace_size: size of the square array
            fiber_width_sigma: fiber width normal distribution standard deviation
            fiber_length: number of steps to grow fiber. if None, defaults to
                          fiberspace_size
        '''
        fiberspace = np.zeros(director_field.shape)
        alignment_space = fiberspace.copy()

        # pre-allocate here once, instead of every loop in expand_fiber
        fiber_count = fiberspace.copy()
        fiber_orientation = fiberspace.copy()
        fiber_center = fiberspace.copy()



        if fiber_length is None:
            fiber_length = director_field.shape[0]


        for i in range(fiber_number):

            xlist, ylist = self.grow_fiber_drfield(fiber_length,director_field,sigma)

            xsmooth, ysmooth = self.smooth_fiber_core(xlist,ylist,avg_width)
            theta = self.axial_director(xsmooth,ysmooth)
            fiber_count, fiber_orientation = self.expand_fiber(xlist,
                                                               ylist,
                                                               theta,
                                                               self.rng.normal(fiber_width,fiber_width_sigma),
                                                               fiberspace,
                                                               fiber_count,
                                                               fiber_orientation,
                                                               fiber_center)
            fiberspace += fiber_count
            alignment_space += fiber_orientation

        alignment_space /= fiberspace

        return fiberspace, alignment_space


    def grow_sino_fibers(self,fiber_number,director,sigma1,sigma2,fiber_width,avg_width,
                    fiberspace_size,amplitude, period, fiber_width_sigma=0, fiber_length=None):
        ''' inputs-
            fiber_number: number of fibers to grow
            director: overall alignment direction of fibers
            sigma1: standard deviation on the normal distribution that determines
                    each individual fibers direction
            sigma2: standard deviation on normal distribution that perturbs an
                    invidual fibers direction as it grows
            fiber_width: radius of fiber short axis in pixels
            avg_width: smoothing window size
            fiberspace_size: size of the square array
            fiber_width_sigma: fiber width normal distribution standard deviation
            fiber_length: number of steps to grow fiber. if None, defaults to
                          fiberspace_size
        '''
        fiberspace = np.zeros((fiberspace_size,fiberspace_size))
        alignment_space = fiberspace.copy()

        # pre-allocate here once, instead of every loop in expand_fiber
        fiber_count = fiberspace.copy()
        fiber_orientation = fiberspace.copy()
        fiber_center = fiberspace.copy()



        if fiber_length is None:
            fiber_length = fiberspace_size


        for i in range(fiber_number):
            mu = self.rng.normal(director,sigma1)

            xlist, ylist = self.grow_sino_fiber_core(fiber_length,mu,sigma2,
                            amplitude, period, fiberspace_size)

            xsmooth, ysmooth = self.smooth_fiber_core(xlist,ylist,avg_width)
            theta = self.axial_director(xsmooth,ysmooth)
            fiber_count, fiber_orientation = self.expand_fiber(xlist,
                                                               ylist,
                                                               theta,
                                                               self.rng.normal(fiber_width,fiber_width_sigma),
                                                               fiberspace,
                                                               fiber_count,
                                                               fiber_orientation,
                                                               fiber_center)
            fiberspace += fiber_count
            alignment_space += fiber_orientation

        alignment_space /= fiberspace

        return fiberspace, alignment_space

    def tangent_components(self,scale,t,theta):
        xcomp = 1
        ycomp = scale*np.cos(t*scale)
        zcomp = scale*np.sin(t*scale)
        xout = xcomp*np.cos(theta) - ycomp*np.sin(theta)
        yout = xcomp*np.sin(theta) + ycomp*np.cos(theta)
        return xout, yout, zcomp

    def normal_components(self,scale,t,theta):
        xcomp = 0
        ycomp = -scale**2*np.sin(t*scale)
        zcomp = scale**2*np.cos(t*scale)
        xout = xcomp*np.cos(theta) - ycomp*np.sin(theta)
        yout = xcomp*np.sin(theta) + ycomp*np.cos(theta)
        return xout, yout, zcomp

    def fill_nans_helical(self, fiber_center,tan_comps,norm_comps,width,theta,scale):
        width = max(1,width)
        fiber_center=dilation(fiber_center,disk(width))

        fill_idx = (fiber_center == 1) & (np.isnan(np.sum(tan_comps,axis=2)))
        fill2 = np.where(fill_idx == True)
        for x,y in zip(fill2[1],fill2[0]):
            tan_comps[y,x,:] = np.nanmean(tan_comps[(y-1):(y+2),(x-1):(x+2),:],axis=(0,1))
            norm_comps[y,x,:] = np.nanmean(norm_comps[(y-1):(y+2),(x-1):(x+2),:],axis=(0,1))
        # get the first bit of the fiber that wasn't caught with nanmean
        fill_idx = (fiber_center == 1) & (np.isnan(tan_comps[:,:,0]))
        fill2 = np.where(fill_idx == True)
        tan_comps[fill2[0].astype(int),fill2[1].astype(int),:] = self.tangent_components(scale,0,theta[0])
        norm_comps[fill2[0].astype(int),fill2[1].astype(int),:] = self.normal_components(scale,0,theta[0])

        # replace nans with zeros
        nan_idx = np.isnan(tan_comps)
        tan_comps[nan_idx] = 0
        norm_comps[nan_idx] = 0
        # make sure fiber_center and component arrays are consistent
        fiber_center[tan_comps[:,:,0] != 0] = 1

        return fiber_center, tan_comps, norm_comps

    def map_helical(self,xlist,ylist,theta,width,scale,
                                                fiberspace,
                                                fiber_count,
                                                tan_comps,
                                                norm_comps,
                                                fiber_center):
        # reset arrays
        tan_comps[:] = 0
        norm_comps[:] = 0
        fiber_count[:] = 0
        fiber_center[:] = 0

        # random offset to the helical phase
        offset = self.rng.normal(0,2)

        #
        r = np.linspace(-width,width,4*round(width))

        for i, (x,y) in enumerate(zip(xlist,ylist)):
            centerx = np.round(x)%fiberspace.shape[1]
            centery = np.round(y)%fiberspace.shape[0]
            fiber_center[centery.astype(int),centerx.astype(int)] = 1

            dx = np.round(x + r*np.cos(np.pi/2+theta[i]))%fiberspace.shape[1]
            dy = np.round(y + r*np.sin(np.pi/2+theta[i]))%fiberspace.shape[0]
            fiber_count[dy.astype(int),dx.astype(int)] += 1

            tan_comps[dy.astype(int),dx.astype(int),:] = self.tangent_components(scale,i+offset,theta[i])
            norm_comps[dy.astype(int),dx.astype(int),:] = self.normal_components(scale,i+offset,theta[i])

        tan_comps /= fiber_count[:,:,np.newaxis]
        norm_comps /= fiber_count[:,:,np.newaxis]

        fiber_center, tan_comps, norm_comps = self.fill_nans_helical(fiber_center,
                                                                    tan_comps,
                                                                    norm_comps,
                                                                    width,
                                                                    theta,
                                                                    scale)
        return fiber_center, tan_comps, norm_comps



    def grow_helicalfibers(self,fiber_number,director,sigma1,sigma2,fiber_width,avg_width,
                    fiberspace_size, helical_scale,fiber_width_sigma=0, fiber_length=None):
        ''' inputs-
            fiber_number: number of fibers to grow
            director: overall alignment direction of fibers
            sigma1: standard deviation on the normal distribution that determines
                    each individual fibers direction
            sigma2: standard deviation on normal distribution that perturbs an
                    invidual fibers direction as it grows
            fiber_width: radius of fiber short axis in pixels
            avg_width: smoothing window size
            fiberspace_size: size of the square array
            helical_scale: parameter to adjust helical pitch. Higher number is
                            smaller pitch
            fiber_width_sigma: fiber width normal distribution standard deviation
            fiber_length: number of steps to grow fiber. if None, defaults to
                          fiberspace_size
        '''
        # pre-allocate here once, instead of every loop in expand_fiber
        fiberspace = np.zeros((fiberspace_size,fiberspace_size))
        fiber_count = fiberspace.copy()
        fiber_center = fiberspace.copy()

        alignment_tan = np.zeros((fiberspace_size,fiberspace_size,3))
        alignment_norm = alignment_tan.copy()
        tan_comps = alignment_tan.copy()
        norm_comps = alignment_tan.copy()

        if fiber_length is None:
            fiber_length = fiberspace_size

        for i in range(fiber_number):
            mu = self.rng.normal(director,sigma1)

            xlist, ylist = self.grow_fiber_core(fiber_length,mu,sigma2,
                            fiberspace_size)

            xsmooth, ysmooth = self.smooth_fiber_core(xlist,ylist,avg_width)
            theta = self.axial_director(xsmooth,ysmooth)
            fiber_count, tan_comps, norm_comps = self.map_helical(xlist,
                                                               ylist,
                                                               theta,
                                                               self.rng.normal(fiber_width,fiber_width_sigma),
                                                               helical_scale,
                                                               fiberspace,
                                                               fiber_count,
                                                               tan_comps,
                                                               norm_comps,
                                                               fiber_center)
            fiberspace += fiber_count
            alignment_tan += tan_comps
            alignment_norm += norm_comps

        # divide sum of tangent and normal components by number of fibers
        alignment_tan /= fiberspace[:,:,np.newaxis]
        alignment_norm /= fiberspace[:,:,np.newaxis]

        alignment_tan[np.isnan(alignment_tan)] = 0
        alignment_norm[np.isnan(alignment_norm)] = 0

        # make sure magnitude of tangential and normal components are both 1
        dmag = np.sqrt(np.sum(alignment_tan**2,axis=2))
        d2mag = np.sqrt(np.sum(alignment_norm**2,axis=2))
        alignment_tan /= dmag[:,:,np.newaxis]
        alignment_norm /= d2mag[:,:,np.newaxis]
        # calculate binormal vector array from cross product
        alignment_binorm = np.cross(alignment_tan,alignment_norm)

        return fiberspace, alignment_tan, alignment_norm, alignment_binorm

    def grow_sino_helicalfibers(self,fiber_number,director,sigma1,sigma2,fiber_width,avg_width,
                    fiberspace_size, helical_scale,amplitude,period,fiber_width_sigma=0, fiber_length=None):
        ''' inputs-
            fiber_number: number of fibers to grow
            director: overall alignment direction of fibers
            sigma1: standard deviation on the normal distribution that determines
                    each individual fibers direction
            sigma2: standard deviation on normal distribution that perturbs an
                    invidual fibers direction as it grows
            fiber_width: radius of fiber short axis in pixels
            avg_width: smoothing window size
            fiberspace_size: size of the square array
            helical_scale: parameter to adjust helical pitch. Higher number is
                            smaller pitch
            fiber_width_sigma: fiber width normal distribution standard deviation
            fiber_length: number of steps to grow fiber. if None, defaults to
                          fiberspace_size
        '''
        # pre-allocate here once, instead of every loop in expand_fiber
        fiberspace = np.zeros((fiberspace_size,fiberspace_size))
        fiber_count = fiberspace.copy()
        fiber_center = fiberspace.copy()

        alignment_tan = np.zeros((fiberspace_size,fiberspace_size,3))
        alignment_norm = alignment_tan.copy()
        tan_comps = alignment_tan.copy()
        norm_comps = alignment_tan.copy()

        if fiber_length is None:
            fiber_length = fiberspace_size

        for i in range(fiber_number):
            mu = self.rng.normal(director,sigma1)

            xlist, ylist = self.grow_sino_fiber_core(fiber_length,mu,sigma2,amplitude, period,
                            fiberspace_size)

            xsmooth, ysmooth = self.smooth_fiber_core(xlist,ylist,avg_width)
            theta = self.axial_director(xsmooth,ysmooth)
            fiber_count, tan_comps, norm_comps = self.map_helical(xlist,
                                                               ylist,
                                                               theta,
                                                               self.rng.normal(fiber_width,fiber_width_sigma),
                                                               helical_scale,
                                                               fiberspace,
                                                               fiber_count,
                                                               tan_comps,
                                                               norm_comps,
                                                               fiber_center)
            fiberspace += fiber_count
            alignment_tan += tan_comps
            alignment_norm += norm_comps

        # divide sum of tangent and normal componenets by number of fibers
        alignment_tan /= fiberspace[:,:,np.newaxis]
        alignment_norm /= fiberspace[:,:,np.newaxis]

        alignment_tan[np.isnan(alignment_tan)] = 0
        alignment_norm[np.isnan(alignment_norm)] = 0

        # make sure magnitude of tangential and normal components are both 1
        dmag = np.sqrt(np.sum(alignment_tan**2,axis=2))
        d2mag = np.sqrt(np.sum(alignment_norm**2,axis=2))
        alignment_tan /= dmag[:,:,np.newaxis]
        alignment_norm /= d2mag[:,:,np.newaxis]
        # calculate binormal vector array from cross product
        alignment_binorm = np.cross(alignment_tan,alignment_norm)

        return fiberspace, alignment_tan, alignment_norm, alignment_binorm
