import numpy as np
from skimage.morphology import dilation, disk

class fibergrowth():

    def __init__(self,random_seed=None):
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
        else:
            self.rng = np.random.default_rng()

    def grow_fiber_core(self,fiber_length,mu,sigma,fiberspace_size):
        ''' Uses a list based method to propagate a fiber with a
        uniform distribution about an overall director

        mu - overall director direction
        sigma - standard deviation of normal distribution

        TO DO: code in other distributions'''
        xlist = []
        ylist = []
        init_pos = self.rng.integers(0,fiberspace_size,2)
        xpos = init_pos[1]
        ypos = init_pos[0]
        ylist.append(init_pos[0])
        xlist.append(init_pos[1])
        # grow fiber for given amount of steps
        for i in np.arange(0,fiber_length):
            # perturb fiber growth direction
            director_perturb = self.rng.normal(mu, sigma)
            director = director_perturb
            xstep = np.cos(director)
            ystep = np.sin(director)
            xpos = (xpos + xstep)
            ypos = (ypos + ystep)


            xlist.append(xpos)
            ylist.append(ypos)

        return xlist, ylist

    def smooth_fiber_core(self,xlist, ylist, avg_width):
        ''' Smooths out grow_fiber_core results using a
        moving window average of width avg_width.
        Used for axial director calculation '''

        xsmooth = []
        ysmooth = []

        for j in np.arange(0,avg_width):
            xsmooth.append(np.mean(xlist[0:(2*j+1)]))
            ysmooth.append(np.mean(ylist[0:(2*j+1)]))
        for j in np.arange(avg_width,len(xlist)-avg_width):
            min_idx = j-avg_width
            max_idx = j+avg_width+1
            xsmooth.append(np.mean(xlist[min_idx:max_idx]))
            ysmooth.append(np.mean(ylist[min_idx:max_idx]))
        for j in np.arange(len(xlist)-avg_width,len(xlist)):
            xsmooth.append(np.mean(xlist[(j-avg_width):]))
            ysmooth.append(np.mean(ylist[(j-avg_width):]))

        return xsmooth, ysmooth

    def axial_director(self, xlist, ylist):
        ''' Calculates axial director of fiber core at each point '''
        diff_y = np.diff(ylist)
        diff_x = np.diff(xlist)
        theta = np.arctan(diff_y/diff_x)
        theta = np.concatenate((theta,theta[-1]),axis=None)

        return theta

    def fill_nans(self,fiber_center,fiber_count,fiber_orientation,width,theta):
        # fill in missing holes using fiber from disk structuring element
        fiber_center = dilation(fiber_center,disk(width))
        fill_idx = (fiber_center == 1) & (np.isnan(fiber_orientation))
        fill2 = np.where(fill_idx == True)

        for y,x in zip(fill2[0],fill2[1]):
            fiber_orientation[y,x] = np.nanmean(fiber_orientation[(y-1):(y+2),(x-1):(x+2)])

        # get the first bit of the fiber that wasn't caught with nanmean
        fill_idx = (fiber_center == 1) & (np.isnan(fiber_orientation))
        fill2 = np.where(fill_idx == True)
        fiber_orientation[fill2[0].astype(int),fill2[1].astype(int)] = theta[0]
        fiber_orientation[np.isnan(fiber_orientation)] = 0

        # go back and make sure fiber_count matches
        fiber_count[fiber_orientation != 0] = 1
        return fiber_count, fiber_orientation

    def expand_fiber(self,xlist,ylist,theta,width,fiberspace,fiber_count,
                                                            fiber_orientation,
                                                            fiber_center):
        ''' assign orientation orthogonal to fiber core out to some width '''

        # zero out fiber_count, fiber_orientation, and fiber_center
        fiber_count[:] = 0
        fiber_orientation[:] = 0
        fiber_center[:] = 0
        r = np.linspace(-width,width,4*round(width))

        for i, (x,y) in enumerate(zip(xlist,ylist)):
            dx = np.round(x + r*np.cos(np.pi/2+theta[i]))%fiberspace.shape[1]
            dy = np.round(y + r*np.sin(np.pi/2+theta[i]))%fiberspace.shape[0]
            fiber_count[dy.astype(int),dx.astype(int)] += 1
            fiber_orientation[dy.astype(int),dx.astype(int)] += theta[i]
            centerx = np.round(x)%fiberspace.shape[1]
            centery = np.round(y)%fiberspace.shape[0]
            fiber_center[centery.astype(int),centerx.astype(int)] = 1

        fiber_orientation /= fiber_count

        fiber_count, fiber_orientation = self.fill_nans(fiber_center,
                                                        fiber_count,
                                                        fiber_orientation,
                                                        width,theta)

        return fiber_count, fiber_orientation

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
