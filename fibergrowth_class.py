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

    def expand_fiber(self,xlist,ylist,theta,width,fiberspace):
        ''' assign orientation orthogonal to fiber core out to some width '''
        fiber_count = np.zeros(fiberspace.shape)
        fiber_orientation = fiber_count.copy()
        fiber_center = fiber_count.copy()
        r = np.linspace(-width,width,4*width)

        for i, (x,y) in enumerate(zip(xlist,ylist)):
            dx = np.round(x + r*np.cos(np.pi/2+theta[i]))%fiberspace.shape[1]
            dy = np.round(y + r*np.sin(np.pi/2+theta[i]))%fiberspace.shape[0]
            fiber_count[dy.astype(int),dx.astype(int)] += 1
            fiber_orientation[dy.astype(int),dx.astype(int)] += theta[i]
            centerx = np.round(x)%fiberspace.shape[1]
            centery = np.round(y)%fiberspace.shape[0]
            fiber_center[centery.astype(int),centerx.astype(int)] = 1

        fiber_orientation /= fiber_count
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

    def grow_fibers(self,fiber_number,director,sigma1,sigma2,fiber_width,avg_width,
                    fiberspace_size, fiber_length=None):
        fiberspace = np.zeros((fiberspace_size,fiberspace_size))
        alignment_space = fiberspace.copy()


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
                                                               fiber_width,
                                                               fiberspace)
            fiberspace += fiber_count
            alignment_space += fiber_orientation

        alignment_space /= fiberspace

        return fiberspace, alignment_space
