import numpy as np

class crystalgrowth():

    def __init__(self, random_seed=None):
        if random_seed is not None:
            self.rng = np.random.default_rng(seed=random_seed)
        else:
            self.rng = np.random.default_rng()

    def new_func(self):
        return 5


    def nucleate(self,blank_array,nucleation_sites):
        """Seeds the structure with given number of sites

        Sites are selected randomly in the domain.

        At each site, an orientation in [-pi/2, pi/2) is selected and assigned to the given location.

        There is no check for distance between sites or
        to ensure sites are unique.

        Passing a random seed allows reproducible results. If no seed is passed, the generator will not be seeded
        and should produce random results each time.

        This function is dimension independent so will work in both 2d and 3d

        Args:
            blank_array: array to fill with seeds
            nucleation_sites: number of nucleation sites to use
            random_seed: seed for random number generator

        Returns:
            A tuple with (the seeded array, a list of the seed locations, the orientation at each location)
        """
        positions = [[self.rng.integers(0, dim) for dim in blank_array.shape] for i in range(nucleation_sites)]
        positions = np.array(positions)
        orientations = self.rng.uniform(-np.pi/2, np.pi/2, size=nucleation_sites)
        for pos, orient in zip(positions, orientations):
            blank_array[tuple(pos)] = orient
        return blank_array, positions, orientations


    def add_orient(self,grow_array, location, base_orientation, mutate, next_crystal_idx):
        """Add orientation at given location and store location in list of crystal sites

        The orientation is the base orientation plus a random factor in the range [-mutate, mutate]

        This function is dimension independent so will work in both 2d and 3d

        Args:
            grow_array: The crystal array data
            location: location of site to add orientation
            base_orientation: base orientation value
            mutate: mutation factor
            next_crystal_idx: list of indices to check in next growth cycle
        """
        # if there is already an orientation here, don't do anything
        if np.isnan(grow_array[location]):

            new_orient = base_orientation + np.random.uniform(-mutate, mutate)

            # force to be within range [-pi/2, pi/2]
            if new_orient > (0.5 * np.pi):
                new_orient %= -(0.5 * np.pi)
            if new_orient < -(0.5 * np.pi):
                new_orient %= (0.5 * np.pi)

            grow_array[location] = new_orient
            next_crystal_idx.append(location)


    def grow_2d(self,grow_array, pos, pi_mutate, c_mutate, next_crystal_idx, do_anisotropic_step, do_periodic):
        """Grow crystal around given location in a 2d crystal

        This will update the grow_array matrix with new crystal orientations and add future growth locations
        to the next_crystal_idx list

        Args:
            grow_array: array for crystal
            pos: location of point to grow around
            pi_mutate:
            c_mutate:
            next_crystal_idx: list of points to grow in the future
            do_anisotropic_step: whether to do growth along the minor crystal axis
            do_periodic: whether to allow periodic wrapping
        """

        xdim = grow_array.shape[1]
        ydim = grow_array.shape[0]
        # crystallization along pi-pi stacking
        orientation = grow_array[pos]
        director_x = round(abs(np.cos(orientation)))
        director_y = round(np.sin(orientation))
        director_xup = (pos[1]+director_x)
        director_yup = (pos[0]+director_y)
        director_xdn = (pos[1]-director_x)
        director_ydn = (pos[0]-director_y)
        if do_periodic:
            director_xup %= xdim
            director_yup %= ydim
            director_xdn %= xdim
            director_ydn %= ydim

        else:
            if (director_yup < ydim and director_yup >= 0):
                yup_in = True
            else:
                yup_in = False
            if (director_xup < xdim and director_xup >= 0):
                xup_in = True
            else:
                xup_in = False
            if (director_ydn < ydim and director_ydn >= 0):
                ydn_in = True
            else:
                ydn_in = False
            if (director_xdn < xdim and director_xdn >= 0):
                xdn_in = True
            else:
                xdn_in = False

        if do_periodic or (yup_in and xup_in):
            self.add_orient(grow_array, (director_yup, director_xup), orientation, pi_mutate, next_crystal_idx)
        if do_periodic or (ydn_in and xdn_in):
            self.add_orient(grow_array, (director_ydn, director_xdn), orientation, pi_mutate, next_crystal_idx)

        # lateral crystal thickening, switch directors
        if do_anisotropic_step:
            # for diagonal pi-pi stacking
            if (np.abs(director_x) + np.abs(director_y)) == 2:
                # fill in checkerboard pattern
                if do_periodic or yup_in:
                    self.add_orient(grow_array, (director_yup, pos[1]), orientation, c_mutate, next_crystal_idx)
                if do_periodic or ydn_in:
                    self.add_orient(grow_array, (director_ydn, pos[1]), orientation, c_mutate, next_crystal_idx)
                if do_periodic or xup_in:
                    self.add_orient(grow_array, (pos[0], director_xup), orientation, c_mutate, next_crystal_idx)
                if do_periodic or xdn_in:
                    self.add_orient(grow_array, (pos[0], director_xdn), orientation, c_mutate, next_crystal_idx)

            # for pi-pi direction vertical or horizontal
            else:
                director_xup = (pos[1]+director_y)
                director_yup = (pos[0]+director_x)
                director_xdn = (pos[1]-director_y)
                director_ydn = (pos[0]-director_x)
                if do_periodic:
                    director_xup %= xdim
                    director_yup %= ydim
                    director_xdn %= xdim
                    director_ydn %= ydim

                else:
                    if (director_yup < ydim and director_yup >= 0):
                        yup_in = True
                    else:
                        yup_in = False
                    if (director_xup < xdim and director_xup >= 0):
                        xup_in = True
                    else:
                        xup_in = False
                    if (director_ydn < ydim and director_ydn >= 0):
                        ydn_in = True
                    else:
                        ydn_in = False
                    if (director_xdn < xdim and director_xdn >= 0):
                        xdn_in = True
                    else:
                        xdn_in = False

                if do_periodic or (yup_in and xdn_in):
                    self.add_orient(grow_array, (director_yup, director_xdn), orientation, c_mutate, next_crystal_idx)
                if do_periodic or (xup_in and ydn_in):
                    self.add_orient(grow_array, (director_ydn, director_xup), orientation, c_mutate, next_crystal_idx)


    def grow_3d(self,grow_array, pos, pi_mutate, c_mutate, next_crystal_idx, do_anisotropic_step, do_periodic):
        """Grow crystal around given location in a 3d crystal

        This will update the grow_array matrix with new crystal orientations and add future growth locations
        to the next_crystal_idx list

        Args:
            grow_array: array for crystal
            pos: location of point to grow around
            pi_mutate:
            c_mutate:
            next_crystal_idx: list of points to grow in the future
            do_anisotropic_step: whether to do growth along the minor crystal axis
            do_periodic: whether to allow periodic wrapping
        """
        z_size = grow_array.shape[2]

        # crystallization along pi-pi stacking
        orientation = grow_array[pos]
        director_x = int(round(np.abs(np.cos(orientation))))
        director_y = int(round(np.abs(np.sin(orientation))))
        director_z = 1
        director_xup = (pos[1]+director_x)
        director_yup = (pos[0]+director_y)
        director_xdn = (pos[1]-director_x)
        director_ydn = (pos[0]-director_y)

        z_value = pos[2]
        director_zup = (pos[2]+director_z)
        director_zdn = (pos[2]-director_z)
        if do_periodic:  # only periodic in x and y
            director_xup %= len(grow_array)
            director_yup %= len(grow_array)
            director_xdn %= len(grow_array)
            director_ydn %= len(grow_array)

        if do_periodic or (director_yup < len(grow_array) and director_xup < len(grow_array)):
            self.add_orient(grow_array, (director_yup, director_xup, z_value), orientation, pi_mutate, next_crystal_idx)
        if do_periodic or (director_ydn >= 0 and director_xdn >= 0):
            self.add_orient(grow_array, (director_ydn, director_xdn, z_value), orientation, pi_mutate, next_crystal_idx)

        # lateral crystal thickening, switch directors
        if do_anisotropic_step:
            # z dimension growth
            if director_zdn >= 0:
                self.add_orient(grow_array, (pos[0], pos[1], director_zdn), orientation, c_mutate, next_crystal_idx)
            if director_zup < z_size:
                self.add_orient(grow_array, (pos[0], pos[1], director_zup), orientation, c_mutate, next_crystal_idx)

            # for diagonal pi-pi stacking
            if (np.abs(director_x) + np.abs(director_y)) == 2:
                # fill in checkerboard pattern
                if do_periodic or director_yup < len(grow_array):
                    self.add_orient(grow_array, (director_yup, pos[1], z_value), orientation, c_mutate, next_crystal_idx)
                if do_periodic or director_ydn >= 0:
                    self.add_orient(grow_array, (director_ydn, pos[1], z_value), orientation, c_mutate, next_crystal_idx)
                if do_periodic or director_xup < len(grow_array):
                    self.add_orient(grow_array, (pos[0], director_xup, z_value), orientation, c_mutate, next_crystal_idx)
                if do_periodic or director_xdn >= 0:
                    self.add_orient(grow_array, (pos[0], director_xdn, z_value), orientation, c_mutate, next_crystal_idx)

            # for pi-pi direction vertical or horizontal
            else:
                director_xup = (pos[1]+director_y)
                director_yup = (pos[0]+director_x)
                director_xdn = (pos[1]-director_y)
                director_ydn = (pos[0]-director_x)
                if do_periodic:
                    director_xup %= len(grow_array)
                    director_yup %= len(grow_array)
                    director_xdn %= len(grow_array)
                    director_ydn %= len(grow_array)

                if do_periodic or (director_yup < len(grow_array) and director_xdn >= 0):
                    self.add_orient(grow_array, (director_yup, director_xdn, z_value), orientation, c_mutate, next_crystal_idx)
                if do_periodic or (director_xup < len(grow_array) and director_ydn >= 0):
                    self.add_orient(grow_array, (director_ydn, director_xup, z_value), orientation, c_mutate, next_crystal_idx)


    def grow_complete(self,nucleated_array, growth_anisotropy=2, pi_mutate=np.deg2rad(5), c_mutate=np.deg2rad(5),
                            do_periodic=False, debug=None):
        """Takes nucleated array and grows crystal until the array is completely filled.

        This uses a queueing method for identifying growth locations.
        The growth can allow or reject periodic wrapping of the crystals

        Args:
            nucleated_array: array with crystal seeds
            growth_anisotropy: anisotropy in growing crystals
            pi_mutate:
            c_mutate:
            do_periodic: whether to allow periodic wrapping
            debug: return the state *after* the given step number (0-indexed so 0 gives the state after 1 step)
                   if set to the default value of None, the final state will be returned

        Returns:
            array with grown crystals
        """
        grow_array = nucleated_array.copy()

        # indices of points to grow from
        crystal_idx = [tuple(i) for i in np.column_stack(np.where(~np.isnan(nucleated_array)))]
        revisit = []  # these are points we need to revisit due to anisotropy

        time = 0  # growth cycle (used for anisotropy)

        ndim = grow_array.ndim

        finished = False
        while not finished:
            next_crystal_idx = []  # indices of growth points for next cycle

            do_anisotropic_step = (time % growth_anisotropy == 0)

            # if we are doing the anisotropic sites, include the sites we need to revisit
            if do_anisotropic_step and time != 0:
                crystal_idx = revisit[:]  # this includes the crystal_idx points
                revisit = []

            # loop over the crystal points and grow around each
            for pos in crystal_idx:
                if ndim == 2:
                    self.grow_2d(grow_array, pos, pi_mutate, c_mutate, next_crystal_idx, do_anisotropic_step, do_periodic)
                elif ndim == 3:
                    self.grow_3d(grow_array, pos, pi_mutate, c_mutate, next_crystal_idx, do_anisotropic_step, do_periodic)

            crystal_idx = next_crystal_idx[:]
            revisit.extend(next_crystal_idx)

            # Check if we are finished with growth. This handles a weird edge case where there may be ungrown sites
            # that can only be reached by the anisotropic portion of the growth process. If that is the case, the
            # list of sites to check will be empty but there will still be empty sites. The list of sites to revisit
            # should not be empty, so when we reach the anisotropic step, they will be filled in during that step.
            # If we only check for an empty list of crystal sites, we will miss this case and end with ungrown sites.
            # instead, if crystal_idx is empty, we need to also check for NANs in the crystal array. If there are none,
            # then we are truly done (regardless of whether there are items in the revisit array).
            # Presumably we could just check for NANs at every step, but this is marginally more efficient.
            if len(crystal_idx) == 0:
                if not np.isnan(np.sum(grow_array)):
                    finished = True

            if debug is not None and time == debug:
                return grow_array
            time += 1
        return grow_array

    def grow_partial(self,nucleated_array, timesteps, growth_anisotropy=2, pi_mutate=np.deg2rad(5), c_mutate=np.deg2rad(5),
                            do_periodic=False, debug=None):
        """Takes nucleated array and grows crystal for a given number of timesteps.

        This uses a queueing method for identifying growth locations.
        The growth can allow or reject periodic wrapping of the crystals

        Args:
            nucleated_array: array with crystal seeds
            growth_anisotropy: anisotropy in growing crystals
            pi_mutate:
            c_mutate:
            do_periodic: whether to allow periodic wrapping
            debug: return the state *after* the given step number (0-indexed so 0 gives the state after 1 step)
                   if set to the default value of None, the final state will be returned

        Returns:
            array with grown crystals
        """
        grow_array = nucleated_array.copy()

        # indices of points to grow from
        crystal_idx = [tuple(i) for i in np.column_stack(np.where(~np.isnan(nucleated_array)))]
        revisit = []  # these are points we need to revisit due to anisotropy

        time = 0  # growth cycle (used for anisotropy)

        ndim = grow_array.ndim

        for time in np.arange(timesteps):
            next_crystal_idx = []  # indices of growth points for next cycle

            do_anisotropic_step = (time % growth_anisotropy == 0)

            # if we are doing the anisotropic sites, include the sites we need to revisit
            if do_anisotropic_step and time != 0:
                crystal_idx = revisit[:]  # this includes the crystal_idx points
                revisit = []

            # loop over the crystal points and grow around each
            for pos in crystal_idx:
                if ndim == 2:
                    self.grow_2d(grow_array, pos, pi_mutate, c_mutate, next_crystal_idx, do_anisotropic_step, do_periodic)
                elif ndim == 3:
                    self.grow_3d(grow_array, pos, pi_mutate, c_mutate, next_crystal_idx, do_anisotropic_step, do_periodic)

            crystal_idx = next_crystal_idx[:]
            revisit.extend(next_crystal_idx)

            # Check if we are finished with growth. This handles a weird edge case where there may be ungrown sites
            # that can only be reached by the anisotropic portion of the growth process. If that is the case, the
            # list of sites to check will be empty but there will still be empty sites. The list of sites to revisit
            # should not be empty, so when we reach the anisotropic step, they will be filled in during that step.
            # If we only check for an empty list of crystal sites, we will miss this case and end with ungrown sites.
            # instead, if crystal_idx is empty, we need to also check for NANs in the crystal array. If there are none,
            # then we are truly done (regardless of whether there are items in the revisit array).
            # Presumably we could just check for NANs at every step, but this is marginally more efficient.
            if len(crystal_idx) == 0:
                if not np.isnan(np.sum(grow_array)):
                    finished = True

            if debug is not None and time == debug:
                return grow_array
            # time += 1
        return grow_array
