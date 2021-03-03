from libc.math cimport sin, cos, isnan
import numpy as np
cimport numpy as cnp



def nucleate(blank_array,nucleation_sites):
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
