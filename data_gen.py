import numpy as np
from lightpath import LightPath

def gen_trajectories(num_tracks):
    """
    Generate N random trajectories.
    ---------
    Arguments
        num_tracks: int, number of tpc trajectories to be generated
    -------
    Returns
        a list of trajectories, each is a pair of 3D start and end points
    """

    res = []

    #TODO: load detector dimension 
    # xmin, ymin, zmin = detector['ActiveVolumeMin']
    # xmax, ymax, zmax = detector['ActiveVolumeMax']

    for i in range(num_tracks):
        if self.track_algo=="random":
            start_pt = [np.random.random() * (xmax - xmin) + xmin,
                        np.random.random() * (ymax - ymin) + ymin,
                        np.random.random() * (zmax - zmin) + zmin]
            end_pt = [np.random.random() * (xmax - xmin) + xmin,
                        np.random.random() * (ymax - ymin) + ymin,
                        np.random.random() * (zmax - zmin) + zmin]
        elif self._track_algo=="top-bottom":
            start_pt = [np.random.random() * (xmax - xmin) + xmin,
                        ymax,
                        np.random.random() * (zmax - zmin) + zmin]
            end_pt = [np.random.random() * (xmax - xmin) + xmin,
                        ymin,
                        np.random.random() * (zmax - zmin) + zmin]
        else:
            raise Exception("Track algo not recognized, must be one of ['random', 'top-bottom']")
        res.append([start_pt, end_pt])

    return res