import numpy as np
from points import scatter_points


def plot_flash(toymc, flash, x=None, **kwargs):
    """
    Plot a flashmatch::Flash_t
    """
    nopch = toymc.detector['PhotonLibraryNOpDetChannels']
    pmt_positions = []
    if x is not None and not isinstance(x, float):
        raise Exception('x needs to be a float')

    for pmt in range(nopch):
        pmt_position = toymc.detector['PMT' + str(pmt)]
        if x is not None and isinstance(x, float):
            if not np.isclose(pmt_position[0], x):
                continue
        pmt_positions.append(np.array([[pmt_position[0], pmt_position[1], pmt_position[2], flash[pmt]]]))
    pmt_positions = np.vstack(pmt_positions)
    if x is None:
        return scatter_points(pmt_positions[:, 0:3], color=pmt_positions[:, -1], dim=3, markersize=3, **kwargs)
    else:
        return scatter_points(pmt_positions[:, [2, 1]], color=pmt_positions[:, -1], dim=2, markersize=15, **kwargs)


def plot_qcluster(qcluster, npts=100):
    """
    Plot a flashmatch::QCluster_t
    """
    return scatter_points(qcluster,markersize=3)


#modified from original
def plot_track(track, npts=100):
    """
    Plot a flashmatch::Trajectory_t with only 2 points (start and end of track)
    """
    xmin, ymin, zmin = track[0][0], track[0][1], track[0][2]
    xmax, ymax, zmax = track[1][0], track[1][1], track[1][2]

    xyzs=np.zeros(shape=(npts,4),dtype=np.float32)
    xyzs[:,0] = [i*(xmax - xmin)/npts + xmin for i in range(npts)]
    xyzs[:,1] = [ymin + i*(ymax-ymin)/npts for i in range(npts)]
    xyzs[:,2] = [i*(zmax - zmin)/npts + zmin for i in range(npts)]
    # let's generate visibility per pmt
    # vis_array=toymc.plib.VisibilityFromXYZ(xyzs[:, :3]).cpu().numpy()
    # xyzs[:, 3] = np.sum(vis_array, axis=1)
    # visualize
    return scatter_points(xyzs,markersize=3)