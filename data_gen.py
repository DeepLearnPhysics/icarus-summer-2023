# largely taken from ToyMC class in pytorch-flashmatch repo

import numpy as np
from lightpath import LightPath
from flashalgo import FlashAlgo
from flashmatch_types import FlashMatchInput, Flash
import yaml
from photon_library import PhotonLibrary
from .points import scatter_points

cfg_file = "icarus-summer-2023/flashmatch.cfg"
config = yaml.load(open("icarus-summer-2023/flashmatch.cfg"), Loader=yaml.Loader)["ToyMC"]
time_algo = config["TimeAlgo"]
track_algo = config["TrackAlgo"]
periodTPC = config["PeriodTPC"]
periodPMT = config["PeriodPMT"]
ly_variation = config["LightYieldVariation"]
pe_variation = config["PEVariation"]
posx_variation = config['PosXVariation']
truncate_tpc = config["TruncateTPC"]
num_tracks = config["NumTracks"]
if 'NumpySeed' in config:
    np.random.seed(config['NumpySeed'])

detector = yaml.load(open("icarus-summer-2023/detector_specs.yml"), Loader=yaml.Loader)['DetectorSpecs']
plib = PhotonLibrary()
qcluster_algo = LightPath(detector, cfg_file)
flash_algo = FlashAlgo(detector, plib, cfg_file)


def make_flashmatch_inputs(num_match=None):
    """
    Make N input pairs for flash matching
    --------
    Arguments
    --------
    Returns
    Generated trajectory, tpc, pmt, and raw tpc arrays
    """
    #num_tracks = 10

    if num_match is None:
        num_match = num_tracks

    result = FlashMatchInput()

    # generate 3D trajectories inside the detector
    track_v = gen_trajectories(num_match)
    result.track_v = track_v

    # generate flash time and x shift (for reco x position assuming trigger time)
    xt_v = gen_xt_shift(len(track_v))

    # Defined allowed x recording regions
    min_tpcx, max_tpcx = [t * detector['DriftVelocity'] for t in periodTPC]

    # generate flash and qclusters
    for idx, track in enumerate(track_v):
        # create raw TPC position and light info
        raw_qcluster = make_qcluster(track)
        raw_qcluster.idx = idx
        # Create PMT PE spectrum from raw qcluster
        flash = make_flash(raw_qcluster.qpt_v)
        flash.idx = idx
        # Apply x shift and set flash time
        ftime, dx = xt_v[idx]
        flash.time = ftime
        flash.time_true = ftime
        qcluster = raw_qcluster.shift(dx)
        qcluster.idx = idx
        qcluster.time_true = ftime
        raw_qcluster.time_true = ftime
        # Drop qcluster points that are outside the recording range
        if truncate_tpc:
            qcluster.drop(min_tpcx, max_tpcx)
        # check for orphan
        valid_match = len(qcluster) > 0 and flash.sum() > 0
        if len(qcluster) > 0:
            result.qcluster_v.append(qcluster)
            result.raw_qcluster_v.append(raw_qcluster)
        if flash.sum() > 0:
            result.flash_v.append(flash)
        if valid_match:
            result.true_match.append((idx,idx))

    print(result.true_match)
    scatter_points(result.qcluster_v,color=qcluster[:,3],markersize=3)
    return result

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
    #track_algo = 'random'

    res = []

    #load detector dimension 
    xmin, ymin, zmin = detector['ActiveVolumeMin']
    xmax, ymax, zmax = detector['ActiveVolumeMax']

    for i in range(num_tracks):
        if track_algo=="random":
            start_pt = [np.random.random() * (xmax - xmin) + xmin,
                        np.random.random() * (ymax - ymin) + ymin,
                        np.random.random() * (zmax - zmin) + zmin]
            end_pt = [np.random.random() * (xmax - xmin) + xmin,
                        np.random.random() * (ymax - ymin) + ymin,
                        np.random.random() * (zmax - zmin) + zmin]
        elif track_algo=="top-bottom": 
        #probably dont need
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

def gen_xt_shift(n):
    """
    Generate flash timing and corresponding X shift
    ---------
    Arguments
        n: int, number of track/flash (number of flash time to be generated)
    -------
    Returns
        a list of pairs, (flash time, dx to be applied on TPC points)
    """
    #can be configured with config file, but default in previous code is to not have one
    #time_algo = 'random'
    #periodPMT = [-1000, 1000]

    time_dx_v = []
    duration = periodPMT[1] - periodPMT[0]
    for idx in range(n):
        t,x=0.,0.
        if time_algo == 'random':
            t = np.random.random() * duration + periodPMT[0]
        elif time_algo == 'periodic':
            t = (idx + 0.5) * duration / n + periodPMT[0]
        elif time_algo == 'same':
            t = 0.
        else:
            raise Exception("Time algo not recognized, must be one of ['random', 'periodic']")
        x = t * detector['DriftVelocity']
        time_dx_v.append((t,x))
    return time_dx_v

def make_qcluster(track):
    """
    Create a qcluster instance from a trajectory
    ---------
    Arguments
        track: trajectory defined by 3D points
    -------
    Returns
        a qcluster instance 
    """
    qcluster_algo = LightPath(detector, cfg_file=None)
    #ly_variation = 0.0
    #posx_variation = 0.0

    qcluster = qcluster_algo.make_qcluster_from_track(track)
    # apply variation if needed
    if ly_variation > 0:
        var = abs(np.random.normal(1.0, ly_variation, len(qcluster)))
        for idx in range(len(qcluster)): qcluster.qpt_v[idx][-1] *= var[idx]
    if posx_variation > 0:
        var = abs(np.random.normal(1.0, posx_variation/qcluster.xsum(), len(qcluster)))
        for idx in range(len(qcluster)): qcluster.qpt_v[idx][0] *= var[idx]

    return qcluster

def make_flash(qcluster):
    """
    Create a flash instance from a qcluster
    ---------
    Arguments
        qcluster: array of 3D position + charge
    -------
    Returns
        a flash instance 
    """
    flash = Flash()
    flash.pe_v = flash_algo.fill_estimate(qcluster)
    # apply variation if needed
    var = np.ones(shape=(len(flash)),dtype=np.float32)
    if pe_variation>0.:
        var = abs(np.random.normal(1.0,pe_variation,len(flash)))
    for idx in range(len(flash)):
        estimate = float(int(np.random.poisson(flash.pe_v[idx].item() * var[idx])))
        flash.pe_v[idx] = estimate
        flash.pe_err_v.append(np.sqrt(estimate))

    return flash

make_flashmatch_inputs()