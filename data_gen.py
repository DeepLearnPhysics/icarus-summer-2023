# largely taken from ToyMC class in pytorch-flashmatch repo

import numpy as np
from lightpath import LightPath
from flashalgo import FlashAlgo
from flashmatch_types import FlashMatchInput, Flash
from plot import plot_qcluster
import yaml
from photon_library import PhotonLibrary
from points import scatter_points

class DataGen():

    def __init__(self, detector_cfg, cfg, photon_library=None):
        self.configure(detector_cfg, cfg, photon_library)

    def configure(self, detector_file, cfg, particleana, opflashana, photon_library):
        #cfg_file = "icarus-summer-2023/flashmatch.cfg"
        config = yaml.load(open("icarus-summer-2023/flashmatch.cfg"), Loader=yaml.Loader)["ToyMC"]
        self.time_algo = config["TimeAlgo"]
        self.track_algo = config["TrackAlgo"]
        self.periodTPC = config["PeriodTPC"]
        self.periodPMT = config["PeriodPMT"]
        self.ly_variation = config["LightYieldVariation"]
        self.pe_variation = config["PEVariation"]
        self.posx_variation = config['PosXVariation']
        self.truncate_tpc = config["TruncateTPC"]
        self.num_tracks = config["NumTracks"]

        if 'NumpySeed' in config:
            np.random.seed(config['NumpySeed'])

        self.detector = yaml.load(open("icarus-summer-2023/detector_specs.yml"), Loader=yaml.Loader)['DetectorSpecs']
        self.plib = PhotonLibrary()
        self.qcluster_algo = LightPath(self.detector, self.cfg_file)
        self.flash_algo = FlashAlgo(self.detector, self.plib, self.cfg_file)


    def make_flashmatch_inputs(self, num_match=None):
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
            num_match = self.num_tracks

        result = FlashMatchInput()

        # generate 3D trajectories inside the detector
        track_v = self.gen_trajectories(self, num_match)
        result.track_v = track_v

        # generate flash time and x shift (for reco x position assuming trigger time)
        xt_v = self.gen_xt_shift(len(track_v))

        # Defined allowed x recording regions
        min_tpcx, max_tpcx = [t * self.detector['DriftVelocity'] for t in self.periodTPC]

        # generate flash and qclusters
        for idx, track in enumerate(track_v):
            # create raw TPC position and light info
            raw_qcluster = self.make_qcluster(track)
            raw_qcluster.idx = idx
            # Create PMT PE spectrum from raw qcluster
            flash = self.make_flash(raw_qcluster.qpt_v)
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
            if self.truncate_tpc:
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
        return result

    def gen_trajectories(self, num_tracks):
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
        xmin, ymin, zmin = self.detector['ActiveVolumeMin']
        xmax, ymax, zmax = self.detector['ActiveVolumeMax']

        for i in range(num_tracks):
            if self.track_algo=="random":
                start_pt = [np.random.random() * (xmax - xmin) + xmin,
                            np.random.random() * (ymax - ymin) + ymin,
                            np.random.random() * (zmax - zmin) + zmin]
                end_pt = [np.random.random() * (xmax - xmin) + xmin,
                            np.random.random() * (ymax - ymin) + ymin,
                            np.random.random() * (zmax - zmin) + zmin]
            elif self.track_algo=="top-bottom": 
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

    def gen_xt_shift(self, n):
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
        duration = self.periodPMT[1] - self.periodPMT[0]
        for idx in range(n):
            t,x=0.,0.
            if self.time_algo == 'random':
                t = np.random.random() * duration + self.periodPMT[0]
            elif self.time_algo == 'periodic':
                t = (idx + 0.5) * duration / n + self.periodPMT[0]
            elif self.time_algo == 'same':
                t = 0.
            else:
                raise Exception("Time algo not recognized, must be one of ['random', 'periodic']")
            x = t * self.detector['DriftVelocity']
            time_dx_v.append((t,x))
        return time_dx_v

    def make_qcluster(self, track):
        """
        Create a qcluster instance from a trajectory
        ---------
        Arguments
            track: trajectory defined by 3D points
        -------
        Returns
            a qcluster instance 
        """
        qcluster_algo = LightPath(self.detector, cfg_file=None)
        #ly_variation = 0.0
        #posx_variation = 0.0

        qcluster = qcluster_algo.make_qcluster_from_track(track)
        # apply variation if needed
        if self.ly_variation > 0:
            var = abs(np.random.normal(1.0, self.ly_variation, len(qcluster)))
            for idx in range(len(qcluster)): qcluster.qpt_v[idx][-1] *= var[idx]
        if self.posx_variation > 0:
            var = abs(np.random.normal(1.0, self.posx_variation/qcluster.xsum(), len(qcluster)))
            for idx in range(len(qcluster)): qcluster.qpt_v[idx][0] *= var[idx]

        return qcluster

    def make_flash(self, qcluster):
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
        flash.pe_v = self.flash_algo.fill_estimate(qcluster)
        # apply variation if needed
        var = np.ones(shape=(len(flash)),dtype=np.float32)
        if self.pe_variation>0.:
            var = abs(np.random.normal(1.0, self.pe_variation,len(flash)))
        for idx in range(len(flash)):
            estimate = float(int(np.random.poisson(flash.pe_v[idx].item() * var[idx])))
            flash.pe_v[idx] = estimate
            flash.pe_err_v.append(np.sqrt(estimate))

        return flash

#make_flashmatch_inputs()

#writing input to outfile
gen = DataGen()
match_input = gen.make_flashmatch_inputs()
np_result = None

for idx in range(0, len(match_input.track_v)):
    #tpc data
    qcluster = match_input.qcluster_v[idx]
    raw_qcluster = match_input.raw_qcluster_v[idx]
    tpc_idx = match_input.qcluster_v[idx].idx

    #pmt data
    flash = match_input.flash_v[idx]
    flash_idx = match_input.flash_v[idx].idx

    data = []
    store = np.array([[
        idx,
        flash.idx,
        qcluster.idx,
        raw_qcluster.xmin,
        raw_qcluster.xmax,
        qcluster.xmin,
        qcluster.xmax,
        qcluster.sum(),
        qcluster.length(),
        qcluster.time_true,
        flash.sum(),
        flash.time,
        flash.time_true,
        flash.dt_prev,
        flash.dt_next
    ]])
    data.append(store)
    np_idx = np.concatenate(data, axis=0)
    if np_result is None:
        np_result = np_idx
    else:
        np_result = np.concatenate([np_result,np_idx],axis=0)

def attribute_names():
    return [
        'idx',
        'flash_idx',
        'qcluster_idx',
        'raw_qcluster_min_x',
        'raw_qcluster_max_x',
        'qcluster_min_x'
        'qcluster_max_x'
        'qcluster_sum',
        'qcluster_len',
        'qcluster_time_true',
        'flash_sum',
        'flash_time_true',
        'flash_dt_prev',
        'flash_dt_next',
    ]

np.savetxt('test.csv', np_result, delimiter=',', header=','.join(attribute_names()))

#visualizing
def gen_plot():
    print(match_input.qcluster_v[0])
    q_graph = plot_qcluster(match_input.qcluster_v[0])
    return q_graph

gen_plot()