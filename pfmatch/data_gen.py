# largely taken from ToyMC class in pytorch-flashmatch repo
import numpy as np
from .algorithm.lightpath import LightPath
from .algorithm.flashalgo import FlashAlgo
from .flashmatch_types import FlashMatchInput, Flash, QCluster
from .plot import plot_qcluster
import yaml
from .photonlib.photon_library import PhotonLibrary
from .points import scatter_points
import torch

class DataGen():
    #TODO: Modify to work with photon library or siren input for visibility

    def __init__(self,detector_cfg, match_cfg, photon_lib):
        self.configure(detector_cfg, match_cfg, photon_lib)
        
    def configure_from_yaml(self, detector_yml, match_yml, photon_lib):
        fmatch_cfg   = yaml.load(open(match_yml),    Loader=yaml.Loader)
        detector_cfg = yaml.load(open(detector_yml), Loader=yaml.Loader)['DetectorSpecs']
        self.configure(detector_cfg, fmatch_cfg, photon_lib)
        
    def configure(self,detector_cfg, fmatch_cfg, photon_lib):
        
        gen_cfg = fmatch_cfg['ToyMC']
        self.time_algo  = gen_cfg["TimeAlgo"]
        self.track_algo = gen_cfg["TrackAlgo"]
        self.periodTPC  = gen_cfg["PeriodTPC"]
        self.periodPMT  = gen_cfg["PeriodPMT"]
        self.ly_variation = gen_cfg["LightYieldVariation"]
        self.pe_variation = gen_cfg["PEVariation"]
        self.posx_variation = gen_cfg['PosXVariation']
        self.truncate_tpc = gen_cfg["TruncateTPC"]
        self.num_tracks = gen_cfg["NumTracks"]

        if 'NumpySeed' in gen_cfg:
            np.random.seed(gen_cfg['NumpySeed'])

        self.detector = detector_cfg
        self.plib = photon_lib
        self.qcluster_algo = LightPath(self.detector, fmatch_cfg)
        self.flash_algo = FlashAlgo(self.detector, self.plib, fmatch_cfg)
        
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
        track_v = self.gen_trajectories(num_match)
        result.track_v = track_v

        # generate flash time and x shift (for reco x position assuming trigger time)
        xt_v = self.gen_xt_shift(len(track_v))

        # Defined allowed x recording regions in "reconstructed x" coordinate (assuming neutrino timing)
        min_tpcx, max_tpcx = [t * self.detector['DriftVelocity'] for t in self.periodTPC]

        # generate flash and qclusters in 5 steps
        for idx, track in enumerate(track_v):
            
            # 1. create raw TPC position and light info
            raw_qcluster = self.make_qcluster(track)
            raw_qcluster.idx = idx
            
            # 2. Create PMT PE spectrum from raw qcluster
            flash = self.make_flash(raw_qcluster.qpt_v)
            flash.idx = idx
            
            # 3. Apply x shift and set flash time
            ftime, dx = xt_v[idx]
            flash.time = ftime
            flash.time_true = ftime            
            qcluster = raw_qcluster.shift(dx)
            qcluster.idx = idx
            qcluster.time_true = ftime
            raw_qcluster.time_true = ftime
            
            # 4. Drop qcluster points that are outside the recording range
            if self.truncate_tpc:
                qcluster.drop(min_tpcx, max_tpcx)
                
            # 5. check for orphan
            valid_match = len(qcluster) > 0 and flash.sum() > 0
            
            if len(qcluster) > 0:
                result.qcluster_v.append(qcluster)
                result.raw_qcluster_v.append(raw_qcluster)
                
            if flash.sum() > 0:
                result.flash_v.append(flash)
                
            if valid_match:
                result.true_match.append((idx,idx))

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
        #ly_variation = 0.0
        #posx_variation = 0.0

        qcluster = self.qcluster_algo.make_qcluster_from_track(track)
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
        qpt_v = qcluster
        if type(qcluster) == type(QCluster()):
            qpt_v = qcluster.qpt_v

        pe_v = self.flash_algo.fill_estimate(qpt_v)
        pe_err_v = []
        # apply variation if needed
        var = np.ones(shape=(len(pe_v)),dtype=np.float32)
        if self.pe_variation>0.:
            var = abs(np.random.normal(1.0, self.pe_variation,len(pe_v)))
        for idx in range(len(pe_v)):
            estimate = float(int(np.random.poisson(pe_v[idx].item() * var[idx])))
            pe_v[idx] = estimate
            pe_err_v.append(np.sqrt(estimate))

        flash = Flash()
        flash.pe_v = torch.tensor(pe_v,device=pe_v.device)
        flash.pe_err_v = torch.tensor(pe_err_v,device=pe_v.device)
        return flash

#make_flashmatch_inputs()

#writing input to outfile
#gen = DataGen()
#match_input = gen.make_flashmatch_inputs()
#np_result = None

#for idx in range(0, len(match_input.track_v)):
#    #tpc data
#    qcluster = match_input.qcluster_v[idx]
#    raw_qcluster = match_input.raw_qcluster_v[idx]
#    tpc_idx = match_input.qcluster_v[idx].idx
#
#    #pmt data
#    flash = match_input.flash_v[idx]
#    flash_idx = match_input.flash_v[idx].idx
#
#    data = []
#    store = np.array([[
#        idx,
#        flash.idx,
#        qcluster.idx,
#        raw_qcluster.xmin,
#        raw_qcluster.xmax,
#        qcluster.xmin,
#        qcluster.xmax,
#        qcluster.sum(),
#        qcluster.length(),
#        qcluster.time_true,
#        flash.sum(),
#        flash.time,
#        flash.time_true,
#        flash.dt_prev,
#        flash.dt_next
#    ]])
#    data.append(store)
#    np_idx = np.concatenate(data, axis=0)
#    if np_result is None:
#        np_result = np_idx
#    else:
#        np_result = np.concatenate([np_result,np_idx],axis=0)

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

#np.savetxt('test.csv', np_result, delimiter=',', header=','.join(attribute_names()))