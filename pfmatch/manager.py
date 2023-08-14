import yaml
import itertools
import torch
import time
import numpy as np
from .data_gen import DataGen
from .flashmatch_types import FlashMatch
from .algorithm.flashalgo import FlashAlgo
from .photon_library import PhotonLibrary
from .algorithm.match_model import GradientModel, PoissonMatchLoss, EarlyStopping
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Manager():
    def __init__(self, detector_cfg, flashmatch_cfg, photon_library=None):
        self.configure(detector_cfg,  flashmatch_cfg, photon_library)

    def configure(self, detector_cfg, flashmatch_cfg, photon_library):
        config = yaml.load(open(flashmatch_cfg), Loader=yaml.Loader)['FlashMatchManager']
        self.detector_specs = yaml.load(open(detector_cfg), Loader=yaml.Loader)['DetectorSpecs']
        #self.det_cfg = yaml.load(open(detector_cfg), Loader=yaml.Loader)
        self.det_cfg = detector_cfg
        #self.flash_cfg = yaml.load(open(flashmatch_cfg), Loader=yaml.Loader)
        self.flash_cfg = flashmatch_cfg

        #self.photon_library = PhotonLibrary()
        self.photon_library = photon_library

        #for calculate_dx0 method
        self.vol_xmin = self.detector_specs["ActiveVolumeMin"][0]
        self.vol_xmax = self.detector_specs["ActiveVolumeMax"][0]

        self.drift_velocity = self.detector_specs["DriftVelocity"]
        self.time_shift = config['BeamTimeShift']
        self.touching_track_window = config['TouchingTrackWindow']
        self.offset = config['Offset']
        self.num_processes = config['NumProcesses']

        self.exp_frac_v = config['PhotonDecayFractions']
        self.exp_tau_v = config['PhotonDecayTimes']

        #for train method
        self.init_lr = config['InitLearningRate']
        self.min_lr = config['MinLearningRate']
        self.scheduler_factor = config['SchedulerFactor']
        self.stopping_patience = config['StoppingPatience']
        self.stopping_delta = config['StoppingDelta']
        self.max_iteration = int(config['MaxIteration'])
        self.loss_threshold = config['LossThreshold']
        self.flash_algo = FlashAlgo(self.detector_specs, self.photon_library, self.flash_cfg)
        self.loss_fn = PoissonMatchLoss()

    def make_flashmatch_inputs(self):
        gen = DataGen(self.det_cfg, self.flash_cfg)
        return gen.make_flashmatch_inputs()
    
    def visualize_inputs(self):
        #TODO: Kazu has code to do this
        pass

    def flash_match(self, flashmatch_input):
        #TODO
        """
        Run flash matching on flashmatch input
        --------
        Arguments
          flashmatch_input: FlashMatchInput object
        --------
        Returns
          FlashMatch object storing the result of the match
        """
        match = FlashMatch(len(flashmatch_input.qcluster_v), len(flashmatch_input.flash_v))
        paramlist = list(itertools.product(flashmatch_input.qcluster_v, flashmatch_input.flash_v))

        import torch.multiprocessing as mp
        from multiprocessing.pool import ThreadPool

        ctx = mp.get_context("spawn")
        track_id, flash_id = 0, 0

        with ThreadPool(processes=self.num_processes) as pool:
            for loss, reco_x, reco_pe, duration in pool.imap(self.one_pmt_match, paramlist):
                match.loss_matrix[track_id, flash_id] = loss
                match.reco_x_matrix[track_id, flash_id] = reco_x
                match.reco_pe_matrix[track_id, flash_id] = reco_pe
                match.duration[track_id, flash_id] = duration
                if flash_id < len(flashmatch_input.flash_v) - 1:
                  flash_id += 1
                else:
                  track_id += 1
                  flash_id = 0

        match.bipartite_match()
        return match

    def one_pmt_match(self, params):
        """
        Run flash matching on for one pair of qcluster and flash input
        --------
        Arguments
          params: tuple of (qcluster, flash)
        --------
        Returns
          loss, reco_x, reco_pe
        """
        res = []
        qcluster, flash = params

        dx0_v, dx_min, dx_max = self.calculate_dx0(flash, qcluster)
        if len(dx0_v) == 0:
          return np.inf, np.inf, np.inf, np.inf
        
        # calculate the integral factor to reweight flash based on its time width
        integral_factor = 0
        for i in range(len(self.exp_frac_v)):
            integral_factor += self.exp_frac_v[i] * (1 - np.exp(-1 * flash.time_width / self.exp_tau_v[i]))

        input = qcluster.qpt_v
        target = flash.pe_v / integral_factor

        min_loss = np.inf
        for dx_0 in dx0_v:
            loss, reco_x, reco_pe, duration = self.train(input, target, dx_0, dx_min, dx_max)
            if loss < min_loss:
                min_loss = loss
                res = [loss, reco_x, reco_pe, duration]
        return res
        
    def calculate_dx0(self, flash, qcluster):
        x0_v = []
        track_xmin, track_xmax = qcluster.xmin, qcluster.xmax
        dx_min, dx_max = self.vol_xmin - track_xmin, self.vol_xmax - track_xmax
        dx0 = (flash.time - self.time_shift) * self.drift_velocity

        # determine initial x0
        tolerence = self.touching_track_window/2. * self.drift_velocity
        contained_tpc0 = (-dx0>=dx_min-tolerence) and (-dx0<=dx_max+tolerence)
        contained_tpc1 = (dx0>=dx_min-tolerence) and (dx0<=dx_max+tolerence)

        # Inspect, in either assumption (original track is in tpc0 or tpc1), the track is contained in the whole active volume or not
        if contained_tpc0:
            x0_v.append(max(-dx0, self.vol_xmin - track_xmin + self.offset))
        if contained_tpc1:
            x0_v.append(min(dx0, self.vol_xmax - track_xmax - self.offset))

        return x0_v, dx_min, dx_max

    def train(self, input, target, dx0, dx_min, dx_max):
        """
        Run gradient descent model on input
        --------
        Arguments
          input: qcluster input as tensor
          target: flash target as tensor
          dx0: initial xshift in cm
          dx_min: miminal allowed value of dx in cm
          dx_max: maximum allowed value of dx in cm
        --------
        Returns
          loss, reco_x, reco_pe
        """
        model = GradientModel(self.flash_algo, dx0, dx_min, dx_max)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=self.min_lr, factor=self.scheduler_factor)
        early_stopping = EarlyStopping(self.stopping_patience, self.stopping_delta)

        start = time.time()
        for i in range(self.max_iteration):
            pred = model(input)
            loss = self.loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            early_stopping(loss)

            if loss > self.loss_threshold or early_stopping.early_stop:
                break

        end = time.time()

        return loss.item(), model.xshift.dx.item(), torch.sum(pred).item(), end-start