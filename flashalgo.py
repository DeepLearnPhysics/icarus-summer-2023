import torch
import yaml
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlashAlgo():
    def __init__(self, detector_specs, photon_library=None, cfg_file=None):
        self.plib = photon_library
        self.global_qe = 0.0093
        self.reco_pe_calib = 1
        self.qe_v = []  # CCVCorrection factor array
        self.vol_min = torch.tensor(detector_specs["ActiveVolumeMin"], device=device)
        self.vol_max = torch.tensor(detector_specs["ActiveVolumeMax"], device=device)
        if cfg_file:
          self.configure(cfg_file)

    def configure(self, cfg_file):
        config = yaml.load(open(cfg_file), Loader=yaml.Loader)["PhotonLibHypothesis"]
        self.global_qe = config["GlobalQE"]
        self.reco_pe_calib = config["RecoPECalibFactor"]
        self.qe_v = torch.tensor(config["CCVCorrection"], device=device)
        self.siren_path = config["SirenPath"]
        if not self.siren_path and not self.plib:
          print("Must provide either a photon library file or Siren model path")
          raise Exception

    def NormalizePosition(self, pos):
        '''
        Convert position in world coordinate to normalized coordinate      
        '''
        return ((self.plib.Position2AxisID(pos) + 0.5) / self.plib.shape - 0.5) * 2

    def fill_estimate(self, track):
        """
        fill flash hypothsis based on given qcluster track
        ---------
        Arguments
          track: qcluster track of 3D position + charge
        -------
        Returns
          a hypothesis Flash object
        """
        # fill estimate
        if not torch.is_tensor(track):
          track = torch.tensor(track, device=device)
        local_pe_v = torch.sum(self.plib.VisibilityFromXYZ(track[:, :3])*(track[:, 3].unsqueeze(-1)), axis = 0)

        if len(self.qe_v) == 0:
          self.qe_v = torch.ones(local_pe_v.shape, device=device)
        return local_pe_v * self.global_qe * self.reco_pe_calib / self.qe_v

    def backward_gradient(self, track):
        """
        Compue the gradient of the fill_estimate step for given track
        ---------
        Arguments
          track: qcluster track of 3D position + charge
        -------
        Returns
          gradient value of the fill_estimate step for track
        """
        vids = self.plib.Position2VoxID(track[:, :3])
        grad = (self.plib.Visibility(vids+1) - self.plib.Visibility(vids)) / self.plib.gap
        return grad * (track[:, 3].unsqueeze(-1)) * self.global_qe * self.reco_pe_calib / self.qe_v