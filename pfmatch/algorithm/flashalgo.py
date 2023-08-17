import torch
import yaml
from ..photonlib.siren_alt import SirenLibrary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class FlashAlgo():
    def __init__(self, detector_specs, photon_library, cfg_file):
        self.plib = photon_library
        #self.slib = SirenLibrary(cfg_file)
        self.global_qe = 0.0093
        self.reco_pe_calib = 1
        self.qe_v = []  # CCVCorrection factor array
        self.vol_min = torch.tensor(detector_specs["ActiveVolumeMin"], device=device)
        self.vol_max = torch.tensor(detector_specs["ActiveVolumeMax"], device=device)
        if cfg_file:
          self.configure(cfg_file)

    def configure_from_yaml(self, fmatch_yaml):
        self.configure(yaml.load(open(cfg_file), Loader=yaml.Loader)["LightPath"])

        
    def configure(self, fmatch_config):
        config = fmatch_config['PhotonLibHypothesis']
        self.global_qe = config["GlobalQE"]
        self.reco_pe_calib = config["RecoPECalibFactor"]
        self.qe_v = torch.tensor(config["CCVCorrection"], device=device)
        self.siren_path = config["SirenPath"]
        self.use_siren = config["UseSiren"]
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
        #if not torch.is_tensor(track):
        #  track = torch.tensor(track, device=device)

        #fill estimate
        if self.use_siren:
          #local_pe_v = torch.sum(self.slib.VisibilityFromXYZ(track[:, :3])*(track[:, 3].unsqueeze(-1)), axis = 0)
          pass
        else:
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
        
        if self.use_siren:
          #neighboring voxel vis values - track voxel vis values / distance btwn voxel pairs
          # neighbor_track = track[:, :3]
          # neighbor_track[:, 0] += self.slib.voxel_width

          # grad = (self.slib.Visibility(neighbor_track) - self.slib.Visibility(track[:, :3])) / self.slib.voxel_width
          pass

        else:
          vids = self.plib.Position2VoxID(track[:, :3])
          grad = (self.plib.Visibility(vids+1) - self.plib.Visibility(vids)) / self.plib.gap

        return grad * (track[:, 3].unsqueeze(-1)) * self.global_qe * self.reco_pe_calib / self.qe_v