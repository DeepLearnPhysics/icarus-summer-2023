#provides same functionality as photon_library but with a siren model

import numpy as np
import torch
import torch.nn as nn
import yaml
from ..algorithm.siren_modules import Siren
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = list(range(torch.cuda.device_count()))

class SirenLibrary(nn.Module):
    def __init__(self, cfg_file, in_features=3, hidden_features=512, hidden_layers=5, out_features=180, outermost_linear=True, omega=30):
        config = yaml.load(open(cfg_file), Loader=yaml.Loader)["PhotonLibHypothesis"]
        self.siren_path = config["SirenPath"]
        super().__init__()
        self.model = Siren(in_features, hidden_features, hidden_layers, out_features, outermost_linear, omega)
        self.model = self.model.float()
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model.cuda()
        self.model.load_state_dict(torch.load(self.siren_path))

        self.voxel_width = 5

    def LoadData(self, transform=True, eps=1e-5):
        '''
        Load photon library visibility data. Apply scale transform if specified
        '''
        pass
        #i dont think this is actually called anywhere
    
    def DataTransform(self, data, eps=1e-5):
        '''
        Transform vis data to log scale for training
        '''
        pass

    def DataTransformInv(self, data, eps=1e-5):
        '''
        Inverse log scale transform
        '''
        pass

    def LoadCoord(self, normalize=True, extend=False):
        '''
        Load input coord for training/evaluation
        '''
        pass

    def CoordFromVoxID(self, idx, normalize=True):
        '''
        Load input coord from vox id 
        '''
        pass

    def VisibilityFromAxisID(self, axis_id, ch=None):
        pass

    def VisibilityFromXYZ(self, pos, ch=None):
        #used in flash_algo in "fill_estimate" method
        if not torch.is_tensor(pos):
            pos = torch.tensor(pos, device=device)
        return self.model(pos)['model_out']

    def Visibility(self, vids, ch=None):
        '''
        Returns a probability for a detector to observe a photon.
        If ch (=detector ID) is unspecified, returns an array of probability for all detectors
        INPUT
          vids - Tensor of integer voxel IDs
          ch  - Integer (valid range 0 to N-1) to specify an optical detector (optional)
        RETURN
          Probability(ies) in FP32 to observe a photon at a specified location for each vid
        '''
        #unclear whether default return of entire array is necessary
        #take input array of coordinates
        #return 1D array of visibilities at those coordinates
        result = []
        for vid in vids:
            result += self.VisibilityFromXYZ(vid)
        return result

    # def AxisID2VoxID(self, axis_id):
    #     '''
    #     Takes an integer ID for voxels along xyz axis (ix, iy, iz) and converts to a voxel ID
    #     INPUT
    #       axis_id - Length 3 integer array noting the position in discretized index along xyz axis
    #     RETURN
    #       The voxel ID (single integer)          
    #     '''
    #     pass

    def AxisID2Position(self, axis_id):
        '''
        Takes a axis ID (discretized location along xyz axis) and converts to a xyz position (x,y,z)
        INPUT
          axis_id - The axis ID in an integer array (ix,iy,iz)
        RETURN
          Length 3 floating point array noting the position along xyz axis
        '''    
        pass

    def Position2AxisID(self, pos):
        '''
        Takes a tensor of xyz position (x,y,z) and converts to a tensor of voxel IDs
        INPUT
          pos - Tensor of length 3 floating point array noting the position along xyz axis
        RETURN
          Tensor of sigle integer voxel IDs       
        '''
        pass

    def Position2VoxID(self, pos):
        '''
        Takes a tensor of xyz position (x,y,z) and converts to a tensor of voxel IDs
        INPUT
          pos - Tensor of length 3 floating point array noting the position along xyz axis
        RETURN
          Tensor of sigle integer voxel IDs       
        '''
        pass

    def VoxID2AxisID(self, vid):
        '''
        Takes a voxel ID and converts to discretized index along xyz axis
        INPUT
          vid - The voxel ID (single integer)          
        RETURN
          Length 3 integer array noting the position in discretized index along xyz axis
        '''
        pass

    def VoxID2Coord(self, vid):
        '''
        Takes a voxel ID and converts to normalized coordniate
        INPUT
          vid - The voxel ID (single integer)          
        RETURN
          Length 3 normalized coordinate array
        '''
        pass

