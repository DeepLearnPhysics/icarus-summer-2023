import h5py  as h5
import numpy as np
import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PhotonLibrary(object):
    def __init__(self, fname='photon_library/plib_combined.h5', lut_file=None):
        if not os.path.isfile(fname):
            print('Downloading photon library file... (>300MByte, may take minutes')
            os.system('curl -O https://www.nevis.columbia.edu/~kazuhiro/plib.h5 ./')
        if not os.path.isfile(fname):
            print('Error: failed to download the photon library file...')
            raise Exception

        with h5.File(fname,'r') as f:
            self._vis  = torch.from_numpy(np.array(f['vis'], dtype=np.float32)).to(device)
            self._min  = torch.tensor(f['min'], dtype=torch.float32).to(device)
            self._max  = torch.tensor(f['max'], dtype=torch.float32).to(device)
            self.shape = torch.tensor(f['numvox'], dtype=torch.float32).to(device)

        self.gap = (self._max[0] - self._min[0]) / self.shape[0] # x distance between adjacent voxels
        self._max[0] += 10
        self._min[0] += 10

        # Load weighting look up table if provided
        if lut_file:
            with h5.File(lut_file,'r') as f:
                self.lut = torch.tensor(f['lut'], dtype=torch.float32).to(device)
                self.bins = torch.tensor(f['bins'], dtype=torch.float32).to(device)
                self.pmt_groups = torch.cat((torch.zeros(90), torch.ones(90))).to(device)
        
    def LoadData(self, transform=True, eps=1e-5):
        '''
        Load photon library visibility data. Apply scale transform if specified
        '''
        data = self._vis
        if transform:
            data = self.DataTransform(data, eps)

        return data

    def DataTransform(self, data, eps=1e-5):
        '''
        Transform vis data to log scale for training
        '''
        v0 = np.log10(eps)
        v1 = np.log10(1.+eps)
        return (torch.log10(data+eps) - v0) / (v1 - v0)

    def DataTransformInv(self, data, eps=1e-5):
        '''
        Inverse log scale transform
        '''
        v0 = np.log10(eps)
        v1 = np.log10(1.+eps)
        return torch.pow(10, data * (v1 - v0) + v0) - eps

    def LoadCoord(self, normalize=True, extend=False):
        '''
        Load input coord for training/evaluation
        '''
        vox_ids = torch.arange(self._vis.shape[0]).to(device)
        
        return self.CoordFromVoxID(vox_ids, normalize=normalize)

    def CoordFromVoxID(self, idx, normalize=True):
        '''
        Load input coord from vox id 
        '''
        if np.isscalar(idx):
            idx = np.array([idx])
        
        pos_coord = self.VoxID2Coord(idx)       
        if normalize:
            pos_coord = 2 * (pos_coord - 0.5)
        
        return pos_coord.squeeze()

    def VisibilityFromAxisID(self, axis_id, ch=None):
        return self.Visibility(self.AxisID2VoxID(axis_id),ch)

    def VisibilityFromXYZ(self, pos, ch=None):
        if not torch.is_tensor(pos):
            pos = torch.tensor(pos, device=device)
        return self.Visibility(self.Position2VoxID(pos), ch)

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
        if ch is None:
            return self._vis[vids]
        return self._vis[vids][ch]

    def AxisID2VoxID(self, axis_id):
        '''
        Takes an integer ID for voxels along xyz axis (ix, iy, iz) and converts to a voxel ID
        INPUT
          axis_id - Length 3 integer array noting the position in discretized index along xyz axis
        RETURN
          The voxel ID (single integer)          
        '''
        return axis_id[:, 0] + axis_id[:, 1]*self.shape[0] + axis_id[:, 2]*(self.shape[0] * self.shape[1])

    def AxisID2Position(self, axis_id):
        '''
        Takes a axis ID (discretized location along xyz axis) and converts to a xyz position (x,y,z)
        INPUT
          axis_id - The axis ID in an integer array (ix,iy,iz)
        RETURN
          Length 3 floating point array noting the position along xyz axis
        '''    
        return self._min + (self._max - self._min) / self.shape * (axis_id + 0.5)

    def Position2AxisID(self, pos):
        '''
        Takes a tensor of xyz position (x,y,z) and converts to a tensor of voxel IDs
        INPUT
          pos - Tensor of length 3 floating point array noting the position along xyz axis
        RETURN
          Tensor of sigle integer voxel IDs       
        '''
        return torch.floor((pos - self._min) / (self._max - self._min) * self.shape)

    def Position2VoxID(self, pos):
        '''
        Takes a tensor of xyz position (x,y,z) and converts to a tensor of voxel IDs
        INPUT
          pos - Tensor of length 3 floating point array noting the position along xyz axis
        RETURN
          Tensor of sigle integer voxel IDs       
        '''
        axis_ids = ((pos - self._min) / (self._max - self._min) * self.shape).int()

        return (axis_ids[:, 0] + axis_ids[:, 1] * self.shape[0] +  axis_ids[:, 2]*(self.shape[0] * self.shape[1])).long()

    def VoxID2AxisID(self, vid):
        '''
        Takes a voxel ID and converts to discretized index along xyz axis
        INPUT
          vid - The voxel ID (single integer)          
        RETURN
          Length 3 integer array noting the position in discretized index along xyz axis
        '''
        xid = vid.int() % self.shape[0]
        yid = ((vid - xid) / self.shape[0]).int() % self.shape[1]
        zid = ((vid - xid - (yid * self.shape[0])) / (self.shape[0] * self.shape[1])).int() % self.shape[2]
        
        return torch.reshape(torch.stack([xid,yid,zid], -1), (-1, 3)).float()

    def VoxID2Coord(self, vid):
        '''
        Takes a voxel ID and converts to normalized coordniate
        INPUT
          vid - The voxel ID (single integer)          
        RETURN
          Length 3 normalized coordinate array
        '''
        axis_id = self.VoxID2AxisID(vid)
        
        return (axis_id + 0.5) / self.shape

    def WeightFromPos(self, pos):
        '''
          Weighting factor for data at pos based on the provided weight lut file
        '''
        vis = self.VisibilityFromXYZ(pos)
        weight = vis * 1e6
        weight[vis==0] = 1.
        
        return torch.mean(weight, 0)