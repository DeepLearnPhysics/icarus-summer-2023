import numpy as np
import yaml
from ..flashmatch_types import QCluster
import torch

class LightPath():
    def __init__(self, detector_specs, cfg_file=None):
        self.gap = 0.5
        self.dEdxMIP = detector_specs['MIPdEdx']
        self.light_yield = detector_specs['LightYield']
        if cfg_file:
            self.configure(cfg_file)

    def configure_from_yaml(self, cfg_file):
        self.configure(yaml.load(open(cfg_file), Loader=yaml.Loader)["LightPath"])
        
    def configure(self,cfg):
        self.gap = cfg["LightPath"]["SegmentSize"]
    
    def fill_qcluster(self, pt1, pt2, res):
        """
        Make a qcluster instance based given trajectory points and detector specs
        ---------
        Arguments
          pt1, pt2: 3D trajectory points
          dedx: stopping power
          light yield: light yield
        -------
        Returns
        """
        norm_alg = np.linalg.norm
        if type(pt1) == type(torch.tensor([])):
            norm_alg = torch.linalg.norm
            
        qpt_v=[]
        
        dist = norm_alg(pt1 - pt2)
        # segment less than gap threshold
        if dist < self.gap:
            mid_pt = (pt1 + pt2) / 2 
            q = self.light_yield * self.dEdxMIP * dist
            qpt_v.append([mid_pt[0], mid_pt[1], mid_pt[2], q])
            return
        # segment larger than gap threshold
        num_div = int(dist / self.gap)
        direct = (pt1 - pt2) / dist

        for div_idx in range(num_div+1):
            if div_idx < num_div:
                mid_pt = pt2 + direct * (self.gap * div_idx + self.gap / 2.)
                q = self.light_yield * self.dEdxMIP * self.gap
                qpt_v.append([mid_pt[0], mid_pt[1], mid_pt[2], q])
            else:
                weight = (dist - int(dist / self.gap) * self.gap)
                mid_pt = pt2 + direct * (self.gap * div_idx + weight / 2.)
                q = self.light_yield * self.dEdxMIP * weight
                qpt_v.append([mid_pt[0], mid_pt[1], mid_pt[2], q])

        res.qpt_v = torch.concat([res.qpt_v,torch.tensor(qpt_v,device=res.qpt_v.device)])
        
    def make_qcluster_from_track(self, track):
        """
        Create a qcluster instance from a trajectory
        ---------
        Arguments
          track: array of trajectory points
        -------
        Returns
          a qcluster instance
        """
        res = QCluster()
        #qpt_v = []
        
        # add first point of trajectory
        res.qpt_v = torch.tensor([track[0][0], track[0][1], track[0][2], 0.],
                                 device=res.qpt_v.device,
                                ).reshape(1,-1)

        for i in range(len(track)-1):
            self.fill_qcluster(np.array(track[i]), np.array(track[i+1]), res)

        # add last point of trajectory
        res.qpt_v = torch.concat([res.qpt_v, torch.tensor([track[-1][0], track[-1][1], track[-1][2], 0.], 
                                                          device=res.qpt_v.device,
                                                         ).reshape(1,-1)])

        return res