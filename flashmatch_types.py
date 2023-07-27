import itertools
import numpy as np
import copy
import torch
from scipy.optimize import linear_sum_assignment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlashMatchInput:
    def __init__(self):
        # input array of Flash
        self.flash_v = []
        # input array of QCluster
        self.qcluster_v = []
        # "RAW" QCluster (optional, may not be present, before x-shift)
        self.raw_qcluster_v = []
        # "RAW" flashmatch::QCluster_t (optional, may not be present, before active BB cut)
        self.all_pts_v = []
        # trajectory segment points
        self.track_v = []
        # True matches, an array of integer-pairs.
        self.true_match = []

class FlashMatch:
    def __init__(self, num_qclusters, num_flashes):
        self.loss_matrix = np.zeros((num_qclusters, num_flashes))
        self.reco_x_matrix = np.zeros((num_qclusters, num_flashes))
        self.reco_pe_matrix = np.zeros((num_qclusters, num_flashes))
        self.duration = np.zeros((num_qclusters, num_flashes))
    
    def bipartite_match(self):
        row_filter, col_filter, filtered_loss_matrix = self.filter_loss_matrix(700)
        row_idx, col_idx = linear_sum_assignment(filtered_loss_matrix)
        self.tpc_ids = row_filter[row_idx]
        self.flash_ids = col_filter[col_idx]
        self.loss_v = self.loss_matrix[self.tpc_ids, self.flash_ids]
        self.reco_x_v = self.reco_x_matrix[self.tpc_ids, self.flash_ids]
        self.reco_pe_v = self.reco_pe_matrix[self.tpc_ids, self.flash_ids]
        self.duration = self.duration[self.tpc_ids, self.flash_ids]

    def local_match(self):
        self.tpc_ids = np.arange(self.loss_matrix.shape[0])
        self.flash_ids = np.argmin(self.loss_matrix, axis = 1)
        self.loss_v = self.loss_matrix[self.tpc_ids, self.flash_ids]
        self.reco_x_v = self.reco_x_matrix[self.tpc_ids, self.flash_ids]
        self.reco_pe_v = self.reco_pe_matrix[self.tpc_ids, self.flash_ids]

    def global_match(self, loss_threshold):
        row_filter, col_filter, filtered_loss_matrix = self.filter_loss_matrix(loss_threshold)
        min_loss = np.inf

        num_tpc, num_pmt = filtered_loss_matrix.shape[0], filtered_loss_matrix.shape[1]
        col_idx = np.arange(num_pmt)
        for row_idx in itertools.product(np.arange(num_tpc), repeat=num_pmt):
            losses = filtered_loss_matrix[row_idx, col_idx]
            if np.sum(losses) < min_loss:
                min_loss = np.sum(losses)
                self.tpc_ids = row_idx
                self.flash_ids = col_idx

        self.tpc_ids = row_filter[self.tpc_ids]
        self.flash_ids = col_filter[self.flash_ids]

        self.loss_v = self.loss_matrix[self.tpc_ids, self.flash_ids]
        self.reco_x_v = self.reco_x_matrix[self.tpc_ids, self.flash_ids]
        self.reco_pe_v = self.reco_pe_matrix[self.tpc_ids, self.flash_ids]

    def filter_loss_matrix(self, loss_threshold):
        row_filter = []
        col_filter = []
        for i, row in enumerate(self.loss_matrix):
            if np.min(row) <= loss_threshold:
                row_filter.append(i)

        for j in range(self.loss_matrix.shape[1]):
            col = self.loss_matrix[:, j]
            if np.min(col) <= loss_threshold:
                col_filter.append(j)


        return np.array(row_filter), np.array(col_filter), self.loss_matrix[row_filter, :][:, col_filter]

class Flash:
    def __init__(self, *args):
        self.pe_v = []
        self.pe_true_v = []
        self.pe_err_v = []
        self.idx = np.inf    # index from original larlite vector
        self.time = np.inf   # Flash timing, a candidate T0
        self.time_true = np.inf  # MCFlash timing
        self.time_width = np.inf # flash time integration window
        self.dt_next = np.inf   # dt to next flash
        self.dt_prev = np.inf   # dt to previous flash

    def __len__(self):
        return len(self.pe_v)

    def to_torch(self):
        self.pe_v = torch.tensor(self.pe_v, device=device)
        self.pe_true_v = torch.tensor(self.pe_true_v, device=device)

    def sum(self):
        if len(self.pe_v) == 0:
            return 0
        return torch.sum(self.pe_v).item()

class QCluster:
    def __init__(self, *args):
        self.qpt_v = []
        self.idx = np.inf # index from original larlite vector
        self.time = np.inf # assumed time w.r.t trigger for reconstruction
        self.time_true = np.inf # time from MCTrack information
        self.xshift = 0

    def __len__(self):
        return len(self.qpt_v)

    def __iadd__(self, other):
        if len(self.qpt_v) == 0:
            return other.copy()
        else:
            self.qpt_v = torch.cat((self.qpt_v, other.qpt_v), 0)
        return self

    def copy(self):
        return copy.deepcopy(self)

    # total length of the track
    def length(self):
        res = 0
        for i in range(1, len(self.qpt_v)):
            res += torch.linalg.norm(self.qpt_v[i, :3] - self.qpt_v[i-1, :3]).item()
        return res

    # sum over charge 
    def sum(self):
        if len(self.qpt_v) == 0:
            return 0
        return torch.sum(self.qpt_v[:, -1]).item()

    # sum over x coordinates of the track
    def xsum(self):
        if len(self.qpt_v) == 0:
            return 0
        return torch.sum(self.qpt_v[:, 0]).item()

    # shift qcluster_v by given dx
    def shift(self, dx):
        other = copy.deepcopy(self)
        other.qpt_v[:, 0] += dx
        other.xmin += dx
        other.xmax += dx
        return other

    # fill qucluster content from a qcluster_v list
    def fill(self, qpt_v):
        self.qpt_v = torch.tensor(qpt_v, device=device)
        self.xmin = torch.min(self.qpt_v[:, 0]).item()
        self.xmax = torch.max(self.qpt_v[:, 0]).item()

    # drop points outside specified recording range
    def drop(self, x_min, x_max, y_min = -np.inf, y_max = np.inf, z_min = -np.inf, z_max = np.inf):
        mask = (self.qpt_v[:, 0] >= x_min) & (self.qpt_v[:, 0] <= x_max) & \
            (self.qpt_v[:, 1] >= y_min) & (self.qpt_v[:, 1] <= y_max) & \
            (self.qpt_v[:, 2] >= z_min) & (self.qpt_v[:, 2] <= z_max)
        self.qpt_v = self.qpt_v[mask]