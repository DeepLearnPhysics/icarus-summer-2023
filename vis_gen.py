import numpy as np
from lightpath import LightPath
from flashalgo import FlashAlgo
from flashmatch_types import FlashMatchInput, Flash
from plot import plot_qcluster
import yaml
from photon_library import PhotonLibrary
from points import scatter_points
from data_gen import DataGen

gen = DataGen()
match_input = gen.make_flashmatch_inputs()
np_result = None

#visualizing
def gen_plot():
    print(match_input.qcluster_v[0].qpt_v)
    q_graph = plot_qcluster(match_input.qcluster_v[0].qpt_v)
    return q_graph

gen_plot()