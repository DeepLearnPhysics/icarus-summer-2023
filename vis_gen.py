from plot import plot_qcluster
from data_gen import DataGen
import plotly.express as px
import plotly.graph_objects as go
import yaml
import numpy as np
from points import scatter_points

gen = DataGen()
match_input = gen.make_flashmatch_inputs()
np_result = None

#visualizing
def gen_qcluster_plot():
    print(match_input.qcluster_v[1].qpt_v)
    q_graph = plot_qcluster(match_input.qcluster_v[1].qpt_v)
    return q_graph

def make_flash_plot(flash, x=None, **kwargs):
    detector = yaml.load(open("icarus-summer-2023/detector_specs.yml"), Loader=yaml.Loader)['DetectorSpecs']
    nopch = detector['PhotonLibraryNOpDetChannels']
    pmt_positions = []
    if x is not None and not isinstance(x, float):
        raise Exception('x needs to be a float')
    
    for pmt in range(nopch):
        pmt_position = detector['PMT' + str(pmt)]
        if x is not None and isinstance(x, float):
            if not np.isclose(pmt_position[0], x):
                continue
        pmt_positions.append(np.array([[pmt_position[0], pmt_position[1], pmt_position[2], flash[pmt]]]))
    pmt_positions = np.vstack(pmt_positions)
    if x is None:
        return scatter_points(pmt_positions[:, 0:3], color=pmt_positions[:, -1], dim=3, markersize=3, **kwargs)
    else:
        return scatter_points(pmt_positions[:, [2, 1]], color=pmt_positions[:, -1], dim=2, markersize=15, **kwargs)
    
def gen_flash_plot():
    f_graph = make_flash_plot(match_input.flash_v[0].pe_v)
    return f_graph