from plot import plot_qcluster
from data_gen import DataGen
import plotly.express as px
import plotly.graph_objects as go

gen = DataGen()
match_input = gen.make_flashmatch_inputs()
np_result = None

#visualizing
def gen_plot():
    print(match_input.qcluster_v[0].qpt_v)
    q_graph = plot_qcluster(match_input.qcluster_v[0].qpt_v)
    return q_graph