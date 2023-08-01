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

go_obj = gen_plot()

fig = px.scatter_3d(go_obj.voxels, x = go_obj.voxels[:, 0], y = go_obj.voxels[:, 1], z = go_obj.voxels[:, 2], color = 'species')
fig.show()