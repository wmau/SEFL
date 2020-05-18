import matplotlib.pyplot as plt

from CaImaging.CellReg import *
from CaImaging.util import bin_transients, filter_sessions
from SEFL_utils import batch_load, load_cellmap, project_dirs
from CaImaging.Assemblies import lapsed_activation

project_df, project_path = project_dirs()

#Depracated.
def EnsembleReactivation(mouse, sessions):
    """
    Plots activation of all assemblies across multiple sessions in a
    somewhat messy way.

    :param mouse:
    :param sessions:
    :return:
    """
    keys = ['Mouse', 'Session']
    keywords = sessions
    keywords.append(mouse)
    sessions = filter_sessions(project_df, keys, keywords, 'all')

    spikes = batch_load(sessions, 'S')
    cell_map = load_cellmap(sessions, detected='everyday')

    lapsed = rearrange_neurons(cell_map[cell_map.columns[1]],
                               spikes[1])
    template = rearrange_neurons(cell_map[cell_map.columns[0]],
                                 spikes[0])

    activations, patterns, sorted_spikes, fig, axes = \
        lapsed_activation(template[0], lapsed, percentile=99.9,
                          plot=False)

    pass

if __name__ == '__main__':
    EnsembleReactivation('pp1', ['TraumaEnd', 'TraumaPost'])