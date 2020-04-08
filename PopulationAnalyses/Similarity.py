from CaImaging.CellReg import *
import numpy as np
from SEFL_utils import project_dirs, load_cellmap, batch_load
from CaImaging.util import bin_transients, filter_sessions
import matplotlib.pyplot as plt

def session_similarity(sessions, bin_size_in_seconds=30):

    # Get cell registration.
    cell_map = load_cellmap(sessions, detected='either_day')

    # Load each session.
    data = batch_load(sessions, 'S')

    # Rearrange neurons based on registration mappings.
    # Then bin the transients.
    data = rearrange_neurons(cell_map, data)
    transients = []
    for session in data:
        transients.append(bin_transients(session, bin_size_in_seconds))

    # Do every correlation between time bins.
    r = np.corrcoef(np.hstack(transients).T)
    plt.imshow(r, cmap='bwr')
    plt.clim([-1, 1])
    plt.show()
    pass


def transient_rate_change(sessions, bin_size_in_seconds=30):
    cell_map = load_cellmap(sessions, detected='everyday')
    data = batch_load(sessions, 'S')

    data = rearrange_neurons(cell_map, data)
    transients = []
    for session in data:
        transients.append(np.mean(bin_transients(session, bin_size_in_seconds), axis=1))
    plt.plot(np.vstack(transients), '.-')

    pass

if __name__ == '__main__':
    df, path = project_dirs()
    keys = ['Mouse', 'Session']
    keywords = ['pp2', 'TraumaTest', 'MildStressor', 'SEFLtest']
    sessions = filter_sessions(df, keys, keywords, 'all')

    # minian_outputs = []
    # for session in sessions:
    #     minian_outputs.append(open_minian(session))
    #
    # C = CellRegObj(r'Z:\Will\SEFL\pp2\SpatialFootprints\CellRegResults')
    # cell_map = trim_map(C.cell_map, [0, 1], detected='everyday')
    # template = np.asarray(minian_outputs[0].S)
    # lapsed = rearrange_neurons(cell_map[:,1], [np.asarray(minian_outputs[1].S)])
    # template = rearrange_neurons(cell_map[:,0], [template])
    #
    # lapsed_activation(template[0], lapsed)

    #session_similarity(sessions)
    transient_rate_change(sessions)
    pass