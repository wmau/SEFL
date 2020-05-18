import matplotlib.pyplot as plt
import numpy as np
import cv2 

from CaImaging.CellReg import *
from CaImaging.util import bin_transients, filter_sessions
from SEFL_utils import batch_load, load_cellmap, project_dirs


def session_similarity(sessions, bin_size_in_seconds=30, ax=None):
    """
    Compute a correlation matrix between 2+ sessions based on population
    vectors made up of # of transients per bin.

    :parameters
    ---
    sessions: pandas Series
        List of sessions to compare.

    bin_size_in_seconds: int
        Number of seconds per bin.

    return
    """

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

    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(r, cmap='bwr')
    ax.clim([-1, 1])

    return r


def transient_rate_change(sessions, bin_size_in_seconds=30, ax=None):
    # Get cell registration.
    cell_map = load_cellmap(sessions, detected='everyday')

    # Load each session.
    data = batch_load(sessions, 'S')

    # Rearrange neurons based on registration mappings.
    # Then bin the transients and take the mean across time.
    data = rearrange_neurons(cell_map, data)
    transients = []
    for session in data:
        transients.append(np.mean(bin_transients(session,
                                                 bin_size_in_seconds),
                                  axis=1))

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(np.vstack(transients), 'k.-')
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels(sessions.Session)
    ax.set_ylabel('Average number of transients')

    pass

if __name__ == '__main__':
    df, path = project_dirs()
    keys = ['Mouse', 'Session']
    keywords = ['pp1', 'Baseline', 'TraumaTest']
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
