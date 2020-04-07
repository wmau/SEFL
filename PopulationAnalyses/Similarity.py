from CaImaging.CellReg import *
import numpy as np
from SEFL_utils import project_dirs
from CaImaging.util import bin_transients, filter_sessions


def session_similarity(sessions, bin_size_in_seconds=30):

    # Get cell registration.
    cellreg_path = get_cellreg_path(sessions, sessions.loc[0, 'Mouse'])
    C = CellRegObj(cellreg_path)
    cell_map = trim_map(C.map, list(sessions.Session))

    session_paths = sessions.Path
    data = []
    for path, session_column in zip(session_paths, list(cell_map)):
        minian = open_minian(path)
        S = np.asarray(minian.S)
        cells = cell_map[session_column]
        data.append(bin_transients(S[cells], bin_size_in_seconds))

    pass

if __name__ == '__main__':
    df, path = project_dirs()
    keys = ['Mouse', 'Session']
    keywords = ['pp1', 'Baseline', 'TraumaPost']
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
    session_similarity(sessions)
    pass