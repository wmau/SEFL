import matplotlib.pyplot as plt

from CaImaging.CellReg import *
from CaImaging.util import bin_transients, filter_sessions
from SEFL_utils import batch_load, load_cellmap, project_dirs
from CaImaging.Assemblies import lapsed_activation

def EnsembleReactivation(mouse, sessions):
    df, path = project_dirs()
    keys = ['Mouse', 'Session']
    keywords = sessions
    keywords.append(mouse)
    sessions = filter_sessions(df, keys, keywords, 'all')

    minian_outputs = []
    for session_path in sessions['Path']:
        minian_outputs.append(open_minian(session_path))

    C = CellRegObj(sessions['CellRegPath'].iloc[0])
    cell_map = trim_map(C.map, list(sessions['Session']), detected='everyday')

    lapsed = rearrange_neurons(cell_map[cell_map.columns[1]],
                               [np.asarray(minian_outputs[1].S)])
    template = rearrange_neurons(cell_map[cell_map.columns[0]],
                                 [np.asarray(minian_outputs[0].S)])

    lapsed_activation(template[0], lapsed)

    pass

if __name__ == '__main__':
    EnsembleReactivation('pp1', ['TraumaEnd', 'TraumaPost'])