import os
import pandas as pd
from CaImaging.CellReg import *

project_path = r'Z:\Will\SEFL'
directories_path = os.path.join(project_path, 'ProjectDirectories.csv')

def project_dirs(path=directories_path):
    df = pd.read_csv(path)
    df.Session = df.Session.str.lower()
    df.Group = df.Group.str.lower()

    return df, path


def load_cellmap(sessions, detected='either_day'):
    cellreg_path = get_cellreg_path(sessions, sessions.Mouse.iloc[0])
    C = CellRegObj(cellreg_path)
    cell_map = trim_map(C.map, list(sessions.Session), detected=detected)

    return cell_map


def batch_load(sessions, datatype):
    session_paths = sessions.Path
    data = []
    for path in session_paths:
        minian = open_minian(path)
        data.append(np.asarray(minian[datatype]))

    return data