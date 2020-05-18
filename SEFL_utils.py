import os
import pandas as pd
from CaImaging.CellReg import *

project_path = r'Z:\Will\SEFL'
directories_path = os.path.join(project_path, 'ProjectDirectories.csv')

def project_dirs(path=directories_path):
    """
    Get the DataFrame containing metadata for each session.

    :parameter
    ---
    path: str
        Directory to csv containing metadata.

    :return
    ---
    sessions: DataFrame
        With columns: Mouse, Group, Session, Path, CellRegPath.

    path: str
        Directory to csv containing metadata.
    """
    sessions = pd.read_csv(path)
    sessions.Session = sessions.Session.str.lower()
    sessions.Group = sessions.Group.str.lower()

    return sessions, path


def load_cellmap(sessions, detected='either_day'):
    """
    Load the cell registration map for this mouse.

    :parameters
    ---
    sessions: DataFrame
        Metadata.

    detected: str
        'either_day', 'everyday', or 'first_day'. Determines how to trim
        cell map. If 'everyday', cells must be active everyday to be considered,
        etc.

    :return
    ---
    cell_map: DataFrame
        Cell registration mappings.

    """
    cellreg_path = get_cellreg_path(sessions, sessions.Mouse.iloc[0])
    C = CellRegObj(cellreg_path)
    cell_map = trim_map(C.map, list(sessions.Session), detected=detected)

    return cell_map


def batch_load(sessions, datatype):
    """
    Load a bunch of sessions' data.

    :parameter
    ---
    sessions: DataFrame
        Metadata.

    datatype: str
        Dict key for minian data object (e.g., 'S').

    :return
    ---
    data: list of arrays
        Data arrays requested.
    """
    session_paths = sessions.Path
    data = []
    for path in session_paths:
        minian = open_minian(path)
        data.append(np.asarray(minian[datatype]))

    return data