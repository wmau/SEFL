import os
import pandas as pd

project_path = r'Z:\Will\SEFL'
directories_path = os.path.join(project_path, 'ProjectDirectories.csv')

def project_dirs(path=directories_path):
    df = pd.read_csv(path)

    return df, path