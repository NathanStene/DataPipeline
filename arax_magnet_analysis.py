import pandas as pd
import tkinter as tk
from tkinter import filedialog
from constants import arax_magnets_column_names
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
from os.path import dirname, expanduser

home = expanduser('~')
working_dir = os.getcwd()

arax_file_path = os.path.join(home, "miDIAGNOSTICS\Michela Cagna - ARAX")
arax_full_wafer_excel_file_name = 'Arax-diced, comments KAMILA.xlsx'
excel_file_path = os.path.join(arax_file_path, arax_full_wafer_excel_file_name)

df = pd.read_excel(io=excel_file_path, sheet_name=1, header=2)

df.drop(labels=df.columns[[0, 1]], inplace=True, axis=1)
df.drop(labels=df.columns[18:], inplace=True, axis=1)
df.columns = arax_magnets_column_names

f = lambda x: x.split(' ')[0] if isinstance(x, str) else x
g = lambda x: 2 if x == 'PCR1' else (3 if x == 'PCR2' else x)

df['die'] = df['die'].apply(g)

df['bubble_inlet'] = df['bubble_inlet'].apply(f)
df['bubble_pre'] = df['bubble_pre'].apply(f)

df['magnet'] = 0
df['magnet'].iloc[20:] = 1

rows = list(np.linspace(0, 9, 10).astype(int)) + list(np.linspace(40, 81, 82-40).astype(int))
df['swab'] = 0
df['swab'].iloc[rows] = 1

df['testing_level'] = 'die'

df.to_pickle(os.path.join(working_dir, 'pickles\\arax_beads.pkl'))