import os
from os.path import dirname, expanduser
import pandas as pd

home = expanduser('~')

arax_file_path = os.path.join(home, "OneDrive - miDIAGNOSTICS\\Desktop\\Team\\Projects\\NA\\PCR die\\ARAX")
arax_full_wafer_excel_file_name = 'Arax PCR.xlsx'
excel_file_path = os.path.join(arax_file_path, arax_full_wafer_excel_file_name)

if __name__ == '__main__':
    df_overview = pd.read_excel(io=excel_file_path, sheet_name='PCR')
