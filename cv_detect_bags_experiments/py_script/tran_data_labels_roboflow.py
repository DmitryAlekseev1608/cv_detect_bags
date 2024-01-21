import pandas as pd
import os
import numpy as np

names_files = os.listdir('/home/oem/Desktop/dataset_roboflow/val/labels_old')

for name_file in names_files:

    df_table = pd.read_table(f'/home/oem/Desktop/dataset_roboflow/val/labels_old/{name_file}', delim_whitespace=True, names=('CLASS', 'POINT1', 'POINT2', 'POINT3', 'POINT4'))
    for i in range(len(df_table)):
        if df_table['CLASS'].iloc[i] == 0:
            df_table['CLASS'].iloc[i] = 1

    #specify path for export
    path = f'/home/oem/Desktop/dataset_roboflow/val/labels_new/{name_file}'

    #export DataFrame to text file
    with open(path, 'a') as f:
        df_string = df_table.to_string(header=False, index=False)
        f.write(df_string)

