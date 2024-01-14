import pandas as pd
import os

names_files = os.listdir('/home/oem/Desktop/alex_data/test/labels_old')

for name_file in names_files:

    df_table = pd.read_table(f'/home/oem/Desktop/alex_data/test/labels_old/{name_file}', delim_whitespace=True, names=('CLASS', 'POINT1', 'POINT2', 'POINT3', 'POINT4'))
    df_table["CLASS"] = 0
    #specify path for export
    path = f'/home/oem/Desktop/alex_data/test/labels_new/{name_file}'

    #export DataFrame to text file
    with open(path, 'a') as f:
        df_string = df_table.to_string(header=False, index=False)
        f.write(df_string)  
