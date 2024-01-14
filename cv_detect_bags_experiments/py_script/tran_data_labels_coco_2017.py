import pandas as pd
import os

names_files = os.listdir('/home/oem/Desktop/dataset_coco/data/train/labels_old')

for name_file in names_files:

    df_table_old = pd.read_table(f'data/train/labels_old/{name_file}', delim_whitespace=True, names=('CLASS', 'POINT1', 'POINT2', 'POINT3', 'POINT4'))
    df_table_new = pd.DataFrame(columns=('CLASS', 'POINT1', 'POINT2', 'POINT3', 'POINT4'))

    if not df_table_old[df_table_old['CLASS']==24].empty or not df_table_old[df_table_old['CLASS']==26].empty or not df_table_old[df_table_old['CLASS']==28].empty:
        df_table_new = pd.concat([df_table_new, df_table_old[df_table_old['CLASS']==24]], ignore_index=True)
        df_table_new = pd.concat([df_table_new, df_table_old[df_table_old['CLASS']==26]], ignore_index=True)
        df_table_new = pd.concat([df_table_new, df_table_old[df_table_old['CLASS']==28]], ignore_index=True)
        df_table_new["CLASS"] = 0
        #specify path for export
        path = f'data/train/labels_new/{name_file}'

        #export DataFrame to text file
        with open(path, 'a') as f:
            df_string = df_table_new.to_string(header=False, index=False)
            f.write(df_string)  
    else:
        continue