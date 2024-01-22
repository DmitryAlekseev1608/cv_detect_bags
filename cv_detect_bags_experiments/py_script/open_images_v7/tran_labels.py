import pandas as pd
import os
import numpy as np

names_files = os.listdir('/home/oem/Desktop/datasets/open-images-v7/labels/val_old')

for name_file in names_files:
    
    df_table = pd.read_table(f'/home/oem/Desktop/datasets/open-images-v7/labels/val_old/{name_file}', delim_whitespace=True, names=('CLASS', 'POINT1', 'POINT2', 'POINT3', 'POINT4'))
    df_table_new = pd.DataFrame(columns=('CLASS', 'POINT1', 'POINT2', 'POINT3', 'POINT4'))

    for i in range(len(df_table)):
        if df_table['CLASS'].iloc[i] == 6:
            df_table_new = pd.concat([df_table_new, df_table[df_table['CLASS']==6]], ignore_index=True)
            df_table_new['CLASS'].mask(df_table_new['CLASS']==6, 0, inplace=True)
        elif df_table['CLASS'].iloc[i] == 30:
            df_table_new = pd.concat([df_table_new, df_table[df_table['CLASS']==30]], ignore_index=True)            
            df_table_new['CLASS'].mask(df_table_new['CLASS']==30, 1, inplace=True)
        elif df_table['CLASS'].iloc[i] == 100:
            df_table_new = pd.concat([df_table_new, df_table[df_table['CLASS']==100]], ignore_index=True)
            df_table_new['CLASS'].mask(df_table_new['CLASS']==100, 1, inplace=True)      
        elif df_table['CLASS'].iloc[i] == 138:
            df_table_new = pd.concat([df_table_new, df_table[df_table['CLASS']==138]], ignore_index=True)   
            df_table_new['CLASS'].mask(df_table_new['CLASS']==138, 1, inplace=True)
        elif df_table['CLASS'].iloc[i] == 169:
            df_table_new = pd.concat([df_table_new, df_table[df_table['CLASS']==169]], ignore_index=True)  
            df_table_new['CLASS'].mask(df_table_new['CLASS']==169, 1, inplace=True)
        elif df_table['CLASS'].iloc[i] == 174:
            df_table_new = pd.concat([df_table_new, df_table[df_table['CLASS']==174]], ignore_index=True) 
            df_table_new['CLASS'].mask(df_table_new['CLASS']==174, 1, inplace=True)
        elif df_table['CLASS'].iloc[i] == 212:
            df_table_new = pd.concat([df_table_new, df_table[df_table['CLASS']==212]], ignore_index=True) 
            df_table_new['CLASS'].mask(df_table_new['CLASS']==212, 2, inplace=True)

    #specify path for export
    path = f'/home/oem/Desktop/datasets/open-images-v7/labels/val_new/{name_file}'

    #export DataFrame to text file
    with open(path, 'a') as f:
        df_string = df_table_new.to_string(header=False, index=False)
        f.write(df_string)