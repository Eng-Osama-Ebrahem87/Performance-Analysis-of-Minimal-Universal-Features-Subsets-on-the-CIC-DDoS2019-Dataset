import pandas as pd
import numpy as np
import os

# import NearMiss  for sampling
from imblearn.under_sampling import NearMiss


Target_file_loc = r"E:\Cic-DDos2019 Original\03-11\Portmap_Pre.csv"

#Target_file_loc = r"E:\Cic-DDos2019 Original\03-11\NetBIOS_Pre.csv"

#Target_file_loc = r"E:\Cic-DDos2019 Original\03-11\MSSQL_Pre.csv"

#Target_file_loc = r"E:\Cic-DDos2019 Original\03-11\Syn_Pre.csv"

# **************************************  

## Reading a CSV file with low_memory set to False

Data_target_df = pd.read_csv(Target_file_loc, low_memory=False)

Data_target_df.info()

print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. .. .. .. .")

#get file size  and # Conversion to kilobytes, megabytes

file_size_bytes = os.path.getsize(Target_file_loc)
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024


print("File Size is :", file_size_mb, "MB")


#The problem is with csv files in CICDDos 2019, there are leading whistespaces in column names, because of which the key error is coming .. .

Data_target_df.columns = Data_target_df.columns.str.strip()

print("analyze class distribution ", Data_target_df.groupby("Label").size())

print(" **************************************")

################################################################

x=Data_target_df.drop(["Label"],axis=1)
y=Data_target_df["Label"]


# Set up the undersampling method
undersampler = NearMiss(version=1, n_neighbors=3)


# Apply the transformation to the dataset
x, y = undersampler.fit_resample(x, y)

y.value_counts()

print(y.value_counts())

# the default behaviour is join='outer'
# inner join

final_df = pd.concat([x, y], axis=1, join='inner') 

###################################################################

# save Balaced dataframe to new location

final_df.to_csv(
        r"E:\Cic-DDos2019 Original\03-11\Portmap_undersampling.csv",
        index=False)










