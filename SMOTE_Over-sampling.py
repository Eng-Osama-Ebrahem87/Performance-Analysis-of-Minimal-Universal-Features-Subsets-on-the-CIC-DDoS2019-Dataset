
import pandas as pd
import numpy as np
import os

# import SMOTE for sampling
from imblearn.over_sampling import SMOTE, ADASYN  



Target_file_loc = r"E:\Cic-DDos2019 Original\03-11\Portmap.csv"

#Target_file_loc = r"E:\Cic-DDos2019 Original\03-11\NetBIOS.csv"

#Target_file_loc = r"E:\Cic-DDos2019 Original\03-11\MSSQL.csv" 

#Target_file_loc = r"E:\Cic-DDos2019 Original\03-11\Syn.csv"


#**************************************  

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




######################################
#Data Cleaning and Feature Engineering

#There are some columns that are not really useful and hence we will proceed to drop them.
#Also, there are some missing values so let’s drop all those rows with empty values:


# drop the column("Flow ID") and other metadata features  .. .

print("drop the column Flow ID and other metadata features  .. .")

Data_target_df.drop(labels=["Flow ID"],axis=1,inplace=True)
Data_target_df.drop(labels=["Source IP"],axis=1,inplace=True)
Data_target_df.drop(labels=["Destination IP"],axis=1,inplace=True)
Data_target_df.drop(labels=["Source Port"],axis=1,inplace=True)
Data_target_df.drop(labels=["Destination Port"],axis=1,inplace=True)
Data_target_df.drop(labels=["Timestamp"],axis=1,inplace=True)


print(" **************************************")

    
print("DataFrame  after modified  >>> ")    
Data_target_df.head()

# The noisy data rectifying step:
#Removing duplicate records can help reduce noise and redundancy in our dataset.
# Remove duplicate rows:  
Data_target_df = Data_target_df.drop_duplicates()


# The null value data rectifying step:
# fill null values
for col in Data_target_df.columns:
    Data_target_df[col] = Data_target_df[col].fillna(Data_target_df[col].mode()[0])
    
#Removing rows or columns with a significant amount of missing data. 
#Remove rows with missing values:  
Data_target_df = Data_target_df.dropna()


# The infinity data values rectifying step:
Data_target_df.replace([np.inf, -np.inf], np.nan)
Data_target_df.dropna(inplace=True)


# label encoding
from sklearn.preprocessing import LabelEncoder
for col in Data_target_df.columns:
    le = LabelEncoder()
    Data_target_df[col] = le.fit_transform(Data_target_df[col])
Data_target_df.info()



################################################################
x=Data_target_df.drop(["Label"],axis=1)
y=Data_target_df["Label"]

# smote=SMOTE()

smote=SMOTE(sampling_strategy='minority') 
x,y=smote.fit_resample(x,y)
y.value_counts()

print(y.value_counts())

# the default behaviour is join='outer'
# inner join

final_df = pd.concat([x, y], axis=1, join='inner') 

 
###################################################################

# save Balaced dataframe to new location .. 
final_df.to_csv(
        r"E:\Cic-DDos2019 Original\03-11\Portmap_balanced.csv",
        index=False)









