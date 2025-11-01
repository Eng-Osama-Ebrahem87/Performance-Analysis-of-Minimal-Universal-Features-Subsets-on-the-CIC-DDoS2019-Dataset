# The libraries we need .. . 


import os
import pandas as pd
import numpy as np



#Target_file_loc  = r"E:\CIC-IoT 2023\CIC-IoT2023 form kaggle\test_Pre.csv"
Target_file_loc  = r"E:\VeReMi Dataset\VeReMi and BSMList\Main_data_shuffled_CIC_Features.csv"




# **************************************  

## Reading a CSV file with low_memory set to False

Data_target_df = pd.read_csv(Target_file_loc, low_memory=False)

Data_target_df.info()  


print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. .. .. .. .")

#get file size  # Conversion to kilobytes, megabytes  .. .

file_size_bytes = os.path.getsize(Target_file_loc)
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024


print("Sample Size is :", file_size_mb, "MB")

#The problem is with csv files in CICDDos 2019, there are leading whistespaces in column names, because of which the key error is coming .. .

Data_target_df.columns = Data_target_df.columns.str.strip()


print("analyze class distribution ", Data_target_df.groupby("Label").size())

#print("analyze class distribution ", Data_target_df.groupby("label").size())

#print("analyze class distribution ", Data_target_df.groupby("binary_label_encoded").size())



print(Data_target_df.info())


print(" **************************************")

#y = Data_target_df['label']  


# label binary encoding: 

'''for index, value in y.items():
    if value != 1:
        value = 0'''
Data_target_df. loc[Data_target_df['Label'] != 1, 'Label'] = 0

Data_target_df.info()

print("analyze class distribution .. . After change encoding: \n ", Data_target_df.groupby("Label").size())


        
# save target dataframe to new location 
Data_target_df.to_csv(
         r"E:\VeReMi Dataset\VeReMi and BSMList\Main_data_shuffled_CIC_Features_to_binary.csv", 
        index=False)

 

        






















 
