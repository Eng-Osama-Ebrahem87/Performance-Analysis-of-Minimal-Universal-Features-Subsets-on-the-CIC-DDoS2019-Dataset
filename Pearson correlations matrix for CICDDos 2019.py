# The libraries we need

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

 
Target_file_loc = r"E:\Cic-DDos2019 Original\03-11\NetBIOS_Pre.csv" 

  

Data_target_df = pd.read_csv(Target_file_loc)

Data_target_df.info()

print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. .. .. .. .")

#The problem is with csv files in CICDDos 2019, there are leading whistespaces in column names, because of which the key error is coming.
Data_target_df.columns = Data_target_df.columns.str.strip()

Data_target_df = Data_target_df[[ 'Packet Length Mean', 'Average Packet Size', 'Bwd Packet Length Min', 'Fwd Packets/s' , 'Min Packet Length', 'Down/Up Ratio']]



# Find the pearson correlations matrix

print("\n Pearson correlations matrix: \n ")

corr = Data_target_df.corr(method = 'pearson')
#method : In method we can choose any one from {'pearson', 'kendall', 'spearman'}
#pearson is the standard correlation coefficient matrix i.e default

# Set display option to show all columns
pd.set_option('display.max_columns', None)

# Show the columns
print(corr ) 

#Plot the correlation matrix with the seaborn heatmap:

plt.figure(figsize=(8,10), dpi =100)
sns.heatmap(corr,annot=True,fmt=".2f", linewidth=.7)

plt.title("The Pearson Correlations Matrix for CICDDos 2019")
  
# displaying the plotted heatmap 
plt.show()





















  


