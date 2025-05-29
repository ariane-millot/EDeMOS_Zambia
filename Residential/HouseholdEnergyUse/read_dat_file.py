#%%
import pandas as pd
from pandas import read_fwf
import numpy as np

# This script reads data files from the DHS survey
# https://dhsprogram.com/data/dataset/Zambia_Standard-DHS_2018.cfm
# and then outputs selected columns to a csv file

#%%

# Folder where data files are located
data_folder = '../Data/DHSSurvey/'

# Choose the file prefix.
filename = 'KEHR8CFL'

# Full path to the Stata file
stata_file = f"{data_folder}{filename}.dta"

# Read the Stata file using pandas
df = pd.read_stata(stata_file)

# Print the first few rows of the dataframe to inspect the data
print(df.head())

# Print the number of columns in the dataframe
print(f"Number of columns: {df.shape[1]}")


#%%

# N is the number of columns in the file.
# This currently needs to be found before running this script by inspecting the files in a text editor
N = df.shape[1]

#%%

# Read the .DCT file which specifies which characters of the .DAT file encode which columns
# replace - with 3 spaces in the .dct file before running the code
dct = read_fwf(data_folder+filename+'.DCT', skiprows=2, header=None, nrows=N)
print('Read .dct file: shape =', dct.shape)
# Read the names of the columns from the .DO file
do = read_fwf(data_folder+filename+'.DO', skiprows=2, header=None, nrows=N)
print('Read .do file: shape =', do.shape)

#%%
# Append the column names to the character positions
dct = dct.merge(do,left_on=1,right_on=2,how='left')

#%%
# Make array of column names
#cols = np.array(dct[dct.columns[[4, 5]]]) original
cols = np.array(dct[dct.columns[[6, 7]]])

#%%

# Reduce the character positions by 1 because python starts counting from 0 (not from 1).
cols[:, 0]-=1
#%%

# Now read .DAT file containing encoded data
data = read_fwf(data_folder+filename+'.DAT',header=None, colspecs=list(map(tuple,cols)))

#%%
# Get rid of any duplicated quotation marks in column names
data.columns = np.char.strip(dct.iloc[:,-1].to_numpy(str),'"')

# List of useful columns for this project
cols_required = ["Sample strata for sampling errors",
                 "Household sample weight (6 decimals)",
                 "Stratification used in sample design",
                 "Province",
                 "Number of household members",
                 "Type of place of residence",
                 "Has electricity",
                 "Has radio",
                 "Has mobile telephone",
                 "Has television",
                 "Has refrigerator",
                 "Has a computer",
                 "Washing machine",
                 "Air conditioner",
                 "Generator",
                 "Microwave",
                 "Wealth index combined",
                 "Wealth index factor score combined (5 decimals)",
                 "Wealth index for urban/rural",
                 "Wealth index factor score for urban/rural (5 decimals)"]

# Define a dictionary to map old names to new names
old_to_new_names = {
    "Has electricity": "Electricity",
    "Has radio": "Radio",
    "Has mobile telephone": "Mobile telephone",
    "Has television": "Television",
    "Has refrigerator": "Refrigerator",
    "Has a computer": "Computer"
}

# Write the data for useful columns only in .csv format
outfile = data_folder+'household_data.csv'
data = data[cols_required].rename(columns=old_to_new_names)


# Write the data for useful columns only in .csv format
outfile = data_folder+'household_data.csv'
# data[cols_required].to_csv(outfile,index=False)
data.to_csv(outfile,index=False)
print('Written to ', outfile)
# N.B. The data will still need to be translated from numeric values to their meaning
# using the information in the .DO file

