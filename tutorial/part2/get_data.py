# Imports
import requests
import pandas as pd
import numpy as np
# Get the data from the api
r = requests.get("http://api.sealevelresearch.com/1/sea-level/tide/observations/liverpool-gladstone-dock/?start=2018-01-01T00:00:00Z&end=2018-01-09T00:00:00Z")
df = pd.DataFrame(r.json())
# Clean the dataset
df.fillna(0, inplace=True)
array = []
for i in range(0,len(df),10):
    array.append(np.mean(df.loc[i:i+10, 'tide_level']))
df = pd.DataFrame({'tide_level':array})
# Print the cleaned dataset
print(df.head())
# Export to csv
df.to_csv('data.csv')
