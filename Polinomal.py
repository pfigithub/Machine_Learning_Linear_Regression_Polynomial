# import needed packages
import pandas as pd
import numpy as np

# getting and reading data
df = pd.read_csv("yourdata.csv")

# take a look at the dataset
df.head()

# selecting some feautures(our data it is FuelConsumption)
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
