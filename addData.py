import numpy as np
import pandas as pd
import sys


## Documentation for importingData function
# The function would read/import the ads optimisation data for the contextual bandit. The agent will be able to select ads.
# We split the data so we will have 5 ads.
# The lower the value, the greater the change of a positive reward since it is based on sampling a number greater than the number stored in the bandit from a normal distribution.
def importingData():
    adsDF = pd.read_csv('data/Ads_Optimisation.csv')  # importing the data
    meansDF = adsDF.mean()  # averaging the click for each ad.
    newArr = np.array_split(meansDF, 2)  # working with five ads
    data = np.array([newArr[0], newArr[1]])
    data = np.negative([newArr[0], newArr[1]])  # converting them into negative number
    return data

## Documentation for get_integer function
# The function would handle the exceptions if the value is not an integer.
def get_integer(num):
    try:
        number = int(num)
    except:
        print("An exception occurred: you entered a non-integer value.")
        sys.exit(1)
    else:
        return number
