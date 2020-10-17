import numpy as np
import pandas as pd
import sys


# the lower the value, the greater the change of a positive reward.
def importingData():
    adsDF = pd.read_csv('data/Ads_Optimisation.csv')  # importing the data
    meansDF = adsDF.mean()  # averaging the click for each ad.
    newArr = np.array_split(meansDF, 2)  # working with five ads
    data = np.array([newArr[0], newArr[1]])
    data = np.negative([newArr[0], newArr[1]])  # converting them into negative number
    return data


def get_integer(num):
    try:
        number = int(num)
    except:
        print("An exception occurred: you entered a non-integer value.")
        sys.exit(1)
    else:
        return number
