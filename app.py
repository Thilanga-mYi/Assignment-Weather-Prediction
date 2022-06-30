# COMMENT: linear algebra
import numpy as np

# COMMENT: data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd

import matplotlib as plt
import matplotlib.pyplot as plt
import os

# COMMENT: Any results you write to the current directory are saved as output.
weather = pd.read_csv("weatherHistory.csv")
weather

# print(weather.head(10))

# print(weather.groupby('Summary').mean()["Apparent Temperature (C)"].plot(kind='bar'))

# print(weather[weather["Summary"] == "Dry"].mean())

def convert_summary(col):
    return len(col)

# COMMENT: Need to find the Apparent temperature when humidity given
weather_temp = weather[["Humidity","Apparent Temperature (C)"]]
#weather_temp["Summary"] = weather["Summary"].apply(convert_summary)
# print(weather_temp.head(12))

weather.groupby('Summary').mean()["Apparent Temperature (C)"].plot(kind='bar')