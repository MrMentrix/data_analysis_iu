import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as t
import logging
import stats
from sklearn.preprocessing import LabelEncoder
import utility as utils

# set logging config
logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s:%(lineno)s %(message)s',
                    datefmt="%H:%M:%S",
                    level=logging.INFO)
df = pd.read_csv("Placement_Data_Full_Class.csv")   # reading the data

cat_cols = df.select_dtypes(include=['object']).columns # get categorical columns
mcle = utils.MCLE(columns=cat_cols)                      # set up the label encoder
df = mcle.fit_transform(df)                             # encode the categorical data into numeric data

"""
Create the following figures:
1. Boxplot and histogram of ssc_p
2. Boxplot and histogram of hsc_p
3. Boxplot and histogram of etest_p
4. Boxplot and histogram of degree_p
5. Boxplot and histogram of mba_p
6. Boxplot and histogram of offered salary
7. Pie Chart of hsc_s
8. Pie Chart of degree_t
"""

boxplots    = ["ssc_p", "hsc_p", "etest_p", "degree_p", "mba_p", "salary"]
histograms  = ["ssc_p", "hsc_p", "etest_p", "degree_p", "mba_p", "salary"]
pie_charts  = ["hsc_s", "degree_t"]

for feature in boxplots:
    utils.boxplot(df[feature], feature)
for feature in histograms:
    utils.histogram(df[feature], feature, bins=20)
for feature in pie_charts:
    utils.pie_chart(mcle.inverse_transform(df)[feature], feature) # using the inverse transformation to get correct labels