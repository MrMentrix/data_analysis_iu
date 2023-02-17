import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as t
import logging
import stats
from sklearn.preprocessing import LabelEncoder
import utility as util

# set logging config
logging.basicConfig(format='%(levelname)s %(filename)s:%(lineno)s %(message)s',
                    level=logging.DEBUG)
plt.style.use('ggplot')                             # set matplotlib styling
df = pd.read_csv("Placement_Data_Full_Class.csv")   # reading the data

cat_cols = df.select_dtypes(include=['object']).columns # get categorical columns
mcle = util.MCLE(columns=cat_cols)                      # set up the label encoder
df = mcle.fit_transform(df)                             # encode the categorical data into numeric data

# 1. Which features correlate with the offered salary?
general_corr, female_corr, male_corr = util.corr(df=df, column="salary", threshold=0.25, split_gender=True)

# 2. Which features correlate with the placement status?
general_corr, female_corr, male_corr = util.corr(df=df, column="status", threshold=0.25, split_gender=True)

# 3. Is there a differnece in placement percentage between genders?