import pandas as pd
import numpy as np
import logging
import utility as utils
import json

config = json.load(open("config.json", "r"))

# set logging config
logging.basicConfig(format='%(asctime)s %(levelname)s %(funcName)s %(filename)s:%(lineno)s %(message)s',
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

if True:
    for feature in boxplots:
        percentile_dict = utils.percentiles(series=df[feature], percentiles=[0.95])
        logging.debug(f"{feature} percentiles: {percentile_dict}")

# remove salaries above 95th percentile, also keep missing values
df = df[(df["salary"] <= 423250) | (df["salary"].isna())]

if True:
    for feature in boxplots:
        utils.boxplot(df[feature], feature)
    for feature in histograms:
        utils.histogram(df[feature], feature, bins=config["hist_bins"])
    for feature in pie_charts:
        utils.pie_chart(mcle.inverse_transform(df)[feature], feature) # using the inverse transformation to get correct labels

if True:
# calculate correlation for salary feature
    salary_corr = utils.corr(df, "salary", config["corr_threshold"], split_genders=True, file="salary_corr")
    status_corr = utils.corr(df, "status", config["corr_threshold"], split_genders=True, file="status_corr")
    utils.get_corr(df, "salary", True, "salary_corr", ["degree_p", "etest_p", "mba_p", "ssc_p", "workex"])

    # get covariance of ["etest_p", "mba_p", "degree_p", "ssc_p"] and "salary"/"status" using pandas
    utils.cov(df, "salary", ["etest_p", "mba_p", "degree_p", "ssc_p"], True, "salary_cov")
    utils.cov(df, "status", ["etest_p", "mba_p", "degree_p", "ssc_p"], True, "status_cov")

    utils.scatter(df, ["salary", "etest_p"], "salary", True)

if True:
    for column in df.columns:
        utils.gender_plot(df, column, f"{column}")