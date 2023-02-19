import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import json
import re

config = json.load(open("config.json", "r"))

fig_path    = "./figures"
corr_path   = "./correlations"
plt.style.use('ggplot') # set matplotlib styling

"""
This is a file for general utility, where some functions are stored that are used multiple times.
This is to keep the main.py file clean and easy to follow and improve reusability of the code.
"""
# multi-column label encoder 
class MCLE:

    def __init__(self, columns=None):
        self.columns = columns # defines the columns which should be encoded or decoded
        
    def fit(self, df, y=None):
        try:
            columns = df.columns if self.columns is None else self.columns # if no columns are specified, all columns are encoded
            self.encoders = {col:LabelEncoder().fit(df[col]) for col in columns} # creates a dictionary of all the label encoders
            return self
        except Exception as e:
            logging.error(e)
            return None

    def transform(self, df):
        try:
            output = df.copy()
            columns = df.columns if self.columns is None else self.columns
            for col in columns:
                output[col] = self.encoders[col].transform(df[col])
            return output
        except Exception as e:
            logging.error(e)
            return None

    def fit_transform(self, df, y=None):
        return self.fit(df,y).transform(df)

    def inverse_transform(self, df):
        try:
            output = df.copy()
            columns = df.columns if self.columns is None else self.columns
            for col in columns:
                output[col] = self.encoders[col].inverse_transform(df[col])
            return output
        except Exception as e:
            logging.error(e)
            return None

    # add function for inverse transform with a single value based on column
    def inverse_transform_single(self, column, value):
        try:
            return self.encoders[column].transform([value])[0]
        except Exception as e:
            logging.error(e)
            return None

def _corr_string(dir):
    string = ""
    for i, value in enumerate(dir):
        string += f"{value:<15} {round(dir[value], 5)}"

        if i != len(dir) - 1:
            string += "\n"
    
    return string

def _correlations(df, column, threshold=0):    
    correlations = df.corr()[column]
    logging.debug(f"Correlations before drop: {correlations}")
    correlations.drop(column, inplace=True)
    logging.debug(f"Correlations after drop: {correlations}")
    corr_dict = dict()
    for i, corr in enumerate(correlations):
        logging.log(level=0, msg=f"{correlations.index[i]:<15}: {round(corr, 5)}")
        if threshold is not None:
            if abs(corr) >= threshold:
                corr_dict[correlations.index[i]] = corr

    return corr_dict

def get_corr(df=None, column=None, split_genders=False, file=None, features=[]):
    if not isinstance(df, pd.DataFrame):
        logging.warning("df is not a DataFrame")
        return None
    
    if column is None or column not in df.columns:
        logging.warning(f"Column {column} is None or not in DataFrame.")
        return None
    
    if features is str:
        features = [features]
    
    try:
        f = open(f"{corr_path}/{file}.txt", "r")
        groups = f.read().split("\n\n")
        f.close()
    except Exception:
        logging.error(e)
        logging.warning(f"Create correlation file '{file}.txt' first using .corr()")

    try:
        female_df = df[df["gender"] == 0] if split_genders == True else None
        male_df = df[df["gender"] == 1] if split_genders == True else None
    except Exception as e:
        split_genders = False
        logging.error(e)
        logging.error("Couldn't split genders")

    for feature in features:
        if feature not in df.columns:
            logging.warning(f"Feature {feature} not in df")
            continue
        if feature == column:
            logging.debug(f"Skipped correlation for double {feature}")
            continue
        try:
            corrs = {
                "general":  _correlations(df[[column, feature]], column)[feature],
                "female":   _correlations(female_df[[column, feature]], column)[feature] if split_genders is True else None,
                "male":     _correlations(male_df[[column, feature]], column)[feature] if split_genders is True else None}
        except Exception as e:
            logging.error(e)
            return None
        
        for i, group in enumerate(groups):
            listed_features = re.findall(r"\n(w+)", group)
            if feature not in listed_features:
                value = corrs["general"] if i == 0 else corrs["female"] if i == 1 else corrs["male"]
                if value is not None:
                    groups[i] += f"\n{feature:<15} {round(value, 5)}"
    
    with open(f"{corr_path}/{file}.txt", "w") as file:
        file.write("\n\n".join(groups))
    return None

def corr(df=None, column=None, threshold=0.25, split_genders=False, file=None):
    
    # checking that all given parameters are valid
    if not isinstance(df, pd.DataFrame):
        logging.warning("df is not a DataFrame")
        return None

    if column is None or column not in df.columns:
        logging.warning(f"Column {column} is None or not in DataFrame.")
        return None
    
    if threshold is not None and (type(threshold) is not float or threshold < 0.01 or threshold > 0.99):
        logging.warning(f"Threshold {threshold} is not a float between 0.01 and 0.99")
        return None

    try:
        female_df = df[df["gender"] == 0] if split_genders == True else None
        male_df = df[df["gender"] == 1] if split_genders == True else None
    except Exception as e:
        split_genders = False
        logging.error(e)
        logging.error("Couldn't split genders")

    general_corr = _correlations(df, column, threshold)
    female_corr = _correlations(female_df, column, threshold) if split_genders is True else None
    male_corr = _correlations(male_df, column, threshold) if split_genders is True else None
    
    if file is not None:
        file = open(f"{corr_path}/{file}.txt", "w")
        file.write(f"General Correlations: \n{_corr_string(general_corr)}\n\n")
        if male_corr and female_corr:
            file.write(f"Female Correlations: \n{_corr_string(female_corr)}\n\n")
            file.write(f"Male Correlations: \n{_corr_string(male_corr)}")
        file.close()

    logging.debug(f"General Correlations: {general_corr}")
    if male_corr and female_corr:
        logging.debug(f"Male Group Correlations: {male_corr}")
        logging.debug(f"Female Group Correlations: {female_corr}")
        return general_corr, female_corr, male_corr
    else:
        return general_corr

def boxplot(series=None, name=None):

    series.dropna(inplace=True) # dropping all NaN values from series to avoid errors with boxplot

    # checking that all given parameters are valid
    if series is None or not isinstance(series, pd.Series):
        logging.warning("Series is None or not a Series")
        return None

    if name is None or type(name) is not str:
        logging.warning("Name is None or not a string")
        return None

    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.boxplot(series)

    # add formatting
    plt.title(f"Boxplot of {name}")
    # set y axis range
    plt.ylim(series.min() - ((series.max()-series.min()) * 0.1), series.max() + ((series.max()-series.min()) * 0.1))
    
    # add yticks in 10% steps from min to max
    plt.yticks(np.arange(series.min(), series.max() + ((series.max()-series.min()) * 0.1), (series.max()-series.min()) * 0.1))

    plt.ylabel(f"{name}")
    plt.xticks([])
    plt.text(1.05, series.median(), f"Median: {round(series.median(), 2)}")
    plt.text(1.05, series.min(), f"Min: {round(series.min(), 2)}")
    plt.text(1.05, series.max(), f"Max: {round(series.max(), 2)}")

    # add text box with outlines below the boxplot
    plt.text(x=1, y=series.min()-(series.max()-series.min())*0.2, s=f"{'Variance:'} {round(series.var(), 2)}\n{'ST Deviation:'} {round(series.std(), 2)}", fontsize=12, horizontalalignment='center', multialignment="left").set_bbox({"edgecolor": "black", "facecolor": "white", "pad": 10})


    plt.savefig(f"{fig_path}/{name}_boxplot.png")

    logging.info(f"Created boxplot of {name}")

def histogram(series=None, name=None, bins=20):
    
    # checking that all given parameters are valid
    if series is None or not isinstance(series, pd.Series):
        logging.warning("Series is None or not a Series")
        return None

    if name is None or type(name) is not str:
        logging.warning("Name is None or not a string")
        return None

    if bins is None or type(bins) is not int:
        logging.warning("Bins is None or not an integer")
        return None

    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.hist(series, bins=bins, color=config["colors"]["sky_blue"])

    # add formatting
    plt.title(f"Histogram of {name}")
    plt.ylabel("Frequency")
    plt.xlabel(f"{name}")

    plt.savefig(f"{fig_path}/{name}_histogram.png")

    logging.info(f"Created histogram of {name}")

def pie_chart(series=None, name=None):
    if series is None or not isinstance(series, pd.Series):
        logging.warning("Series is None or not a Series")
        return None

    if name is None or type(name) is not str:
        logging.warning("Name is None or not a string")
        return None

    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.pie(series.value_counts(), autopct="%1.1f%%", textprops={"fontsize": 14, "color": "white", "weight": "bold"}, explode=[0.03 for i in range(len(series.value_counts()))])
    # add formatting
    plt.title(f"Pie Chart of {name}")
    plt.legend(series.value_counts().index, loc="best")
    # add percentage labels
    plt.gca().set_aspect("equal")

    plt.savefig(f"{fig_path}/{name}_pie_chart.png")

    logging.info(f"Created pie chart of {name}")

def percentiles(series=None, percentiles=[0.25, 0.5, 0.75]):
    if series is None or not isinstance(series, pd.Series):
        logging.warning("Series is None or not a Series")
        return None

    if percentiles is None or not isinstance(percentiles, list):
        logging.warning("percentiles is None or not a list")
        return None

    percentie_dict = {}
    for percentile in percentiles:
        percentie_dict[percentile] = series.quantile(percentile)

    return percentie_dict