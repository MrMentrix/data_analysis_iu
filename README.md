# Exploratory Data Analysis and Visualization
Welcome to the coding part of the written assignment for the course Exploratory Data Analysis course! Here, you will find some information and documentation over what steps where taken and how the coding part of this assignment was completed.

## Libraries to be used
There is a number of libraries that will be used for this project. To install all of them, use `pip install -r requirements.txt` or install them manually. To ensure that all packages are up to date, use `pip install --upgrade -r requirements.txt`.

Pandas and Numpy will be used to handle the data, perform mathematical actions on the data and to create dataframes. Matplotlib will be used to create visualizations.
Stats and scikit-learn will be used to perform statistical tests on the data and logging and tqdm will be used to log the progress of the code and support the debugging progress.

## Choosing a Dataset
The dataset "[Campus Recruitment](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement?resource=download&select=Placement_Data_Full_Class.csv)" from [Kaggle](https://www.kaggle.com/) was chosen for this project task. The downloaded data was stored in an XLS format, even though the data itself was in CSV format, so the extention had to be renamed from .xls to .csv in order to properly work with the data. The raw dataset has a shape of (215, 15), which should be sufficient for a visualization and exploratory analysis task. While too small datasets may not provide sufficient information for deeper analysis, a too large dataset may make visualization difficult. Therefore, a dataset with 215 datapoints seems suitable for this cause.

When testing for missing values with `df.isnull().sum()`, there are only missing values (67) in the salary feature. All other features don't have any missing values. This speaks for the quality of the dataset and will make analysis easier. The assumption is made that the data provided is correct and has already been cleaned and preprocessed.

After this, the dataset was manually studied to understand the features and their values. The dataset contains 215 datapoints, each with 15 features. The features are:
- sl_no           | Serial Number (index)
- gender          | Students Gender (Male = "M", Female = "F")
- ssc_p           | Secondary Education percentage (10th Grade)
- ssc_b           | Board of Education (Central/Others)
- hsc_p           | Higher Secondary Education percentage (12th Grade)
- hsc_b           | Board of Education (Central/Others)
- hsc_s           | Specialization in Higher Secondary Education
- degree_p        | Degree Percentage
- degree_t        | Under Graduation (Degree type)
- workex          | Work Experience (Yes/No)
- etest_p         | Employability test percentage (conducted by college)
- specialisation  | Post Graduation(MBA) Specialization
- mba_p           | MBA percentage
- status          | Status of placement (Placed/Not Placed)
- salary          | Salary offered by corporate to candidates

# Utility File
To ensure good readability, easy access to important functions, and better code reusability, the `utility.py` file was created. This contains the following functions and classes:

## Classes
### MCLE (Multi-Column Label Encoder)
This is an encoder based on the `LabelEncoder` from `sklearn.preprocessing`. It can be used to encode multiple columns at once, storing all encoders in `self.encoders`. This approach allows to easily reverse the encoding process by using the `inverse_transform` method on a column or a dataframe, or the `inverse_transform_single` method on a single value.

## Functions
### .corr()
This function stores all as significant considered correlations in a file. It also allows to calculate additional correlations for the male and female sub-group, which can be used to compare the correlations among the two groups.

### ._correlations()
This is a private function executed by `.corr()` which performs the actual correlation calculation and returns a dictionary of all correlations.

### ._corr_string()
This private function is executed by `.corr()` and creates a well formatted string of the correlations, to properly store the values in a file.

### .get_corr()
This function can be used to get a correlation and store it in the existing correlation file, regardless of the correlation being significant or not. There is no threshold for the correlation coefficient. This allows for comparison between, e.g., genders, even if only one of the genders has a significant correlation between two features.

### .gender_plot()
This creates a basic `matplotlib.pyplot` plot where values for the male and female group can be compared. The index are forced to be the same range for both groups to enable comparison within the same plot.

### .boxplot()
This creates a general boxplot for a specified column in a dataframe.

### .histogram()
This creates a general histogram for a specified column in a dataframe.

### .pie_chart()
This creates a general pie chart for a specified column in a dataframe.

### .scatter_plot()
This creates a general scatter plot for two specified columns in a dataframe. This also allows to highlight male and female data points by giving them individual colors.

### .percentiles()
This returns percentiles for a specified column in a dataframe. The percentiles can be chosen arbitrarily.

### .cov()
This returns the covariance for two specified features in a dataframe. It also allows to split the general dataframe into two sub-dataframes, based on gender.

### .mwu()
This performs the Mann-Whitney U test and returns the p and u value. This automatically splits the dataset into the male and female sub-set. The p-value is based on the difference between the male and female values.

## Getting a first impression of the data
To get a first impression of the data, it is recommended to get some visual impressions. This helps to better understand the data and get a feeling for how certain features are distributed. To do this, the following figuers were created. You can find them in the "figures" folder.

1. Boxplot and histogram of ssc_p
2. Boxplot and histogram of hsc_p
3. Boxplot and histogram of etest_p
4. Boxplot and histogram of degree_p
5. Boxplot and histogram of mba_p
6. Boxplot and histogram of offered salary
7. Pie Chart of hsc_s
8. Pie Chart of degree_t

Thanks to the earlier created functions in `utility.py`, these visualizations can easily be created.

## Further investigating the salary feature
The salary feature appeared to be interesting due to its high range of values. To get a better understanding of the salary feature, correlations were calculated. After comparing, covariances were calculated. Further visualizations were created to get a better understanding of the salary feature and the role the gender may play.

## Findings
The findings are, that while gender may influence the offered salary, it can't be said certainly since there may be other factors playing a role that are not provided by the dataset.