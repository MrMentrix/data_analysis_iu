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


## Choosing Research Questions

Now that the dataset has been chosen and studied, the next step is to choose research questions that can be answered with the data. The following research questions were chosen:

1. Which features correlate with the offered salary?
2. Which features correlate with the placement status?
3. Is there a differnece in placement percentage between genders?
4. Is there a difference in offered salary between genders?
5. IS there a difference in educational background between genders?

All data will be enumerate before any analysis is performed. This is done to ensure that statistical tests can be performed on the data. After all statistical tests have been performed, the enumerized data will be turned back into its original form. This is done with the `multicolumnlabelencoder.py` file and the help of `sklearn.preprocessing.LabelEncoder`.

## Which features correlate with the offered salary?

To answer this question, first, the correation between all variables and `salary` will be calculated. The following correlation levels will be used:
+1.0 or -1.0: Perfect correlation
+0.7 or -0.7: Very high correlation
+0.5 or -0.5: High correlation (significant)
+0.3 or -0.3: Moderate correlation (possible significance)
+0.1 or -0.1: Low correlation (not significant)

**Results:**
When testing the general group and the male and female subset, there are no correlations stronger than 0.25 in both the general and male group. Within the female group, there are three correlations stronger than 0.25:
- `hsc_p`:    0.3988875323722999
- `degree_p`: 0.2592300585156336 
- `mba_p`:    0.2700372257379307
