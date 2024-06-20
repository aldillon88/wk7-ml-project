# IMDB Machine Learning Project
### Challenge:
This project aims to build a machine learning model that can be used to accurately predict the `No_of_votes` of the movies in the dataset.

### Data Source:
The data for this project can be found at [kaggle.com](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows/data).

### Data Description:
The data is collated into one dataset with 1000 rows and 16 columns. The columns are as follows:
- `Poster_Link` - Link of the poster that imdb using
- `Series_Title` = Name of the movie
- `Released_Year` - Year at which that movie released
- `Certificate` - Certificate earned by that movie
- `Runtime` - Total runtime of the movie
- `Genre` - Genre of the movie
- `IMDB_Rating` - Rating of the movie at IMDB site
- `Overview` - mini story/ summary
- `Meta_score` - Score earned by the movie
- `Director` - Name of the Director
- `Star1` - Name of a movie star
- `Star2` - Name of a movie star
- `Star3` - Name of a movie star
- `Star4` - Name of a movie star
- `No_of_votes` - Total number of votes
- `Gross` - Money earned by that movie

### Libraries Required:
#### General:
`pandas` `numpy` 
#### Machine Learning:
`sklearn`
#### Visualization:
`matplotlib`
`seaborn`

### Process:
**Step 1:** Load the data into `main_notebook.ipynb` using `pd.read_csv` and and do an initial exploration of the dataset using functions such as `df.shape`, `df.isna().sum()`, `df.dtypes` etc..\
**Step 2:** Using what was learned from step 1, begin cleaning the data and correcting data types.\
**Step 3:** Grouped categorical columns with its unique values to broader categories for `Genre` `Certificate` `Director` `Star1` `Star2` `Star3` and `Star4` columns.\
**Step 4:** The graphs were plotted to visually analyze and interpret the distribution and trends of various columns in the dataset, enabling better understanding and insights into the data.\
**Step 5:** Drop columns which had low correlation.\
**Step 6:** Begin experimenting with various machine learning methods from the `sklearn` library after defining `target` and `features` where `target` is the `no_of_votes` column\
**Step 7:** Now we perform the division between Train and Test, by reserving 20% of our data to Test. We used `train_test_split` for this.\
**Step 8:** Use One hot encoding (OHE) machine learning technique to encode categorical data to numerical ones.\
**Step 9:** Now we Transformed features by scaling each feature to a given range using `MinMaxScaler`\
**Step 10:** Used `KNeighborsRegressor` model with k=13 on normalized training data, to evaluate its performance on test data using R-squared, and record the model's error metrics.\
**Step 11:** Trained `LinearRegression` model on normalized training data, to evaluate its performance on test data using Mean Absolute Error (MAE), RMSE, and R-squared, and record the model's error metrics.\
**Step 12:** Trained `BaggingRegressor` with `DecisionTreeRegressor` base estimator (max depth=20), 50 estimators, and variable max_samples, evaluated the performance on normalized test data using MAE, RMSE, and R-squared, and record the model's error metrics.\
**Step 13:** Performed grid search using `BaggingRegressor` with `DecisionTreeRegressor` base estimator to optimize hyperparameters, evaluated the performance on normalized test data using MAE, RMSE, and R-squared, and record the model's error metrics.\
**Step 14:** Trained `RandomForestRegressor` with 100 estimators and max depth of 20 on normalized training data, evaluated the performance on test data using MAE, RMSE, and R-squared, and records the model's error metrics.\
**Step 15:** Performed grid search using `RandomForestRegressor` to optimize hyperparameters, evaluates its performance on normalized test data using MAE, RMSE, and R-squared, and record the model's error metrics.\
**Step 16:** Trained `AdaBoostRegressor` with `DecisionTreeRegressor` base estimator (max depth=20) and 100 estimators, evaluated the performance on normalized test data using MAE, RMSE, and R-squared, and record the model's error metrics.\
**Step 17:** Performed grid search using `AdaBoostRegressor` with `DecisionTreeRegressor` base estimator to optimize hyperparameters, evaluated the performance on normalized test data using MAE, RMSE, and R-squared, and record the model's error metrics.\
**Step 18:** Trained `GradientBoostingRegressor` with max depth of 20 and 100 estimators on normalized training data, evaluated the performance on test data using MAE, RMSE, and R-squared, and record the model's error metrics.\
**Step 19:** Performed grid search using `GradientBoostingRegressor` to optimize hyperparameters, evaluated the performance on normalized test data using MAE, RMSE, and R-squared, and record the model's error metrics.

### After testing every model we came to a conclusion that `GradientBoostingRegressor` model gives the highest R-squared value of 0.83 compared to all other models.


|    | model                      | optimized | mean_absolute_error | mean_squared_error | r2_score |
|---:|:---------------------------|:----------|---------------------|--------------------|----------|
|  0 | GradientBoostingRegressor | True      | 110006.69           | 154786.22          | 0.83     |
|  1 | BaggingRegressor          | True      | 112719.13           | 163279.32          | 0.81     |
|  2 | RandomForestRegressor     | False     | 114305.02           | 163066.16          | 0.81     |
|  3 | AdaBoostRegressor         | True      | 109582.78           | 160076.89          | 0.81     |
|  4 | BaggingRegressor          | False     | 115885.33           | 167136.39          | 0.80     |
|  5 | AdaBoostRegressor         | False     | 108685.24           | 166017.95          | 0.80     |
|  6 | RandomForestRegressor     | True      | 117724.93           | 169832.77          | 0.79     |
|  7 | LinearRegression          | False     | 156004.59           | 204939.43          | 0.70     |
|  8 | GradientBoostingRegressor | False     | 152436.75           | 227171.59          | 0.63     |
|  9 | KNeighborsRegressor       | False     | 184238.79           | 266084.68          | 0.49     |

This table summarizes the model performances, indicating the optimized status, mean absolute error, mean squared error, and R-squared score for each model.

### Notebooks:

* [Main Cleaning](main_cleaning.ipynb) has the code related to cleaning and grouping
* [Initial ML](initial-ML.ipynb) has the code related to all the above mentioned models
* [Main Notebook](main_notebook.ipynb) has the code related to just the highest score model i.e., `GradientBoostingRegressor`