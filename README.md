# IMDB Machine Learning Project
### Challenge:
This project aims to build a machine learning model that can be used to accurately predict the ***IMDB_Rating*** of the movies in the dataset.

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
`pandas`
#### Machine Learning:
`sklearn`
#### Visualization:
`matplotlib`
`seaborn`

### Process:
**Step 1:** Load the data into `main_notebook.ipynb` using `pd.read_csv` and and do an initial exploration of the dataset using functions such as `df.shape`, `df.isna().sum()`, `df.dtypes` etc..\
**Step 2:** Using what was learned from step 1, begin cleaning the data and correcting data types.\
**Step 3:** Begin experimenting with various machine learning methods from the `sklearn` library.\
