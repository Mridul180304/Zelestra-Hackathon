# Zelestra Hackathon: Blend Property Prediction

This project focuses on predicting multiple blend properties (e.g., `BlendProperty1` to `BlendProperty10`) for the **Zelestra Hackathon**. It involves comprehensive data preprocessing, feature engineering, and a robust multi-output regression model to provide accurate predictions on unseen test data.

## 📖 Table of Contents

* [Project Overview](#project-overview)

* [Data Description](#data-description)

* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)

* [Data Preprocessing](#data-pre-processing)

* [Feature Engineering](#feature-engineering)

* [Model Building](#model-building)

* [Prediction and Submission](#prediction-and-submission)

* [Tech Stack](#tech-stack)

* [Getting Started](#getting-started)

* [Contact](#contact)

## 🚀 Project Overview

The primary objective of this hackathon project is to build a machine learning model capable of accurately predicting a set of `BlendProperty` values based on a given dataset. This involves handling various data types, addressing missing values, transforming features, and employing an ensemble modeling approach to achieve high predictive performance. The solution utilizes a multi-output regression strategy, training individual models for each target blend property.

## 📊 Data Description

The dataset consists of two main files:

* `train.csv`: Contains the training features and the target `BlendProperty` columns.

* `test.csv`: Contains the test features. Notably, this file does **not** include the target `BlendProperty` columns, as these are what our model aims to predict.

*Example of `train.csv` head.*

*Example of `test.csv` head.*

## 🔍 Exploratory Data Analysis (EDA)

Initial data exploration was performed to understand the structure, data types, and presence of missing values in both the training and test datasets.

* `train_df.info()` and `test_df.info()` provided summaries of column data types and non-null counts.

* `train_df.isnull().sum()` helped identify the extent of missing values in each column of the training set.

## 🧹 Data Preprocessing

Before modeling, the data underwent several preprocessing steps:

* **Type Conversion**: Columns such as `wind_speed`, `pressure`, and `humidity` were converted to numeric types, with `errors='coerce'` to handle any non-numeric entries by converting them to `NaN`.

* **Label Encoding**: Categorical string columns like `string_id`, `error_code`, and `installation_type` were transformed into numerical representations using `LabelEncoder`. This is crucial for machine learning models that require numerical inputs.

## ⚙️ Feature Engineering

Feature engineering focused on preparing the data for the regression models:

* **Correlation Analysis**: A correlation heatmap was generated to visualize relationships between all features and the `efficiency` column. While this analysis was performed, the final model utilizes a comprehensive set of features, and the `FeatureBlender` class handles advanced transformations.
  *Correlation Heatmap of the dataset.*

* **Custom Feature Blending (`FeatureBlender`)**: A custom `FeatureBlender` transformer was implemented as part of the model pipeline. This class:

  * Applies `StandardScaler` to normalize features.

  * Applies `QuantileTransformer` (with `output_distribution='normal'`) to transform features into a normal distribution, making them more suitable for certain models.

  * Concatenates the original features with their scaled and quantile-transformed versions, creating a richer feature set for the subsequent models.

## 🧠 Model Building

The core of the solution is a multi-output regression strategy where a separate model is trained for each of the 10 `BlendProperty` targets.

* **Target Variables**: The targets are `BlendProperty1` through `BlendProperty10`.

* **Model Architecture**: A `Pipeline` is used, integrating the `FeatureBlender` with a `StackingRegressor`.

  * **`StackingRegressor`**: This ensemble method combines predictions from multiple diverse models.

    * **Base Estimators**:

      * `LGBMRegressor`: A gradient boosting framework known for its speed and efficiency. Configured with `n_estimators=50`, `learning_rate=0.1`, `max_depth=4`, etc.

      * `HistGradientBoostingRegressor`: A fast gradient boosting model for large datasets. Configured with `max_iter=125`, `max_depth=5`, etc.

    * **Final Estimator**: `RidgeCV(cv=3)` is used to combine the predictions of the base estimators, providing a robust final output.

  * **Training Loop**: The code explicitly trains a separate `hybrid_model` pipeline for each `BlendProperty` target, ensuring that each prediction is optimized for its specific output.

## 📈 Prediction and Submission

After training, the models are used to generate predictions on the `test_df`.

* The `ID` column from `test_df` is preserved for mapping predictions to the correct entries.

* Predictions for each `BlendProperty` are collected.

* The final predictions are compiled into a CSV file (e.g., `Shell_test_predictions.csv`) in the required submission format.

## 💻 Tech Stack

* **Data Manipulation**: `pandas`, `numpy`

* **Visualization**: `matplotlib.pyplot`, `seaborn`

* **Machine Learning**: `scikit-learn` (e.g., `KNNImputer`, `RandomForestRegressor`, `StandardScaler`, `LabelEncoder`, `Pipeline`, `GridSearchCV`, `StackingRegressor`, `HistGradientBoostingRegressor`, `RidgeCV`, `QuantileTransformer`, `train_test_split`), `lightgbm` (`LGBMRegressor`)

* **Environment**: Google Colab (for `google.colab.files` for file uploads)

## 🚀 Getting Started

To replicate this project locally or run it in a similar environment:

### Clone the repository


git clone https://github.com/Mridul180304/StillPoint.git
cd StillPoint # Assuming you create a new directory for this project


*Note: This specific project might be located in a subdirectory or a separate repository. Adjust the clone command accordingly if the project is hosted elsewhere.*

### Install dependencies

Ensure you have Python installed, then install the required libraries:


pip install pandas matplotlib seaborn scikit-learn lightgbm numpy


### Run the project

Execute the Python script (e.g., a Jupyter Notebook or a `.py` file containing the provided code):


python your_script_name.py


*Remember to upload `train.csv` and `test.csv` if running in an environment like Google Colab using `files.upload()`.*

## 📧 Contact

Feel free to reach out if you have any questions, feedback, or collaboration opportunities!

* **Your Name**: [Mridul Krishan Chawla]



* **GitHub**: [https://github.com/Mridul180304](https://github.com/Mridul180304)

* **LinkedIn**: [Your LinkedIn Profile URL](https://www.linkedin.com/in/mridul-chawla-8234b9250/)

* **Email**: [your.email@example.com](mridulchawla20@gmal.com)
