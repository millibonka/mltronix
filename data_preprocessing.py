import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


class DataPP():
    """Data preprocessing class"""
    model_types = ["regressor", "classifier", "none", None]
    def __init__(self, valid_path:str, model_type:str):
        try:
            self.df = pd.read_csv(valid_path)
        except UnicodeDecodeError: 
            print("Invalid file format. Choose a CSV file.")
        except pd.errors.EmptyDataError:
            print("The chosen CSV file is empty. Choose another file.")
        except FileNotFoundError:
            print("The selected file does not exist.")
        else:
            self.model_type = model_type
        
    @property
    def model_type(self):
        return self._model_type
    @model_type.setter
    def model_type(self, value):
        if value in self.model_types:
            self._model_type = value
        else:
            raise ValueError("Invalid model type.")
    
    def display_info(self):
        """Displays the statistics of the dataset
        """
        print("Here are the statistics of your dataframe:\n")
        print("-"*100)
        print("First five rows:\n")
        print(self.df.head())
        print("-"*100)
        print("Detailed information:")
        print(f"Number of rows: {self.df.shape[0]}\n"+ \
              f"Number of features: {self.df.shape[1]}\n"+ \
              f"Number of missing values: {self.df.isna().sum().sum()}\n")
    
    def choose_y(self):
        while True:
            print("Which feature is the y target?")
            y_target = input(f"{list(self.df.columns)}\n")
            if y_target in self.df.columns:
                self._y = self.df[y_target]
                self._y_name = y_target
                break
            else:
                print("Invalid choice. Try again.")

    def validate_model_type(self):
        # checking if data is all numerical
        try:
            pd.to_numeric(self._y, errors="raise")
            is_numerical = True
            print("Your target column is numerical.")
        except ValueError:
            is_numerical = False
            print("Your target column is non-numerical.")
        # setting the number of unique values in y
        n_unique = self._y.nunique()
        print(f"Your target column contains {n_unique} unique values.")
        
        if is_numerical and n_unique > 10:
            self.model_type = "regressor"
        else:
            self.model_type = "classifier"
        
        print(f"Based on your target column, the model type has been set to {self.model_type}.")
            
    def ready_for_ML(self) -> bool:
        # check for missing values
        if self.df.isna().sum().sum() > 0:
            print("="*50)
            print(self.missing_values_report())
            print("="*50)
            raise ValueError("Your dataframe has missing values. See the report above.")
        # check for too many categories for a classifier (max 10)
        elif self._y.nunique() >= 10 and self.model_type == "classifier":
            raise ValueError("Your dataframe has too many categories. Max 10.")
        return True
        
    def missing_values_report(self):
        df_features = []
        df_missing_count = []
        df_missing_rows = []
        df_missing_procent = []

        for feature in self.df.columns:
            missing_count = 0
            if self.df[feature].isna().sum() > 0:
                df_features.append(feature)
                missing_count = self.df[feature].isna().sum()
                df_missing_count.append(missing_count)
                missing_row = list(self.df[self.df[feature].isna()].index)
                df_missing_rows.append(missing_row)
                missing_procent = round((self.df[feature].isna().sum() / len(self.df[feature])) * 100, 2)
                df_missing_procent.append(missing_procent)

        results = pd.DataFrame({"Feature":df_features,
                        "Missing values count": df_missing_count,
                        "Missing rows": df_missing_rows,
                        "Missing data %": df_missing_procent})
        return results
    
    def set_X(self):
        X = self.df.drop(self._y_name, axis = 1)
        try:
            X.apply(pd.to_numeric, errors="raise")
        except ValueError:
            print("Your X contains non-numerical data. Those values will be converted to dummies.")
            self._X = pd.get_dummies(X, drop_first=True)
        else:
            self._X = X
        
    def tt_split(self):
        x = 0
        if len(self._y) > 1000:
            x = 0.1
        else:
            x = 0.3
        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_test_split(self._X, self._y, test_size=x, random_state=101)

    def df_size_setter(self):
        if self._X.shape[0] > 1000 or self._X.shape[1] > 10:
            print("Your dataframe is over 1000 rows.\n"+\
                "It may take several minutes to create the models.\n"+\
                "Thank you for your patience!\n")
            self.big_df = True
        else:
            self.big_df = False
    
    def check_class_balance(self):
        """Method for checking if the target data is balanced. 
        It is set as balanced if each value count is within 10%
        of the overall mean of value counts.

        Returns:
            True/False or None if the model type is not "classifier"
        """
        if self.model_type == "classifier":
            y = self._y.value_counts()
            mean_min = y.mean() - (y.mean()*0.1)
            mean_max = y.mean() + (y.mean()*0.1)
            self.class_balanced = all(np.where(y.between(mean_min,mean_max),\
                                                        True,False))
        else:
            self.class_balanced = None