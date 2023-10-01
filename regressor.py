# external imports
from joblib import dump
from warnings import filterwarnings
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error

import pandas as pd
import numpy as np
# internal imports
from data_preprocessing import DataPP
from MyANN import MyANN

class RegressorBuilder:
    def __init__(self,df:DataPP):
        self.df = df
        self.recommended_model = None # place-holder for the recommended model after training and review
        filterwarnings('ignore') # setting to stop warnings for when GridSearchCV is being fitted
        # models dict is built {"model_name": model_stats and is filled in as 
        # the models are built and evaluated
        self.models = dict()
        
    # Methods calling on the model constructors and gathering stats and scores
    def build_all(self, file_path):
        self.build_lir()
        self.build_lasso()
        self.build_ridge()
        self.build_elnet()
        self.build_svr()
        self.build_ann(file_path)
    def build_lir(self):
        # Linear Regression
        print("Creating a LINEAR REGRESSION model...")
        self.lir_model = self._poly_lir() # creating the model
        lir_stats = self.reg_evaluation(self.lir_model) # saving the scores
        self.models["Linear Regression"] = lir_stats
        print("Linear Regression ready ", "\u2713")
    def build_lasso(self):
        # Lasso
        print("Creating a LASSO model...")
        self.lasso_model = self._lasso_model() # creating the model
        lasso_stats = self.reg_evaluation(self.lasso_model) # saving the scores
        self.models["Lasso"] = lasso_stats
        print("LASSO ready ", "\u2713")
    def build_ridge(self):
        # Ridge
        print("Creating a RIDGE model...")
        self.ridge_model = self._ridge_model() # creating the model
        ridge_stats = self.reg_evaluation(self.ridge_model) # saving the scores
        self.models["Ridge"] = ridge_stats
        print("Ridge ready ", "\u2713")
    def build_elnet(self):
        # Elastic Net
        print("Creating an ELASTIC NET model...")
        self.elnet_model = self._elnet_model() # creating the model
        elnet_stats = self.reg_evaluation(self.elnet_model) # saving the scores
        self.models["Elastic Net"] = elnet_stats
        print("Elastic Net ready ", "\u2713")
    def build_svr(self):
        # SVR
        print("Creating an SVR model...")
        self.svr_model = self._svr_model() # creating the model
        svr_stats = self.reg_evaluation(self.svr_model) # saving the scores
        self.models["SVR"] = svr_stats
        print("SVR ready ", "\u2713")
    def build_ann(self, file_path):
        # ANN Regressor
        print("Creating an ANN Regressor model...")
        self.ann_reg_model = self._ann_model(file_path) # instance of MyANN class
        ann_reg_stats = self.ann_reg_evaluation()
        self.models["ANN Regressor"] = ann_reg_stats
        print("ANN Regressor ready ", "\u2713")
    
    # Internal methods actually building the GridSearchCV and models
    def _poly_lir(self):
            """Method for using GridSearchCV for finding and creating
            the best Linear Regression with polynomial features.
            Also saves the polynomial degree for the dataset in
            self.poly_degree

            Returns:
                GridSearchCV model with a pipeline of PolynomialFeatures,
                StandardScaler and LinearRegression as estimator
            """
            pipe= make_pipeline(PolynomialFeatures(),\
                    StandardScaler(), LinearRegression())
            param_grid = {'polynomialfeatures__degree': np.arange(6)}
            poly_grid = GridSearchCV(pipe, param_grid, cv=10, \
                scoring='r2')
            poly_grid.fit(self.df.X_train,self.df.y_train)
            self.poly_degree = poly_grid.best_params_["polynomialfeatures__degree"]
            return poly_grid
    def _lasso_model(self):
        """Creating a GridSearchCV model with a pipeline of 
        PolynomialFeatures,StandardScaler and Lasso as estimator
        
        Different parameters if self.big_df is True/False

        Returns:
            GridSearchCV: with Lasso as base
        """
        if self.df.big_df:
            pipe = make_pipeline(PolynomialFeatures(degree = self.poly_degree),\
            StandardScaler(),Lasso(max_iter=10000))
            param_grid = {'lasso__alpha': np.linspace(0.001, 1, 8)}
        else:
            pipe = make_pipeline(PolynomialFeatures(degree = self.poly_degree),\
            StandardScaler(),Lasso(max_iter=1000000))
            param_grid = {'lasso__alpha': np.linspace(0.001, 1, 15)}
        lasso_grid = GridSearchCV(pipe, param_grid, cv=10, \
            scoring='r2')
        lasso_grid.fit(self.df.X_train,self.df.y_train)
        return lasso_grid
    def _ridge_model(self):
        """Creating a GridSearchCV model with a pipeline of 
        PolynomialFeatures,StandardScaler and Ridge as estimator
        
        Different parameters if self.big_df is True/False

        Returns:
            GridSearchCV: with Ridge as base
        """
        pipe = make_pipeline(PolynomialFeatures(degree = self.poly_degree),\
            StandardScaler(),Ridge(max_iter=1000000))
        param_grid = {'ridge__alpha': (0.1,0.5,1,5,10)}
        ridge_grid = GridSearchCV(pipe, param_grid, cv=10, \
            scoring='r2')
        ridge_grid.fit(self.df.X_train,self.df.y_train)
        return ridge_grid
    def _elnet_model(self):
        """Creating a GridSearchCV model with a pipeline of 
        PolynomialFeatures,StandardScaler and Elastic Net as estimator
        
        Different parameters if self.big_df is True/False

        Returns:
            GridSearchCV: with Elastic Net as base
        """
        if self.df.big_df:
            pipe = make_pipeline(PolynomialFeatures(degree = self.poly_degree),\
            StandardScaler(),ElasticNet(max_iter=10000))
            param_grid = {'elasticnet__alpha':[0.1,0.2,0.5,1,10,50,100,200],
                'elasticnet__l1_ratio' : [.1, .5, .7, .9, .95, .99, 1]}    
        else:
            pipe = make_pipeline(PolynomialFeatures(degree = self.poly_degree),\
                            StandardScaler(),ElasticNet(max_iter=1000000))
            param_grid = {'elasticnet__alpha':[0.1,0.2,0.5,1,10,50,100,200],
                'elasticnet__l1_ratio' : [.1, .5, .7, .9, .95, .99, 1]}
        elnet_grid = GridSearchCV(pipe, param_grid, cv=10, \
            scoring='r2')
        elnet_grid.fit(self.df.X_train,self.df.y_train)
        return elnet_grid
    def _svr_model(self):
        """Creating a GridSearchCV model with a pipeline of 
        PolynomialFeatures,StandardScaler and SVR as estimator
        
        Different parameters if self.big_df is True/False

        Returns:
            GridSearchCV: with SVR as base
        """
        if self.df.big_df:
            pipe = make_pipeline(PolynomialFeatures(degree = self.poly_degree),\
                StandardScaler(),SVR(max_iter=100000))
            param_grid = {'svr__kernel':["linear", "poly", "rbf"],
                'svr__C': [0.01,0.05,1,5,10,100,500]}
        else:
            pipe = make_pipeline(PolynomialFeatures(degree = self.poly_degree),\
                StandardScaler(),SVR(max_iter=100000))
            param_grid = {'svr__kernel':["linear", "poly", "rbf"],
                'svr__C': [0.001,0.05,0.01,1,5,10,100,500]}
        svr_grid = GridSearchCV(pipe, param_grid, cv=10, \
        scoring='r2')
        svr_grid.fit(self.df.X_train,self.df.y_train)
        return svr_grid
    def _ann_model(self, file_path):
        my_ann_model = MyANN(data_set=file_path,
                             target = self.df._y_name,
                             loss = "mse")
        return my_ann_model
    
    # Evaluation methods
    def reg_evaluation(self,model):
        """Method for collecting the scores for the model

        Args:
            model : ML model for which the data is gathered

        Returns:
            dict: dict with best_params,MAE,RMSE,r2 and mean fitting time for
                            the given model
        """
        best_params = model.best_params_
        y_pred = model.predict(self.df.X_test)
        MAE = mean_absolute_error(y_true= self.df.y_test,\
                                y_pred = y_pred)
        RMSE = np.sqrt(mean_squared_error(y_true= self.df.y_test,\
                                y_pred = y_pred))
        r2_score = model.score(self.df.X_test,self.df.y_test)
        mean_fit = model.cv_results_["mean_fit_time"].mean()
        return {"Best hyperparameters":best_params,
                "MAE":MAE,
                "RMSE":RMSE,
                "R2 Score": r2_score,
                "Mean fitting time":mean_fit}
    def ann_reg_evaluation(self):
        """Separate method for ANN evaluation, since it's 
        an object of MyANN class and not GridSearchCV"""
        best_params = self.ann_reg_model.model_loss()
        MAE = self.ann_reg_model.mae()
        RMSE = self.ann_reg_model.rmse()
        r2_score = self.ann_reg_model.ann_r2()
        mean_fit = None
        return {"Best hyperparameters":best_params,
                "MAE":MAE,
                "RMSE":RMSE,
                "R2 Score": r2_score,
                "Mean fitting time":mean_fit}
    
    def reg_display_reports(self):
        """Method for displaying the reports for each regressor
        """
        for model, stats in self.models.items():
            print(f"\nModel name: {model}")
            print("-"*45)
            for name, stat in stats.items():
                if name == "Best hyperparameters":
                    print(name)
                    for param, value in stat.items():
                        print(param, ":", value)
                    print()
                else:
                    print(name, ":", stat)
            print("-"*45)
            input("Press [ENTER] to continue.")
    
    def print_summary(self):
        """Method for displaying the summary of the regressor models
        and choosing the recommended model
        """
        print("S U M M A R Y\n"+("-"*100))
        # creating a dataframe with a summary of all the scores
        df_stats = pd.DataFrame(data = self.models).transpose()
        df_stats.drop("Best hyperparameters", axis=1, inplace=True)
        df_stats = df_stats.astype("float")
        df_stats.loc["Best scores"] = [df_stats["MAE"].idxmin(), \
                                    df_stats["RMSE"].idxmin(),\
                                    df_stats["R2 Score"].idxmax(),\
                                    df_stats["Mean fitting time"].idxmin()]
        self.final_summary_df = df_stats
        print(df_stats)
        self.set_recommended_model()
    
    def set_recommended_model(self):
        # displaying the recommendation based on the most common
        # item in "Best scores"
        winner = self.final_summary_df.loc["Best scores"].mode().item()
        print(f"\n\n\tThe recommended model is >>> {winner} <<<")
        print("It has the best performance when it comes to:")
        for item in self.final_summary_df.columns\
                    [self.final_summary_df.isin([winner]).any()]:
            print(item)
        print("-"*100)

        match winner:
            case "Linear Regression":
                self.recommended_model = self.lir_model
            case "Lasso":
                self.recommended_model = self.lasso_model
            case "Ridge":
                self.recommended_model = self.ridge_model
            case "Elastic Net":
                self.recommended_model = self.elnet_model
            case "SVR":
                self.recommended_model = self.svr_model
            case "ANN Regressor":
                self.recommended_model = self.ann_reg_model
            case other:
                print("No such model found.")         
    
    def choose_model(self):
        """Method for choosing the model to save.
        Gives a choice between saving the recommended model or choosing
        manually.
        """
        action = self.user_choice(\
            {"1": "I agree with the recommendation.",
            "2": "I want to choose a model manually."})
        if action == "1":
            self.save_model(self.recommended_model)
        else:
            choice = self.user_choice(\
                {"1": "Linear Regression",
                    "2": "Lasso",
                    "3": "Ridge",
                    "4": "Elastic Net",
                    "5": "SVR",
                    "6": "ANN Regressor"})
            if choice == "1":
                model = self.lir_model
            elif choice == "2":
                model = self.lasso_model
            elif choice == "3":
                model = self.ridge_model
            elif choice == "4":
                model = self.elnet_model
            elif choice == "5":
                model = self.svr_model
            else:
                model = self.ann_reg_model
            self.save_model(model)
            
    def save_model(self,model):
        """Method for saving the chosen model using the dump function from
        joblib library. The user chooses the name for the model that can
        contain only 10 alphanumeric characters.

        Args:
            model : trained model to be saved
        """

        while True:
            name = input("Choose a name for your model (max 10 characters, only letters and numbers):\n")
            if len(name) <= 10 and name.isalnum():
                break
            else:
                print("Invalid name. Try again.")
        
        if model == self.ann_reg_model:
            self.ann_reg_model.save_model(name)
        else:
            name = name + ".joblib"
            dump(model,name)

        print(f"Your model was saved as {name}.\n")
        print("Thank you for using the ML app!")
        print("The programme is now terminated.")
    
    def user_choice(self,choices:dict):
        """Static method for whenever a user has to make
        a choice. 

        Args:
            choices (dict): A list of choices and descriptions
            to be displayed and chosen from
            {choice: description}

        Returns:
            str: user choice
        """
        while True:
            print("Press: ")
            for key in choices:
                print(f"{key} = {choices[key]}")
            u_choice = input("")
            if u_choice not in choices.keys():
                print("Invalid choice. Press any key to try again.")
            else:
                return u_choice