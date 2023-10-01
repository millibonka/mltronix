# importing external libraries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from joblib import dump
from warnings import filterwarnings
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,\
                        classification_report,confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier



from data_preprocessing import DataPP
from MyANN import MyANN

class ClassifierBuilder:
    def __init__(self,df:DataPP):
        self.df = df
        self.recommended_model = None # place-holder for the recommended model after training and review
        filterwarnings('ignore') # setting to stop warnings for when GridSearchCV is being fitted
        # models dict is built {"model_name": model_stats and is filled in as
        # the models are built and evaluated
        self.models = dict()
    # Methods calling on the model constructors and gathering stats and scores
    def build_all(self, file_path):
        self.build_lor()
        self.build_knn()
        self.build_svc()
        self.build_ann(file_path)
    def build_lor(self):
        # Logistic Regression model
        print("Creating a LOGISTIC REGRESSION model...")
        self.lor_model = self._lor_model() # creating the model
        lor_stats = self.class_evaluation(self.lor_model) # saving the scores
        self.models["Logistic Regression"] = lor_stats
        print("Logistic Regression ready ", "\u2713")
    def build_knn(self):    
        # KNN
        print("Creating a KNN model...")
        self.knn_model = self._knn_model() # creating the model
        knn_stats = self.class_evaluation(self.knn_model) # saving the scores
        self.models["KNN"] = knn_stats
        print("KNN ready ", "\u2713")
    def build_svc(self):
        # SVC
        print("Creating an SVC model...")
        self.svc_model = self._svc_model() # creating the model
        svc_stats = self.class_evaluation(self.svc_model) # saving the scores
        self.models["SVC"] = svc_stats
        print("SVC ready ", "\u2713")
    def build_ann(self, file_path): 
        # ANN Classifier
        print("Creating an ANN model...")
        self.ann_class_model = self._ann_class_model(file_path)
        ann_class_stats = self.ann_class_evaluation() # saving the scores
        self.models["ANN Classifier"] = ann_class_stats
        print("ANN Classifier ready ", "\u2713")
    
    # Internal methods actually building the GridSearchCV and models
    def _lor_model(self):
        """Creating a GridSearchCV model with a pipeline of 
        StandardScaler and Logistic Regression as estimator
        
        Different settings depending on self.class_balanced
        Returns:
            GridSearchCV: with Logistic Regression as base
        """
        if self.df.class_balanced:  
            pipe = make_pipeline(StandardScaler(),LogisticRegression())
        else:
            pipe = make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced'))
        param_grid = {"logisticregression__penalty": ["l1","l2"],
                          "logisticregression__C": np.linspace(0.1,4,10)}
        lor_grid = GridSearchCV(pipe,param_grid,cv=10)
        lor_grid.fit(self.df.X_train,self.df.y_train)
        return lor_grid
    def _knn_model(self):
        """Creating a GridSearchCV model with a pipeline of 
        StandardScaler and KNN as estimator
        
        Different settings depending on self.class_balanced
        Returns:
            GridSearchCV: with KNN as base
        """
        pipe = make_pipeline(StandardScaler(),KNeighborsClassifier())
        param_grid = {"kneighborsclassifier__n_neighbors": list(range(1,30))}
        knn_grid = GridSearchCV(pipe,param_grid,cv=10)
        knn_grid.fit(self.df.X_train,self.df.y_train)
        return knn_grid
    def _svc_model(self):
        """Creating a GridSearchCV model with a pipeline of 
        StandardScaler and Logistic Regression as estimator
        
        Different settings depending on self.class_balanced
        Returns:
            GridSearchCV: with Logistic Regression as base
        """
        if self.df.class_balanced:  
            pipe = make_pipeline(StandardScaler(),SVC(max_iter=100000))
        else:
            pipe = make_pipeline(StandardScaler(),SVC(max_iter=100000,class_weight='balanced'))
        if self.df.big_df:
            param_grid = {"svc__C": [0.001,0.05,1,5,50,100,500],
                      'svc__kernel':['linear','rbf']}
        else:
            param_grid = {"svc__C": [0.001,0.005,0.01,0.05,1,5,10,50,100,500,1000,5000],
                      'svc__kernel':['linear','rbf']}
        svc_grid = GridSearchCV(pipe,param_grid,cv=10)
        svc_grid.fit(self.df.X_train,self.df.y_train)
        return svc_grid
    def _ann_class_model(self, file_path):
        my_ann_model = MyANN(data_set=file_path,
                        target = self.df._y_name, 
                        loss = "binary_crossentropy")
        return my_ann_model
    
    # Evaluation methods
    def class_evaluation(self,model):
        """Method for collecting model scores and results for comparison

        Args:
            model: the ML model 

        Returns:
            dict: dict with best_params,accuracy,mean fitting time and
                        confusion matrix (text form)
        """
        best_params = model.best_params_
        preds = model.predict(self.df.X_test) # predictions based on X test
                                            # for the accuracy score
        accuracy = accuracy_score(self.df.y_test,preds)
        mean_fit = model.cv_results_["mean_fit_time"].mean()
        con_preds = model.predict(self.df._X) # predictions based on the whole X for 
                                        # the confusion matrix
        con_matrix = confusion_matrix(self.df._y,con_preds)
        return {"Best hyperparameters":best_params,
                "accuracy":accuracy,
                "mean fitting time":mean_fit,
                "confusion matrix [[TN,FP],[FN,TP]": con_matrix}
        
    def ann_class_evaluation(self):
        best_params = self.ann_class_model.model_loss()
        accuracy = self.ann_class_model.test_accuracy()
        mean_fit = None
        con_matrix = self.ann_class_model.con_matrix()
        return {"Best hyperparameters":best_params,
                "accuracy":accuracy,
                "mean fitting time":mean_fit,
                "confusion matrix [[TN,FP],[FN,TP]": con_matrix}

    def class_eval_display(self):
        """Method for displaying the report for each classifier
        """
        models = {
            "Logistic Regression": self.lor_model,
            "KNN": self.knn_model,
            "SVC": self.svc_model,
            "ANN": self.ann_class_model
        }
        for name, model in models.items(): 
            print(name)
            print("-"*45)
            self.class_report(name,model)
            self.matrix_display(name,model)
            print("-"*45)
        self.print_summary()
    def class_report(self,name,model):
        """Method for printing out the classification report

        Args:
            model: model for which to print out the classification report
        """
        if name == "ANN":
            try:
                self.ann_class_model.print_classification_report()
            except ValueError:
                print("The random division of testing data gives insufficient data"\
                    " for ANN's classification report and confusion matrix.")
        else:
            class_report = classification_report(self.df._y,model.predict(self.df._X))
            print(class_report)
           
              
    def matrix_display(self,name,model):
        """Method for displaying the confusion matrix using the
        ConfusionMatrixDisplay method

        Args:
            name (str): name of the model for the title label
            model: the ML model
        """
        if name == "ANN":
                print("See the confusion matrix in the pop up window.")
                self.ann_class_model.con_matrix_display()
        else:
            print("See the confusion matrix in the pop up window.")
            plot = ConfusionMatrixDisplay.from_estimator(model,self.df._X,self.df._y)
            plt.title(name)
            plt.show()
    def print_summary(self):
        """Method for displaying the summary for all classifiers
        """
        print("S U M M A R Y")
        # creating a dataframe with the stats for all classifiers for comparison
        df_stats = pd.DataFrame(data = self.models).transpose()
        df_stats.drop("Best hyperparameters", axis=1, inplace=True) # best_params is not needed
        df_stats["accuracy"] = df_stats["accuracy"].astype("float") # converting accuracy into float
                                                    # to be able to automatically pick the best value
        
        self.final_summary_df = df_stats
        print(self.final_summary_df)
        
        self.set_recommended_model() # saving the recommended model
    def set_recommended_model(self):
        """Method for saving the recommended model in the self.recommended_model variable

        Args:
            name (str): name of the model that performed best
        """
        winner = self.final_summary_df["accuracy"].idxmax() # choosing the index of the highest accuracy score
        print(f"\nRecommended model based on the accuracy score: >>> {winner} <<<")
        match winner:
            case "Logistic Regression":
                self.recommended_model = self.lor_model
            case "KNN":
                self.recommended_model = self.knn_model
            case "SVC":
                self.recommended_model = self.svc_model
            case "ANN Classifier":
                self.recommended_model = self.ann_class_model
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
                {"1": "Logistic Regression",
                    "2": "KNN",
                    "3": "SVC",
                    "4": "ANN Classifier"})
            if choice == "1":
                model = self.lor_model
            elif choice == "2":
                model = self.knn_model
            elif choice == "3":
                model = self.svc_model
            else:
                model = self.ann_class_model
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
        
        if model == self.ann_class_model:
            self.ann_class_model.save_model(name)
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
    
