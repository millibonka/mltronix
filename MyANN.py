""" Inlämningsuppgift #1 / Deep Learning
Milena Miernik"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

class MyANN:
    """Class to make an ANN model as per specified parameters
    """
    def __init__(self, data_set: str = None,
                 target: str = None,
                 hidden_layer_sizes: tuple = (100,),
                 activation: str = "relu",
                 loss: str = "mse",
                 optimizer: str = "adam",
                 batch_size: int = 32,
                 epochs: int = 1,
                 monitor: str = "val_loss",
                 patience: int = 1,
                 mode: str = "auto",
                 verbose: int = 1,
                 use_multiprocessing: bool = False):
        

        # validation of user's hyperparameters
        self.__df = self._validate_df(data_set)
        self.__target = self._validate_target(target)
        self.__hidden_layers_size = self._validate_hidden_layers(hidden_layer_sizes)
        self.__activation = self._validate_activation(activation)
        self.__loss = self._validate_loss(loss)
        self.__optimizer = self._validate_optimizer(optimizer)
        self.__batch_size = self._validate_batch_size(batch_size)
        self.__epochs = self._validate_epochs(epochs)
        self.__monitor = self._validate_monitor(monitor)
        self.__patience = self._validate_patience(patience)
        self.__mode = self._validate_mode(mode)
        self.__verbose = self._validate_verbose(verbose)
        self.__use_multiprocessing = self._validate_multiprocessing(use_multiprocessing)
                
        # creating other hyperparameters
        self.__X = self.__df.drop(self.__target,axis = 1)
        self._check_num_values() # check for non-numeric values in X
        self.__y = self.__df[self.__target]
        self.__scaler = None
        self.classes_, self.n_outputs_, self.out_activation_, self.__target_type =\
                                                self._check_target_type()
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = \
                                                self._data_preprocessing()
        
        # creating the model
        self.__model = self._build_model()
        
        # saving additional information
        self.__metrics = pd.DataFrame(self.__model.history.history)
        self.loss_ = self.__metrics["loss"].iloc[-1].round(2)
        self.best_loss_ = self.__metrics["loss"].min().round()
        self.features_ = tuple(self.__X.columns)
        self.n_layers_ = len(self.__hidden_layers_size)+2
        
    # a group of validation methods
    def _validate_df(self, csv_path):
        """Method to validate the given csv file
        and return a pandas dataset, otherwise raises an error.
        Specifies three specific errors and one general.
        Includes a check for missing and non-numeric values."""
        try:
            df = pd.read_csv(csv_path)

        except FileNotFoundError:
            print("The csv file was not found at the given path.")
        except pd.errors.ParserError: 
            print("Invalid CSV file.")
        except pd.errors.EmptyDataError:
            print("The chosen CSV file is empty.")
        except:
            print("Invalid CSV file.")
        else:
            df.head()
            if df.isna().sum().sum() > 0:
                raise Exception("The csv file contains missing values and is therefore not ready for DL processing.")
            return df
        
    
    def _validate_target(self,target):
        """Method to validate the target, returns a str with the target name,
        in case of bad user input gives prompt to give a new target name"""
        y_target = target
        while y_target not in self.__df.columns:
            print("Incorrect target name. Try again. Here are the column names:\n")
            print(self.__df.columns)
            y_target = input("Target: ")
        return y_target
    
    def _validate_hidden_layers(self, hidden_layer_sizes):
        """Method to validate the hidden_layer_sizes, returns a tuple
        or raises an error"""
        if not isinstance(hidden_layer_sizes, tuple):
            raise TypeError("Hyperparameter hidden_layer_sizes must be a tuple.")
        if not all(isinstance(x, (int, float)) for x in hidden_layer_sizes):
            raise TypeError("All elements in hidden_layes_sizes must be numbers.")
        return hidden_layer_sizes
    
    def _validate_activation(self, activation):
        """Method to validate the activation. Returns a string or
        raises an error"""
        options = ["relu", "sigmoid", "softmax", "tanh"]
        if activation not in options:
            raise ValueError("Wrong activation value.")
        return activation
    
    def _validate_loss(self,loss):
        """Method to validate loss. Returns a string or
        raises an error."""
        options = ["mse", "binary_crossentropy", "categorical_crossentropy"]
        if loss not in options:
            raise ValueError("Wrong loss value.")
        return loss
    
    def _validate_optimizer(self, optimizer):
        """Method to validate optimizer. Returns a string
        or raises an error."""
        options = ["adam", "rmsprop", "agd"]
        if optimizer not in options:
            raise ValueError("Wrong optimizer value.")
        return optimizer
    
    def _validate_batch_size(self, batch_size):
        """Method to validate batch_size. Returns an integer
        or raises an error.
        A while loop ensures the value is a power of 2."""
        if batch_size != None and not isinstance(batch_size,int):
            raise ValueError("Wrong batch_size value")
        
        if (batch_size & (batch_size - 1)) != 0 or batch_size == 0:
            raise ValueError("Batch size must be a power of 2.")
        return batch_size
    
    def _validate_epochs(self, epochs):
        """Method to validate epochs. Returns an integer
        or raises an error."""
        if not isinstance(epochs, int) or epochs < 0:
            raise ValueError("Wrong epochs value.")
        return epochs
    
    def _validate_monitor(self,monitor):
        """Method to validate monitor. Returns a string
        or raises an error."""
        options = ["val_loss", "accuracy"]
        if monitor not in options:
            raise ValueError("Wrong monitor value.")            
        return monitor
    
    def _validate_patience(self,patience):
        """Method to validate patience. Returns an integer
        or raises an error."""
        if not isinstance(patience, int):
            raise ValueError("Wrong patience value.")
        return patience
    
    def _validate_mode(self, mode):
        """Method to validate mode. Returns a string
        or raises an error."""
        options = ["auto", "min", "max"]
        if mode not in options:
            raise ValueError("Wrong mode value.")
        return mode
    
    def _validate_verbose(self, verbose):
        """Method to validate verbose. Returns an integer
        or raises an error."""
        options = [0,1,2]
        if verbose not in options:
            raise ValueError("Wrong verbose value.")
        return verbose
    
    def _validate_multiprocessing(self,use_multiprocessing):
        """Method to validate use_multiprocessing. Returns a bool
        or raises an error."""
        if not isinstance(use_multiprocessing, bool):
            raise ValueError("Wrong use_multiprocessing value.")
        return use_multiprocessing
    
    # other methods 
    def _check_num_values(self):
        """Method for checking if there are any non-numeric
        values ns X"""
        # "coerce" replaces non-numeric values with NaN
        df = self.__X.apply(pd.to_numeric, errors='coerce')
        # check for NaN values
        if df.isna().values.any():
            self.__X = pd.get_dummies(self.__X, drop_first=True)
            print("Non-numerical values in X have been converted to dummies.")
            
    def _check_target_type(self):
        """Method to check the type of target and set other
        hyperparameters accordingly.
        Returns:
            classes, outputs, out_activation, target
        """
        from sklearn.utils.multiclass import type_of_target
        target = type_of_target(self.__y)
        classes = None
        outputs = None
        out_activation = None
        if target == "continuous":
            classes = 0
            outputs = 1
            out_activation = "Not applicable"
        elif target == "binary":
            classes = 2
            outputs = 2
            out_activation = "sigmoid"
        else:
            classes = self.__y.unique()
            outputs = len(classes)
            out_activation = "softmax"
        return classes, outputs, out_activation, target
    
    # methods that prepare, build and train the model
    def _data_preprocessing(self):
        """Method for further data preprocessing, trains and saves
        the scaler, and returns X_train, X_test, y_train, y_test """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        if self.__target_type != "continuous":
            if self.__target_type == "binary":
                self.__y = pd.get_dummies(self.__y, drop_first=True)
            else:
                self.__y = pd.get_dummies(self.__y)
        Xa = self.__X.values
        ya = self.__y.values
        # splitting the data
        if len(self.__X) > 1000: # if the dataset is more than 1000 rows
            x = 0.1 # test_size set as 0.1
        else:
            x = 0.3 # test_size set as 0.3
        X_train, X_test, y_train, y_test = train_test_split(Xa, ya,\
                                                    test_size=x, random_state=101)
        
        # scaling
        self.__scaler = MinMaxScaler()
        X_train = self.__scaler.fit_transform(X_train)
        X_test = self.__scaler.transform(X_test)
             
        return X_train, X_test, y_train, y_test
    
    def _build_model(self):
        """Method for building and training the model.
        Returns a trained model.
        """
        # libraries
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        
        model = Sequential()
        # input layer
        model.add(Dense(units = self.__X_train.shape[1], activation = self.__activation))
        
        # hidden layers
        for layer in self.__hidden_layers_size:
            if layer > 0: 
                model.add(Dense(units = layer, activation = self.__activation))
            else:
                model.add(Dropout(abs(layer)))
        
        # output and compile layers 
        if self.__target_type == "continuous":
            # building regressor
            model.add(Dense(1))
            model.compile(loss = self.__loss, optimizer = self.__optimizer)
        elif self.__target_type == "binary":
            # building binary classifier
            model.add(Dense(1, activation = self.out_activation_))
            model.compile(loss = self.__loss, optimizer = self.__optimizer,
                metrics = ["accuracy"])
        else:
            # building multiclass classifier
            model.add(Dense(units = len(self.classes_), activation = self.out_activation_))
            model.compile(loss = self.__loss, optimizer = self.__optimizer,
                metrics = ["accuracy"])
        
        # early stop
        early_stop = EarlyStopping(monitor = self.__monitor, mode = self.__mode,
                                patience = self.__patience, verbose = self.__verbose)
        
        # training the model
        model.fit(x = self.__X_train, y = self.__y_train, epochs = self.__epochs,
                batch_size = self.__batch_size, callbacks = [early_stop],
                validation_data = (self.__X_test, self.__y_test),
                verbose = self.__verbose, use_multiprocessing = self.__use_multiprocessing)
        return model
    
    # supporting methods
    def model_loss(self):
        """Method to return a Pandas dataframe with loss,
        val_loss, accuracy and val_accuracy"""
        return self.__metrics

    
    def model_predict(self, user_x: list):
        """Method taking a list of data and returning a prediction"""
        user_x = np.array(user_x) # changing input into an array
        user_x = self.__scaler.transform(user_x.reshape(-1,self.__X.shape[1]))
        prediction = 0
        if self.__target_type == "continuous":
            prediction = self.__model.predict(user_x)
        elif self.__target_type == "binary":
            prediction = (self.__model.predict(user_x)>.5)
        else:
            prediction = np.argmax(self.__model.predict(user_x), axis = 1)
        return prediction
    
    def save_model(self, name: str):
        self.__model.save(name + ".h5")
    def load_model(self, name: str):
        from tensorflow.keras.models import load_model
        if name.split(".")[-1] == "h5":
            model = load_model(name, compile=False)
            return model
        else:
            raise ValueError("Only h5 format is supported.")

    def print_classification_report(self):
        """Method printing the classification report"""
        if self.__target_type != "continuous":
            from sklearn.metrics import classification_report
            if self.__target_type == "binary":
                try:
                    pred = self.__model.predict(self.__X_test)>0.5
                    print(classification_report(self.__y_test, pred))
                except ValueError as err:
                    print(err)
            else:
                try:
                    predictions = np.argmax(self.__model.predict(self.__X_test),\
                                                                        axis=1)
                    pred = pd.get_dummies(predictions)
                    print(classification_report(self.__y_test, pred))
                except ValueError as err:
                    print("Classification report unavailable for this prediction due to random factors. Try again.")
    
    
    def rmse(self):
        """Returning RMSE"""
        from sklearn.metrics import mean_squared_error
        rmse = mean_squared_error(self.__y_test, \
                    self.__model.predict(self.__X_test))**0.5
        return rmse

    def mae(self):
        """Returning MAE"""
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(self.__y_test,self.__model.predict(self.__X_test))
        return mae
    

    def ann_r2(self):
        """Returning r2 score"""
        from sklearn.metrics import r2_score
        r2 = r2_score(self.__y_train, self.__model.predict(self.__X_train))
        return r2
    
    def test_accuracy(self):
        """Returns test accuracy value"""
        test_loss, test_accuracy = self.__model.evaluate(self.__X_test, self.__y_test)
        return test_accuracy

    def con_matrix(self):
        """Method returning the confusion matrix"""
        from sklearn.metrics import confusion_matrix
        if self.__target_type == "binary":
            y_pred_proba = self.__model.predict(self.__X_test)
            y_pred_binary = (y_pred_proba > 0.5).astype(int)
            confusion = confusion_matrix(self.__y_test, y_pred_binary)
        else:
            y_pred = self.__model.predict(self.__X_test)
            y_pred_labels = np.argmax(y_pred,axis=1)
            confusion = confusion_matrix(np.argmax(self.__y_test, axis=1), y_pred_labels)
        return confusion
    
    def con_matrix_display(self):
        confusion = self.con_matrix()
        print(confusion)
        display = ConfusionMatrixDisplay(confusion)
        display.plot(values_format='.4g')
        plt.title('ANN')
        #plt.show() makes the matrix pop up twice
        
    def get_X(self):
        return self.__X
    
    def get_preprocessing_values(self):
        """Method for ease of testing"""
        return self.__X_train, self.__scaler
