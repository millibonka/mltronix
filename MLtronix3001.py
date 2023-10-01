import tkinter as tk
from path_selector import PathSelector
from data_preprocessing import DataPP
from regressor import RegressorBuilder
from classifier import ClassifierBuilder

class MLApp:
    
    def __init__(self, file_path:str = None,
                 model_type:str = None):
        """
            Class for automated Machine Learning.
        Args:
            file_path (str, optional): If no file path is provided, a GUI pop-up window shows up. Defaults to None.
            model_type (str, optional): Model type can be provided at start of the app, but is also later determined by the data. Defaults to None.
        """
        # step 1: loading the csv file and the user specified model type (optional, as can be set later)
        if file_path is None:
            root = tk.Tk()
            selector = PathSelector(root)
            root.mainloop()
            self.file_path = selector.file_path
            self.model_type = selector.model.lower()
        else:
            self.file_path = file_path
            self.model_type = model_type
    
    def step_two(self):
        # step 2: data preprocessing
        self.data = DataPP(self.file_path, self.model_type)
        if self.data != None:
            self.data.display_info()
            self.data.choose_y()
            self.data.validate_model_type()
            if self.data.ready_for_ML():
                self.data.set_X()
                self.data.tt_split()
                self.data.df_size_setter()
                self.data.check_class_balance()
    
    def step_three(self):
            if self.data.model_type == "regressor":
                self.model = RegressorBuilder(df=self.data)
                self.model.build_all(self.file_path)
                self.model.reg_display_reports()
                self.model.print_summary()
                self.model.choose_model()
            elif self.data.model_type == "classifier":
                self.model = ClassifierBuilder(df=self.data)
                self.model.build_all(self.file_path)
                self.model.class_eval_display()
                self.model.choose_model()


if __name__ == "__main__":
    app = MLApp()
    app.step_two()
    app.step_three()