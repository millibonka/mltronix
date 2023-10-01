import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

class PathSelector:
    def __init__(self, root):
        
        self.file_path = None
        
        self.root = root
        self.root.title("Choose your CSV file")
        self.root.config(padx=20, pady=10)
        
        self.label_path = tk.Label(self.root, text="Enter path or browse for your file. ")
        self.label_path.grid(row=0, column=1, columnspan=2)
        
        self.label_path = tk.Label(self.root, text="Enter path: ")
        self.label_path.grid(row=1, column=0)

        self.entry_path = tk.Entry(self.root, width=30)
        self.entry_path.grid(row=1, column=1, columnspan=2)
    
        self.browse_button = tk.Button(self.root, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=1, column=3)
        
        # widget for selecting the model type
        self.label_select = tk.Label(self.root, text="Model type: ")
        self.label_select.grid(row=3, column=0)
        self.selection = tk.StringVar()
        self.combo = ttk.Combobox(self.root, textvariable=self.selection, values=["Classifier", "Regressor"])
        self.combo.grid(row=3, column=1)
        self.combo.set("None")
        
        self.ready_button = tk.Button(self.root, text="Ready!", command=self.load_csv)
        self.ready_button.grid(row=4, column=3)
        
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")], initialdir="/", title="Select a CSV File")
        self.entry_path.delete(0, tk.END)  # Clear any previous text
        self.entry_path.insert(0, file_path)  # Set the selected file path in the entry widget

    def load_csv(self):
        self.file_path = self.entry_path.get()
        self.model = self.selection.get()
        self.root.destroy()
        

if __name__ == "__main__":
    root = tk.Tk()
    app = PathSelector(root)
    root.mainloop()