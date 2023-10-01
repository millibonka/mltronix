# mltronix
automated machine learning app with support for ANN

This app was done as a school project for Python-developer with focus on AI at Teknikhögskolan, Göteborg, Sweden. 
It takes a CSV file with data and automatically builds different kinds of regressor or classifier models, and even a deep learning ANN model (this part is still under development for improved functionality). 
- the app takes in a .csv file either through a GUI window or directly from command line
- verifies the data in the csv file and if the data is ready for machine learning:
- does simple automatic data pre-processing
- verifies if the data requires a regressor (for continuous data) or a classifier (for categorical data)
- uses pipeline and gridsearchCV to find the best parameters for each type of model
- creates the appropriate ANN model with parameters adjusted for user's data
- trains the models on user data
- displays the statistics and performance of each model, and offers recommendation for the best model
- the user can save the already trained model
