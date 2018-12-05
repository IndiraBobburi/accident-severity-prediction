# Accident severity prediction
The project intends to predict the accident severity for given road, weather, climate and other conditions among others. The dataset is taken from kaggle- UK roads safety: Traffic accidents and vehicles. The dataset contains 1.92 million records and 34 columns. Data cleaning and preprocessed the data. The data corresponding to slight severity is 84.84%, serious severity is 13.86% and for fatal severity is 1.30%. Several processing techniques as undersampling and oversampling were tried. Our main aim is to predict the serious and fatal severity with high precision and F1 score. Seven different classifiers were tried on this dataset. Random forest classifier worked best for our dataset. The prediction in the UI is based on that model.

## Technologies used:
#### Front end is developed with HTML, CSS and JavaScript and Backend is developed with python Flask

## Environment setup:

> 1. Clone the github and goto the root folder
> $git clone https://github.com/IndiraBobburi/accident-severity-prediction
>
> 2. Create a virtual environment with python3
> $virtualenv -p /usr/local/bin/python3 dependencies
>
> 3. Activate the virtual environment
> $source dependencies/bin/activate
>
> 4. Install all the dependencies
> $pip install -r requirements.txt
>
> 5. Run the app
> $python3 app/main.py
>
> 6. Open the port on browser and use the site.
> www.localhost:5000

## Preview of the application:
![](images/preview.png)
