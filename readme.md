# Yelp Open Dataset business attributes vectorize tool
## About
Business attributes in Yelp Open Dataset are stored as dict. However, we require vectorized attributes in some scenarios, such as deep learning model training. This piece of script is functioned for this. 

Please be noted: this utility is not tested on any other dataset. If you want to vectorize any other dict, preserve your original dataset well firstly, and then you can have a test. Issues are welcomed if you find any bugs on other dataset. 

## Input and Output
You need indicate where your Yelp business dataset are stored as input. And this script will output two files in the form of pickle file: one is a numpy array kept all the vectorized attributes, and a python dict to illustrate how the string attributes are tranformed into integers.

## Running
```batch
python attrivec.py dataset/yelp_academic_dataset_business.json .
```

## Dependencies
```
numpy = 1.24.3
pandas = 1.5.3
python = 3.9.15
```

## Disclaimer
The development and maintainence of this project has nothing relations with Yelp. In this project, the mention of "Yelp" is purely indicative of the fact that this project is able to work for the Yelp Open Dataset and is not affiliated with or endorsed by Yelp.
