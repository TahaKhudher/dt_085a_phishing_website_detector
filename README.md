# dt_085a_phishing_website_detector

This is a university project in the Course Datamining and Machine learning at Mid Sweden University. 

Created By Taha Khudher and Hamza Ali. 

### Purpose
The main purpose of this project was to create a phishing website detector using traditional machine learning methods. To find the best model fit for this project we compared three different models, KNN, SVM and Random forest to later see what method is best and can be used for deployment later.

### Dataset
Abdelhamid, N. (2014). Website Phishing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5B301.
Instances: 1353
Features: 9
Missing Values: NO
Feature Type: Integer



### Running the project
Start by downloading the library requirements. 

```shell
pip install -r requirments.txt
```

Run the 
```shell
python src/train.py 
```
file from root directory of the project.

Then run the 
```shell
python src/compare_predictions.py 
```
to see the predictions and the result of the models. 
