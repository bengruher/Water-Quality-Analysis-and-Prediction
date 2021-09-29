# Water Quality Analysis and Prediction

### Overview

This project involved data exploration and preprocessing of water potability data. Water is 'potable' if it is safe for human consumption. We used attributes such as pH, hardness, conductivity, and more to create a predictive model to classify a water sample as potable or not potable. 

### Amazon Sagemaker Integration

This project was primarily created to develop and showcase skill deploying various model types on Amazon Sagemaker, AWS' flagship machine learning service. The notebooks in this repository demonstrate the use of Sagemaker built-in algorithms, custom algorithms on popular frameworks such as XGBoost and Scikit-learn (this is called script mode on Sagemaker), and bring-your-own-container models. 

### Running the Project

To run this project, you will need to download the dataset from [this](https://www.kaggle.com/adityakadiwal/water-potability) Kaggle site. You will also need to create an S3 bucket in AWS and upload the dataset to that bucket. Note that you will need to either keep the default name for the data file or change all references in the notebooks and scripts to the name of your file. You will also need to change all instances of "INSERT BUCKET NAME HERE" (also remove the angled brackets) in the notebooks and scripts with the name of your S3 bucket. In order to run the bring-your-own-container algorithm, you will need to create a container using the provided Dockerfile in the Container folder. 
