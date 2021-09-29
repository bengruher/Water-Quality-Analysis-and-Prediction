# Filename: water_quality_preprocessing.py
# Date of creation: 9/10/21
# Credit: the general structure of this code came from the Sagemaker example notebook. I adapted the code to meet the requirements of my project

import sys
import os
import argparse
import csv
import json
from io import StringIO
import numpy as np
import pandas as pd

from sklearn.externals import joblib # works with sklearn versions <= 0.21
# import joblib # works with sklearn versions >= 0.23
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sagemaker_containers.beta.framework import content_types, encoders, env, modules, transformer, worker

feature_columns_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

label_column = 'Potability'

feature_columns_dtype = {
    'ph' : np.float64, 
    'Hardness' : np.float64, 
    'Solids' : np.float64, 
    'Chloramines' : np.float64, 
    'Sulfate' : np.float64, 
    'Conductivity' : np.float64, 
    'Organic_carbon' : np.float64, 
    'Trihalomethanes' : np.float64, 
    'Turbidity' : np.float64
}

label_column_dtype = {'Potability' : np.float64}

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))

    raw_data = [ pd.read_csv(file) for file in input_files ]

    concat_data = pd.concat(raw_data)
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', MinMaxScaler())])
    
    # TODO: implement SMOTE
    
    numeric_transformer.fit(concat_data)
    
    joblib.dump(numeric_transformer, os.path.join(args.model_dir, "model.joblib"))
   


# ---------------------------------------------------------------------------------------------------

# We override several functions (for example, predict_fn because the default predict_fn uses .predict() and we want to use .transform()

def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), 
                         header=None)

        if len(df.columns) == len(feature_columns_names) + 1:
            # This is a labelled example, includes the ring label
            df.columns = feature_columns_names + [label_column]
        elif len(df.columns) == len(feature_columns_names):
            # This is an unlabelled example.
            df.columns = feature_columns_names

        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))
        
        
        
def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), accept, mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))
        

# This function takes the input data (parsed by input_fn into a nice Pandas DF with the right columns) and the deserialized model from model_fn (below, which I guess SageMaker calls automatically since it is not an explicit part of this script). We then call .transform() to transform the data (whether it is the training data or the inference data).
def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    features = model.transform(input_data)

    if label_column in input_data:
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data[label_column], axis=1)
    else:
        # Return only the set of features
        return features
    
    
def model_fn(model_dir):
    ### Deserialize fitted model ###
    preprocessor = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return preprocessor
    