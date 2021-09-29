import numpy as np
import pandas as pd

bucket = '<INSERT BUCKET NAME HERE>'
input_prefix = 'input'
input_key = 'water_potability.csv'

region = 'us-west-2'

s3_input_uri = 's3://{}/{}/{}'.format(bucket, input_prefix, input_key)

df = pd.read_csv(s3_input_uri)

cols = df.columns

df_notpotable = df[df['Potability'] == 0]
df_potable = df[df['Potability'] == 1]

# -----------------------------------------------------
# Imputing
from sklearn.impute import SimpleImputer

impute = SimpleImputer(missing_values=np.nan, strategy='mean')

# for df_notpotable
df_notpotable = pd.DataFrame(impute.fit_transform(df_notpotable), columns = cols)

#for df_potable
df_potable = pd.DataFrame(impute.fit_transform(df_potable), columns = cols)

df = pd.concat([df_notpotable, df_potable])

df = df.sample(frac = 1)

x = df.drop('Potability', axis = 1)
y = df['Potability']

# -----------------------------------------------------
# Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x)

x = scaler.transform(x)

x = pd.DataFrame(x)

# -----------------------------------------------------
# Oversampling
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
x_res, y_res = oversample.fit_resample(x,y)

# -----------------------------------------------------
# Modeling
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, stratify = y)

rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)

import joblib
joblib.dump(rf, '/opt/ml/model/model.joblib')
