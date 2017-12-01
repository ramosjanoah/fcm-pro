import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

# load dataset
path = "CensusIncome/CencusIncome.data.txt"

# read dataset
data_raw = pd.read_csv(path)

# read headers
headers = list(data_raw.columns.values)

# cleaning data, remove blank space infront of string
for index, row in data_raw.iterrows():
    for header in headers:
        if str(type(row[header])) == "<class 'str'>":
            data_raw.at[index,header] = row[header].replace(" ","")

# split categorical and numeric
data_raw_categorical = data_raw.select_dtypes(include=[object])
data_raw_numeric = data_raw.select_dtypes(include=[int,float])
data_raw_numeric = data_raw_numeric.values.tolist()

# create label encoder
label_encoder = preprocessing.LabelEncoder()

# create categorical
categorical_label = data_raw_categorical.apply(label_encoder.fit_transform)

# create one hot encoder
one_hot_encoder = OneHotEncoder()

# one hot with categorical data
one_hot_encoder.fit(categorical_label)

# creating one hot lables
onehotlabels = one_hot_encoder.transform(categorical_label).toarray()

# merge numeric and categorical
data_list = []
for index, value in enumerate(data_raw_numeric):
    data_list.append(data_raw_numeric[index] + list(onehotlabels[index]))
    
