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

# [NORMALIZATION NUMERIC]
# get header to normalize
header_numeric = list(data_raw_numeric.columns.values)

# max value of normalization
max_value_norm_numeric = 1

data_raw_numeric[header_numeric] = data_raw_numeric[header_numeric ].apply(lambda x: (x - x.min()) * max_value_norm_numeric/ (x.max() - x.min()))
data_raw_numeric = data_raw_numeric.values.tolist()

creating_one_hot = True
max_value_norm_nominal = 1 #max value to normalize one hot. If you dont want, just make it 1

if creating_one_hot:
    # [CREATING ONE HOT ON NOMINAL]       
    # create label encoder
    label_encoder = preprocessing.LabelEncoder()
    
    # change category to integer
    categorical_label = data_raw_categorical.apply(label_encoder.fit_transform)
    
    # create one hot encoder
    one_hot_encoder = OneHotEncoder()
    
    # fitting one-hot with categorical label
    one_hot_encoder.fit(categorical_label)
    
    # creating one hot lables
    onehotlabels = one_hot_encoder.transform(categorical_label).toarray()
    
    # normalization one hot
    if max_value_norm_numeric != 1:
        onehotlabels = onehotlabels*max_value_norm_nominal
    
    nominal_value = onehotlabels

else:
    nominal_value = data_raw_categorical.values.tolist()

    
# [MERGE NUMERIC AND NOMINAL]
data_list = []
for index, value in enumerate(data_raw_numeric):
    data_list.append(data_raw_numeric[index] + list(nominal_value[index]))
    
    
