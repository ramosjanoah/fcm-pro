import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

COLUMNS_TO_CONSIDER = ['capital-gain', 'capital-loss', 'hours-per-week']

# load dataset
path = "CensusIncome/CencusIncome.test.txt"

# read dataset
data_raw = pd.read_csv(path)

# read data test
num_train = 0.8 * len(data_raw)
data_test = data_raw[int(num_train):]

# read headers
headers = list(data_raw.columns.values)

# cleaning data, remove blank space infront of string
for index, row in data_raw.iterrows():
    for header in headers:
        if str(type(row[header])) == "<class 'str'>":
            data_raw.at[index,header] = row[header].replace(" ","")

# split categorical and numeric
data_raw_categorical = data_raw.select_dtypes(include=[object])
# data_raw_numeric = data_raw.select_dtypes(exclude=[object])
# print(data_raw_numeric)
data_raw_numeric = data_raw.ix[:, COLUMNS_TO_CONSIDER]

# [NORMALIZATION NUMERIC]
# get header to normalize
header_numeric = list(data_raw_numeric.columns.values)

# max value of normalization
max_value_norm_numeric = 1

data_raw_numeric[header_numeric] = data_raw_numeric[header_numeric ].apply(lambda x: (x - x.min()) * max_value_norm_numeric/ (x.max() - x.min()))
data_raw_numeric = data_raw_numeric.values.tolist()
