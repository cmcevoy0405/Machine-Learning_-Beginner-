import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

X = pd.read_csv(r"train.csv")
X_test = pd.read_csv(r"test.csv")

#Remove missing values from predictor, then drop from train data
X.dropna(axis = 0, subset = ['SalePrice'], inplace = True)
y = X.SalePrice
X.drop(['SalePrice'], axis = 1, inplace = True)

cols_missing_values = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_missing_values, axis = 1, inplace = True)
X_test.drop(cols_missing_values, axis = 1, inplace = True)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators = 100, random_state = 0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

#Drop categorical variables
drop_X_train = X_train.select_dtypes(exclude = 'object')
drop_X_valid = X_valid.select_dtypes(exclude = 'object')

print("MAE from dropping categorical columns: ")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

#Different values in training and validation set
print("Unique values in condition 2 in training data:", X_train['Condition2'].unique())
print("Unique values in condition 2 in validation data:", X_valid['Condition2'].unique())

#Drop not matching categorical
#Categorical columns in training data
object_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

good_label_cols = [col for col in object_cols if set(X_valid[col]).issubset(set(X_train[col]))]

bad_label_cols = list(set(object_cols) - set(good_label_cols))

print('Categorical columns that will be ordinal encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

label_X_train = X_train.drop(bad_label_cols, axis = 1)
label_X_valid = X_valid.drop(bad_label_cols, axis = 1)

ordinal_encoder = OrdinalEncoder()
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols])

print("MAE from ordinal encoder:")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

sorted(d.items(), key = lambda x: x[1])

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]
high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

print("Categorical variables to be one hot-encoded:", low_cardinality_cols)
print("Categorical variables to be dropped:", high_cardinality_cols)

#Apply onehotencoder to dataset
OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

#One hot encoding removes the index
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

#drop object cols, to be replaced by one hot encoder
num_X_train = X_train.drop(object_cols, axis = 1)
num_X_valid = X_valid.drop(object_cols, axis = 1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis = 1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis = 1)

#Ensure only string types occur in OH columns
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("MAE from OneHotEncoder: ")
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
