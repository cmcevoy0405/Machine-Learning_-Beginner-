import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

X_full = pd.read_csv(r"train.csv")
X_test_full = pd.read_csv(r"test.csv")

#Dropping empty rows in predictor variable
X_full.dropna(axis = 0, subset = 'SalePrice', inplace = True)

y = X_full.SalePrice
#Drop predictor variable from train data set
X_full.drop(['SalePrice'], axis = 1, inplace = True)

#Dropping categorical variables
X = X_full.select_dtypes(exclude = 'object')
X_test = X_test_full.select_dtypes(exclude = 'object')

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

print(X_train.shape)
missing_values_by_column_count = (X_train.isnull().sum())
print(missing_values_by_column_count[missing_values_by_column_count > 0])

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators = 100, random_state = 0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

#Dropping columns with missing data
missing_cols = [col for col in X_train.columns if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(missing_cols, axis = 1)
reduced_X_valid = X_valid.drop(missing_cols, axis = 1)

print("MAE for dropping columns: ")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

#Using imputation instead
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE for imputed columns: ")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid ))

#Changing imputation strategy
new_imputer = SimpleImputer(strategy = "median")
new_imputer_X_train = pd.DataFrame(new_imputer.fit_transform(X_train))
new_imputer_X_valid = pd.DataFrame(new_imputer.transform(X_valid))

new_imputer_X_train.columns = X_train.columns
new_imputer_X_valid.columns = X_valid.columns

model = RandomForestRegressor(n_estimators = 100, random_state = 0 )
model.fit(new_imputer_X_train, y_train)
preds_valid = model.predict(new_imputer_X_valid)
print("MAE from median strategy: ")
print(mean_absolute_error(y_valid, preds_valid))

final_X_test = pd.DataFrame(new_imputer.transform(X_test))
preds_test = model.predict(final_X_test)
print(preds_test)



