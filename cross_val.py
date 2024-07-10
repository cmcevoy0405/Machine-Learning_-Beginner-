import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

train_data = pd.read_csv(r"train.csv")
test_data = pd.read_csv(r"test.csv")

train_data.dropna(axis = 0, subset = ['SalePrice'], inplace = True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis = 1, inplace = True)

numerical_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['float64', 'int64']]
X = train_data[numerical_cols].copy()
X_test = test_data[numerical_cols].copy()

my_pipeline = Pipeline(steps =[('preprocessor', SimpleImputer()),
                            ('model', RandomForestRegressor(n_estimators = 50, random_state = 0))])
scores = -1 * cross_val_score(my_pipeline, X, y, cv = 5, scoring = 'neg_mean_absolute_error')

print("MAE =", scores.mean())

def get_score(n_estimators):
    my_pipeline = Pipeline(steps = [('preprocessor', SimpleImputer()), ('model', RandomForestRegressor(n_estimators, random_state = 0))])

    scores = -1 * cross_val_score(my_pipeline, X, y, cv = 3, scoring = 'neg_mean_absolute_error')

    return scores.mean()

results = {}
for i in range(1,9):
    results[50*i] = get_score(50*i)

# Plot the results
print("Results:", results)
plt.plot(list(results.keys()), list(results.values()))
plt.show()
