import argparse
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from Utils.save_config import save_model_config

parser = argparse.ArgumentParser(description='Model_Training')

#Models args
parser.add_argument('--data_path',type = str , default = 'Data/test.csv')
parser.add_argument('--preprocessing', type=str, default='StandardScaler',
                    help='MinMaxScaler for min/max scaling and StandardScaler for standardizing')
parser.add_argument('--RandomSeed',type =int,default=17)
parser.add_argument('--test_size',type =int,default=0.2)
parser.add_argument('--Degree',type = int,default=1)
parser.add_argument('--Random_Search', type = bool, default=True, help='Use RandomizedSearchCV')
parser.add_argument('--save_model_config',type =bool,default=True)
parser.add_argument('--Config_path',type = str, default='Config/model_config.yaml')

args = parser.parse_args()
print(args)


def train(X_t, y_t, model, grid):
    numeric_features = (
        X_t.select_dtypes(include=["float64"])
        .columns
        .tolist()
    )

    categorical_features = (
        X_t.select_dtypes(exclude=["float64"])
        .columns
        .tolist()
    )

    if args.preprocessing == 'MinMaxScaler':
        scaler = MinMaxScaler()

    elif args.preprocessing == 'StandardScaler':
        scaler = StandardScaler()

    numeric_pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=args.Degree, include_bias=False)),
        ("scaler", scaler)
    ])

    clf = ColumnTransformer([
        ('cat',OneHotEncoder(handle_unknown='ignore'),categorical_features),
        ('num',numeric_pipeline,numeric_features)
        ], 
        remainder='passthrough')

    pipeline = Pipeline(steps=[
        ('clf', clf),
        ('model', model)
    ])

    if args.Random_Search:
        print("Starting Randomized Search...")
        search = RandomizedSearchCV(pipeline, grid, scoring='r2', n_iter=15, cv = 3,random_state=args.RandomSeed)
    else:
        print("Starting Bayesian Search...")
        search = BayesSearchCV(pipeline, grid, scoring='r2', n_iter=15, random_state=args.RandomSeed)
    
    search.fit(X_t, y_t)
    
    print("Best Parameters:", search.best_params_)
    
    return search.best_estimator_,search.best_params_


# Example Usage
# if __name__ == '__main__':
#     print("Generating random dummy data...")
#     X_t = pd.DataFrame({
#         'num_feature_1': np.random.rand(50),
#         'num_feature_2': np.random.rand(50) * 10,
#         'cat_feature_1': np.random.choice(['Category_A', 'Category_B'], 50)
#     })

#     y_t = np.random.rand(50) * 100

#     print("Setting up SVM model and parameter grid...")
#     svm_model = SVR()
    
#     # Define the appropriate grid structure based on the search type
#     if args.Random_Search:
#         svm_grid = {
#             'model__C': loguniform(0.01, 100.0), 
#             'model__kernel': ['linear', 'rbf'],    
#         }
#     else:
#         svm_grid = {
#             'model__C': Real(0.01, 100.0, prior='log-uniform'), 
#             'model__kernel': Categorical(['linear', 'rbf']),    
#         }

#     print("Passing to the train function...\n")
#     final_model, params = train(X_t, y_t, model=svm_model, grid=svm_grid)

#     # Ready to use in predicting (Fixed scoring bug)
#     print("\nModel R2 Score:", final_model.score(X_t, y_t))
    
#     if args.save_model_config:
#         # CLEAN THE PARAMS: Convert NumPy types to native Python types
#         clean_params = {
#             key: (value.item() if isinstance(value, np.generic) else value) 
#             for key, value in params.items()
#         }
#         save_model_config(clean_params, 'SVM', args.Config_path)