import os
import joblib
import argparse
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
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
parser.add_argument('--save_model',type =bool,default=True)
parser.add_argument('--Config_path',type = str, default='Config/model_config.yaml')
parser.add_argument('--model_path',type = str, default='Models/model.pkl')

args = parser.parse_args()
print(args)


def train(X_t, y_t, model, grid):
    numeric_features = (
        X_t.select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    categorical_features = (
        X_t.select_dtypes(exclude=["number"])
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


if __name__ == '__main__':


    data = np.load('Data/extracted_features_dog_color_glcm_lbp_lbglcm_glrlm_sfta.npz')
    X = pd.DataFrame(data['X'])
    Y = data['Y']

    #Test 1    
    svm_model = SVR()
    
    if args.Random_Search:
        svm_grid = {
            'model__C': loguniform(0.01, 100.0), 
            'model__gamma': ['scale', 'auto'],
            'model__kernel': ['linear', 'rbf']
        }
    else:
        svm_grid = {
            'model__C': Real(0.01, 100.0, prior='log-uniform'), 
            'model__gamma': Categorical(['scale', 'auto']),
            'model__kernel': Categorical(['linear', 'rbf'])
        }

    final_svm_model, svm_params = train(X, Y, model=svm_model, grid=svm_grid)
    print("\nSVM Final R2 Score:", final_svm_model.score(X, Y))


    # #Test 2
    # xgb_model = XGBRegressor(random_state=args.RandomSeed)
    
    # if args.Random_Search:
    #     xgb_grid = {
    #         'model__n_estimators': randint(50, 200),
    #         'model__max_depth': randint(3, 10),
    #         'model__learning_rate': loguniform(0.01, 0.3),
    #         'model__subsample': [0.6, 0.8, 1.0]
    #     }
    # else:
    #     xgb_grid = {
    #         'model__n_estimators': Integer(50, 200),
    #         'model__max_depth': Integer(3, 10),
    #         'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    #         'model__subsample': Categorical([0.6, 0.8, 1.0])
    #     }

    # final_xgb_model, xgb_params = train(X, Y, model=xgb_model, grid=xgb_grid)
    # print("\nXGBoost Final R2 Score:", final_xgb_model.score(X, Y))

    if args.save_model_config:
        print("\n--- Saving Configurations ---")
        # Clean SVM params
        clean_svm_params = {
            key: (value.item() if isinstance(value, np.generic) else value) 
            for key, value in svm_params.items()
        }
        # # Clean XGB params
        # clean_xgb_params = {
        #     key: (value.item() if isinstance(value, np.generic) else value) 
        #     for key, value in xgb_params.items()
        # }
        save_model_config(clean_svm_params, 'SVM', args.Config_path)
        # save_model_config(clean_xgb_params, 'XGB_Boost', args.Config_path)
    
    # --- SAVE MODELS ---
    if args.save_model:
        
        print("\n--- Saving Models ---")
        # Ensure the specific directory for the model path exists
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        
        # Format the model paths so both models have unique names based on args.model_path
        svm_model_path = args.model_path.replace('.pkl', '_svm.pkl')
        # xgb_model_path = args.model_path.replace('.pkl', '_xgb.pkl')
        
        # Save SVM
        joblib.dump(final_svm_model, svm_model_path)
        print(f"SVM model successfully saved to: {svm_model_path}")
        
        # Save XGBoost
        # joblib.dump(final_xgb_model, xgb_model_path)
        # print(f"XGBoost model successfully saved to: {xgb_model_path}")