import os
import joblib
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

parser = argparse.ArgumentParser(description='Model_Training')

#Models args
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
parser.add_argument('--bypass_search', type=bool, default=False,
                    help='Set to True to skip hyperparameter tuning and train directly')

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
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', numeric_pipeline, numeric_features)
    ],
        remainder='passthrough')

    pipeline = Pipeline(steps=[
        ('clf', clf),
        ('model', model)
    ])


    if args.bypass_search:
        print(f"Bypassing search. Training {model.__class__.__name__} directly with base parameters")
        pipeline.fit(X_t, y_t)
        # Return the fitted pipeline directly, and an empty dict for params
        return pipeline, {}

    elif args.Random_Search:
        print("Starting Randomized Search...")
        search = RandomizedSearchCV(pipeline, grid, scoring='roc_auc', n_iter=15, cv=3, random_state=args.RandomSeed)
    else:
        print("Starting Bayesian Search...")
        search = BayesSearchCV(pipeline, grid, scoring='roc_auc', n_iter=15, random_state=args.RandomSeed)

    search.fit(X_t, y_t)

    print("Best Parameters:", search.best_params_)
    return search.best_estimator_, search.best_params_


if __name__ == '__main__':

    data = np.load('Data/extracted_features_dog_color_glcm_lbp_lbglcm_glrlm_sfta.npz')
    X = pd.DataFrame(data['X'])
    Y = data['Y']


    # stratify=Y ensures the percentage of cancer/normal is the same in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=args.test_size, random_state=args.RandomSeed, stratify=Y
    )
    print(f"Training Data: {X_train.shape[0]} samples")
    print(f"Testing Data:  {X_test.shape[0]} samples")

    # 1. Test SVM
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


    # 2. Test XGBoost
    xgb_model = XGBRegressor(random_state=args.RandomSeed)
    if args.Random_Search:
        xgb_grid = {
            'model__n_estimators': randint(50, 200),
            'model__max_depth': randint(3, 10),
            'model__learning_rate': loguniform(0.01, 0.3),
            'model__subsample': [0.6, 0.8, 1.0]
        }
    else:
        xgb_grid = {
            'model__n_estimators': Integer(50, 200),
            'model__max_depth': Integer(3, 10),
            'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'model__subsample': Categorical([0.6, 0.8, 1.0])
        }

    final_xgb_model, xgb_params = train(X, Y, model=xgb_model, grid=xgb_grid)
    print("\nXGBoost Final R2 Score:", final_xgb_model.score(X, Y))


    # 3. Test Random Forest
    print("\n--- Testing Random Forest ---")
    rf_model = RandomForestClassifier(random_state=args.RandomSeed, n_jobs=-1)

    if args.Random_Search:
        rf_grid = {
            'model__n_estimators': randint(50, 300),
            'model__max_depth': randint(5, 30),
            'model__min_samples_split': randint(2, 10),
            'model__min_samples_leaf': randint(1, 10)
        }
    else:
        rf_grid = {
            'model__n_estimators': Integer(50, 300),
            'model__max_depth': Integer(5, 30),
            'model__min_samples_split': Integer(2, 10),
            'model__min_samples_leaf': Integer(1, 10)
        }


    # Pass ONLY the training data (X_train, y_train) to the model
    final_rf_model, rf_params = train(X_train, y_train, model=svm_model, grid=svm_grid)

    # Evaluate on the Hidden Test Data
    print("\n--- Evaluating Model on Unseen Test Data ---")

    # .predict() gives absolute classes (0 or 1) for Accuracy
    test_preds = final_rf_model.predict(X_test)
    test_probs = final_rf_model.predict_proba(X_test)[:, 1]

    print(f"Test Accuracy: {accuracy_score(y_test, test_preds):.4f}")
    print(f"Test ROC AUC:  {roc_auc_score(y_test, test_probs):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, test_preds))


    if args.save_model_config:
        print("\n--- Saving Configurations ---")
        # Clean SVM params
        clean_svm_params = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in svm_params.items()
        }
        # Clean XGB params
        clean_xgb_params = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in xgb_params.items()
        }
        # Clean RF params
        clean_rf_params = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in rf_params.items()
        }

        save_model_config(clean_rf_params, 'Random_Forest', args.Config_path)
        # save_model_config(clean_svm_params, 'SVM', args.Config_path)
        # save_model_config(clean_xgb_params, 'XGB_Boost', args.Config_path)

    # --- SAVE MODELS ---
    if args.save_model:

        print("\n--- Saving Models ---")
        # Ensure the specific directory for the model path exists
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

        # Format the model paths so both models have unique names based on args.model_path
        svm_model_path = args.model_path.replace('.pkl', '_svm.pkl')
        xgb_model_path = args.model_path.replace('.pkl', '_xgb.pkl')
        rf_model_path = args.model_path.replace('.pkl', '_rf.pkl')

        # # Save SVM
        # joblib.dump(final_svm_model, svm_model_path)
        # print(f"SVM model successfully saved to: {svm_model_path}")
        #
        # # Save XGBoost
        # joblib.dump(final_xgb_model, xgb_model_path)
        # print(f"XGBoost model successfully saved to: {xgb_model_path}")

        # save Random Forest
        joblib.dump(final_rf_model, rf_model_path)
        print(f"Random Forest model successfully saved to: {rf_model_path}")