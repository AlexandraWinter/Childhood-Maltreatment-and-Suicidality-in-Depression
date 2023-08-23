                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
# -------------------- GENERATED WITH PHOTON WIZARD (beta) ------------------------------
# PHOTON Project Folder: [add folder]

import pandas as pd
import numpy as np
from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Switch, Preprocessing
from photonai.optimization import Categorical, IntegerRange, FloatRange
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold    
             
# Specify how results are going to be saved
output_settings = OutputSettings(mongodb_connect_url="[add URL]",                                                                  
                             user_id="awinter",
                             wizard_object_id="643d3caf20c64f246633806b",
                             wizard_project_name="randomforestctq")
                    
# Define hyperpipe
hyperpipe = Hyperpipe('randomforestctqSWITCH',
                      project_folder = '[add folder]',
                      optimizer="grid_search",
                      optimizer_params={},
                      metrics=['mean_squared_error', 'mean_absolute_error', 'explained_variance', 'pearson_correlation', 'r2'],
                      best_config_metric="mean_squared_error",
                      outer_cv = KFold(n_splits=10, shuffle=True, random_state=3210),
                      inner_cv = KFold(n_splits=10, shuffle=True, random_state=3210),
                      output_settings=output_settings)
        
# Add transformer elements
hyperpipe += PipelineElement("SimpleImputer", hyperparameters={}, 
                             test_disabled=False, missing_values=-99, strategy='mean', fill_value=0)
hyperpipe += PipelineElement("StandardScaler", hyperparameters={}, 
                             test_disabled=False, with_mean=True, with_std=True)
# hyperpipe += PipelineElement("RandomForestRegressor", hyperparameters={'max_depth':IntegerRange(2, 15)}, n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_features=None)

estimator_switch = Switch('estimators')
estimator_switch += PipelineElement('SVR', hyperparameters={'C': [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8],
                                                            'kernel': ['rbf', 'linear']}, max_iter=1000)

estimator_switch += PipelineElement('RandomForestRegressor', hyperparameters={"max_features": ["sqrt", "log2"],
                                                                               "min_samples_leaf": [0.01, 0.1, 0.2]})

estimator_switch += PipelineElement('AdaBoostRegressor', hyperparameters={'n_estimators': [10, 25, 50]})

estimator_switch += PipelineElement('LinearRegression')
estimator_switch += PipelineElement('KNeighborsRegressor', hyperparameters={"n_neighbors": [5, 10, 15]})
hyperpipe += estimator_switch


# Load data - adjust column numbers
df = pd.read_excel('[add file path]')
X = np.asarray(df.iloc[:, 1:18])
y = np.asarray(df.iloc[:, 18])

# Fit hyperpipe
hyperpipe.fit(X, y)
feature_importances_df = pd.DataFrame.from_dict({i: fold.best_config.best_config_score.feature_importances for i, fold in enumerate(hyperpipe.results.outer_folds)})
feature_importances_df.to_csv('[add file path]')
