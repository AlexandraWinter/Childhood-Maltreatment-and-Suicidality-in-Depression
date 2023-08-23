                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
# -------------------- GENERATED WITH PHOTON WIZARD (beta) ------------------------------
# PHOTON Project Folder: [add folder]

import pandas as pd
import numpy as np
from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Switch, Preprocessing
from photonai.optimization import Categorical, IntegerRange, FloatRange
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

# Specify how results are going to be saved
output_settings = OutputSettings(mongodb_connect_url="[add URL]",
                                 user_id="awinter",
                                 wizard_object_id="641d9ef604698536839420e7",
                                 wizard_project_name="randomforestclassification")

# Define hyperpipe
hyperpipe = Hyperpipe('randomforestSWITCH',
                      project_folder='[add folder]',
                      optimizer="grid_search",
                      optimizer_params={},
                      metrics=['balanced_accuracy', 'recall', 'specificity','f1_score'],
                      best_config_metric="balanced_accuracy",
                      outer_cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=3210),
                      inner_cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=3210),
                      output_settings=output_settings)
        
# Add transformer elements
hyperpipe += PipelineElement("SimpleImputer", hyperparameters={}, 
                             test_disabled=False, missing_values=-99, strategy='mean', fill_value=0)
hyperpipe += PipelineElement("StandardScaler", hyperparameters={}, 
                             test_disabled=False, with_mean=True, with_std=True)

estimator_switch = Switch('estimators')
estimator_switch += PipelineElement('SVC', hyperparameters={'C': [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8],
                                                            'kernel': ['rbf', 'linear']}, max_iter=1000)

estimator_switch += PipelineElement('RandomForestClassifier', hyperparameters={"max_features": ["sqrt", "log2"],
                                                                               "min_samples_leaf": [0.01, 0.1, 0.2]})

estimator_switch += PipelineElement('AdaBoostClassifier', hyperparameters={'n_estimators': [10, 25, 50]})

estimator_switch += PipelineElement('LogisticRegression',
                                            hyperparameters={"C": [1e-4, 1e-2, 1, 1e2, 1e4],
                                                             "penalty": ['l1', 'l2']},
                                            solver='saga', n_jobs=1)
estimator_switch += PipelineElement('GaussianNB')
estimator_switch += PipelineElement('KNeighborsClassifier', hyperparameters={"n_neighbors": [5, 10, 15]})
hyperpipe += estimator_switch


# Load data - adjust the column numbers
df = pd.read_excel('[add file path]')
X = np.asarray(df.iloc[:, 1:18])
y = np.asarray(df.iloc[:, 18])

# Fit hyperpipe
hyperpipe.fit(X, y)

# Extract feature importances
feature_importances_df = pd.DataFrame.from_dict({i: fold.best_config.best_config_score.feature_importances for i, fold in enumerate(hyperpipe.results.outer_folds)})
feature_importances_df.to_csv('[add file path]')
