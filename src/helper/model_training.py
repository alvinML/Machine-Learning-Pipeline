from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import autogluon.tabular as ag
import pandas as pd
import logging

#######################################################################################
def split_scale(X, y, seed):
	"""
	Split the data into train and test sets and scale the features.

	Args:
		X (pd.DataFrame): The input dataframe with features.
		y (pd.Series): The target variable.
		seed (int): Random seed for reproducibility.

	Returns:
		X_train_scaled (np.ndarray): Scaled training features.
		X_test_scaled (np.ndarray): Scaled testing features.
		y_train (pd.Series): Training target variable.
		y_test (pd.Series): Testing target variable.
	"""
	# Split data into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y)

	# Standardize the features
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	return X_train_scaled, X_test_scaled, y_train, y_test


#######################################################################################
#LOGISTIC REGRESSION MODEL
def model1(X, y, params, seed):
	"""
	Train and evaluate a logistic regression model.

	Args:
		X (pd.DataFrame): The input dataframe with features.
		y (pd.Series): The target variable.
		params (dict): Parameters for LogisticRegression.
		seed (int): Random seed for reproducibility.

	Returns:
		None
	"""
	logging.info("Loading Logistic Regression Model...")

	try:
		# Data preparation
		X_train_scaled, X_test_scaled, y_train, y_test = split_scale(X, y, seed)

		# Initialize and train logistic regression model
		model = LogisticRegression(**params, random_state=seed)
		model.fit(X_train_scaled, y_train)

		# Make predictions
		predictions_train = model.predict(X_train_scaled)
		predictions_test = model.predict(X_test_scaled)

		# Get feature coefficients
		feature_names = X.columns
		coefficients = model.coef_[0]
		weights = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
		weights = weights.sort_values(by='Coefficient')
	
		logging.info(f"Number of iterations: {model.n_iter_}")
		logging.info(f"Weights:\n{weights}")

		# Calculate accuracy scores
		accuracy_train = accuracy_score(predictions_train, y_train)
		accuracy_cv = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
		accuracy_test= accuracy_score(predictions_test, y_test)

		logging.info(f"Train accuracy score: {accuracy_train*100:.3g}%")
		logging.info(f"CV accuracy score: {accuracy_cv*100:.3g}%")
		logging.info(f"Test accuracy score: {accuracy_test*100:.3g}%\n")

	except Exception as e:
		logging.error(f"An error occurred: {e}\n")
	return None


#######################################################################################
# XGBOOST MODEL
def model2(X, y, params, seed, cv):
	"""
	Trains an XGBoost model using grid search cross-validation.

	Args:
		X (pd.DataFrame): The input dataframe containing the features.
		y (pd.Series): The target variable.
		params (dict): A dictionary of hyperparameters and their potential values for tuning.
		seed (int): The random seed for reproducibility.
		cv (int): The number of cross-validation folds.

	Returns:
		None
	"""
	logging.info("Loading XGBoost Model with Grid Search CV...")

	try:
		# Data preparation
		X_train_scaled, X_test_scaled, y_train, y_test = split_scale(X, y, seed)

		# Create XGBClassifier model
		model = XGBClassifier(random_state=seed)

		# Define hyperparameters and their potential values for tuning
		param_grid = {**params}

		# Create GridSearchCV object
		grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)

		# Fit to training data
		grid_search.fit(X_train_scaled, y_train)

		# Get the best model and best parameters
		best_model = grid_search.best_estimator_
		best_params = grid_search.best_params_

		# Make predictions
		predictions_train = best_model.predict(X_train_scaled)
		predictions_test = best_model.predict(X_test_scaled)

		# Calculate accuracy scores
		accuracy_train = accuracy_score(predictions_train, y_train)
		accuracy_cv = grid_search.best_score_
		accuracy_test = accuracy_score(predictions_test, y_test)

		logging.info(f"Best parameters: {best_params}")
		logging.info(f"Train accuracy score: {accuracy_train*100:.3g}%")
		logging.info(f"CV accuracy score: {accuracy_cv*100:.3g}%")
		logging.info(f"Test accuracy score: {accuracy_test*100:.3g}%\n")

	except Exception as e:
		logging.error(f"An error occurred: {e}")

	return None


#######################################################################################
#AUTOGLUON MODEL
def model3(df, params, seed, label):
	"""
	Train and evaluate an Autogluon model.

	Args:
		df (pd.DataFrame): The input dataframe containing the features and target variable.
		params (dict): Parameters for Autogluon model.
		seed (int): Random seed for reproducibility.
		label (str): The name of the target variable column.

	Returns:
		None
	"""
	logging.info("Loading Autogluon...")

	try:
		# Split data into train and test sets
		train_data, test_data = train_test_split(df, random_state=seed, stratify=df[label])
		X_train = train_data.drop(columns=[label])
		y_train = train_data[label]
		X_test = test_data.drop(columns=[label])
		y_test = test_data[label]

		# Initialize and train Autogluon model
		model = ag.TabularPredictor(label=label)
		model.fit(train_data, **params)

		# Get the best model
		leaderboard = model.leaderboard(silent=True)
		best_model = leaderboard.iloc[0]

		# Make predictions
		predictions_train = model.predict(X_train)
		predictions_test = model.predict(X_test)

		# Calculate accuracy scores
		accuracy_train = accuracy_score(predictions_train, y_train)
		accuracy_cv = best_model['score_val']
		accuracy_test = accuracy_score(predictions_test, y_test)

		logging.info(model.leaderboard())
		logging.info(f"Train accuracy score: {accuracy_train*100:.3g}%")
		logging.info(f"CV accuracy score: {accuracy_cv*100:.3g}%")
		logging.info(f"Test accuracy score: {accuracy_test*100:.3g}%\n")

	except Exception as e:
		logging.error(f"An error occurred: {e}")	

	return None

