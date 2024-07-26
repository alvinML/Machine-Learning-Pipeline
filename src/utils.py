
###############################################################################################
#1 Initialise the libraries
###############################################################################################

import sqlite3
import pandas as pd
import yaml
import logging

###############################################################################################
# Function Library #
###############################################################################################

def create_main_logger():
	"""
	Creates and configures the main logger.

	This function sets pandas options for displaying all rows and columns,
	configures the logging settings, and returns the logger instance.

	Returns:
		logger (logging.Logger): The logger instance.
	"""
	# Set pandas options for displaying all rows and columns
	pd.set_option('display.max_rows', None)
	pd.set_option('display.max_columns', None)
	
	# Configure the logging settings
	logging.basicConfig(
		filename='main.log',
		level=logging.INFO,
		format='%(asctime)s:%(levelname)s:%(message)s',
		datefmt='%Y-%m-%d %H:%M:%S'
	)
	
	# Get the logger instance
	logger = logging.getLogger()
	
	# Return the logger
	return logger



def load_config(path):
	"""
	Load the configuration file.

	Parameters:
	- path (str): The path to the configuration file.

	Returns:
	- config (dict): The configuration settings.
	"""
	# Load the configuration file
	with open(path, 'r') as file:
		config = yaml.safe_load(file)

	# Return the configuration settings
	return config





def load_df(config):
	"""
	Load data from SQLite databases into pandas DataFrames.

	Parameters:
	- config (dict): A dictionary containing the paths to the SQLite databases.

	Returns:
	- df1 (pandas.DataFrame): A DataFrame containing the data from the 'weather' table.
	- df2 (pandas.DataFrame): A DataFrame containing the data from the 'air_quality' table.
	"""
	# Connect to the SQLite database using the path specified in the config file
	conn1 = sqlite3.connect(config["data"]["path1"])
	# Read the weather table into a pandas DataFrame
	df1 = pd.read_sql_query('SELECT * FROM weather',conn1)
	# Close the database connection
	conn1.close()

	# Connect to the SQLite database using the path specified in the config file
	conn2 = sqlite3.connect(config['data']['path2'])
	# Read the air_quality table into a pandas DataFrame
	df2 = pd.read_sql_query('SELECT * FROM air_quality',conn2)
	# Close the database connection
	conn2.close()

	# Return the DataFrames
	return df1, df2


def printunique(df):
	"""
	Print the unique length and values of each column in a DataFrame.

	Parameters:
		df (pandas.DataFrame): The DataFrame to analyze.

	Returns:
		None
	"""
	# Print the unique length of each column
	for column in df.columns:
		length = len(df[column].unique())
		print(f"Unique length of '{column}' : {length} \n")

	# Print the unique values in each column
	for column in df.columns:
		unique_values = df[column].unique()
		print(f"Unique values in '{column}' : {unique_values} \n")
 

###############################################################################################
