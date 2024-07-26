
###############################################################################################
#1 Initialise the libraries
###############################################################################################

import pandas as pd
import numpy as np


###############################################################################################
#2. Data Preprocessing Steps #1 to #15
###############################################################################################
def data_processing(weather_df, air_quality_df):


	#########################################################################################
	# 1 Drop duplicates in both dataframes
	weather_df = drop_duplicates(weather_df, 'data_ref')
	air_quality_df = drop_duplicates(air_quality_df, 'data_ref')

	#########################################################################################
	# 2 Merge the two dataframes on 'date' and 'data_ref'
	merged_df = merge_dataframes(weather_df, air_quality_df, ['date', 'data_ref'])
	
	#########################################################################################
	# 3 Drop 'data_ref' column
	merged_df = drop_column(merged_df, 'data_ref')

	#########################################################################################
	# 4 Replace missing values with None
	merged_df = replace_with_none(merged_df, '-')
	merged_df = replace_with_none(merged_df, '--')

	#########################################################################################
	# 5 Convert columns to numeric where possible
	merged_df = datatype_to_float(merged_df)

	#########################################################################################        
	# 6 Fill in relevant rainfall columns with 0.
	merged_df = rainfall_zero_relationship(merged_df)

	#########################################################################################
	# 7 Absolute value of 'Max Wind Speed (km/h)'
	merged_df = absolute_value(merged_df, 'Max Wind Speed (km/h)')

	#########################################################################################
	# 8 Absolute value of 'Wet Bulb Temperature (deg F)'
	merged_df = absolute_value(merged_df, 'Wet Bulb Temperature (deg F)')

	#########################################################################################
	# 9 Convert 'Wet Bulb Temperature (deg F)' to 'Wet Bulb Temperature (deg C)' 
	# and drop the original column
	merged_df = convert_wet_bulb_temp_f_to_c(merged_df, 'Wet Bulb Temperature (deg F)', 
											 'Wet Bulb Temperature (deg C)')

	#########################################################################################
	# 10 Map 'Dew Point Category' to numerical values
	merged_df = map_dew_point_category(merged_df, 'Dew Point Category')

	#########################################################################################
	# 11 Map 'Wind Direction' to standardized values
	merged_df = map_wind_direction(merged_df, 'Wind Direction')

	#########################################################################################
	# 12 Map 'Daily Solar Panel Efficiency' to numerical values
	merged_df = map_solar_panel_efficiency(merged_df, 'Daily Solar Panel Efficiency')

	#########################################################################################
	# 13 Sine and cosine transformation of 'Date'
	merged_df = transform_date(merged_df)

	#########################################################################################
	# 14 Sine and cosine transformation of 'Wind Direction'
	merged_df = transform_winddir(merged_df)

	#########################################################################################
	# 15 Missing data processing section
	merged_df = interpolate_missing(merged_df)

	#########################################################################################

	return merged_df

###############################################################################################


###############################################################################################
# Function Library #
###############################################################################################


def drop_duplicates(df, subset, keep='first'):
	"""
	Remove duplicate rows from a DataFrame.

	Parameters:
	- df (DataFrame): The input DataFrame.
	- subset (str or list): Column name(s) to consider for identifying duplicates.
	- keep (str, default 'first'): Specifies which duplicates (if any) to keep. 
		- 'first': Keep the first occurrence of each duplicated row.
		- 'last': Keep the last occurrence of each duplicated row.
		- False: Remove all occurrences of duplicated rows.

	Returns:
	- DataFrame: A DataFrame with duplicate rows removed.
	"""
	return df.drop_duplicates(subset=[subset], keep=keep)


def merge_dataframes(df1, df2, col, merge_type='inner'):
	"""
	Merge two dataframes based on specified columns.

	Parameters:
	- df1 (pandas.DataFrame): The first dataframe to be merged.
	- df2 (pandas.DataFrame): The second dataframe to be merged.
	- col (str or list): The column(s) to merge on.
	- merge_type (str, default 'inner'): The type of merge to perform.
		- 'inner': Only include common rows between the two dataframes.
		- 'left': Include all rows from the left dataframe and matching rows from the right dataframe.
		- 'right': Include all rows from the right dataframe and matching rows from the left dataframe.
		- 'outer': Include all rows from both dataframes.

	Returns:
	- pandas.DataFrame: The merged dataframe.
	"""
	return pd.merge(df1, df2, on=col, how=merge_type)


def drop_column(df, col, axis=1):
	"""
	Drop a column from a DataFrame.

	Parameters:
	- df (pandas.DataFrame): The input DataFrame.
	- col (str): The name of the column to drop.
	- axis (int, default 1): The axis along which to drop the column.
		- 0: Drop the column from the index (rows).
		- 1: Drop the column from the columns.

	Returns:
	- pandas.DataFrame: A DataFrame with the specified column dropped.
	"""
	return df.drop(col, axis=axis)


def replace_with_none(df, string_to_replace):
	"""
	Replace occurrences of a specific string with None in a DataFrame.

	Parameters:
	- df (pandas.DataFrame): The input DataFrame.
	- string_to_replace (str): The string to be replaced.

	Returns:
	- pandas.DataFrame: A DataFrame with the specified string replaced with None.
	"""
	return df.replace(string_to_replace, None)


def datatype_to_float(df):
	"""
	Convert the data types of columns in a DataFrame to float.

	Parameters:
	- df (pandas.DataFrame): The DataFrame to be processed.

	Returns:
	- pandas.DataFrame: The DataFrame with converted data types.
	"""
	for col in df:
		try:
			df[col] = pd.to_numeric(df[col]).astype('float64')
		except ValueError:
			continue
	return df


def rainfall_zero_relationship_map(row):
	"""
	Apply specific transformations to a row based on rainfall relationships.
	Daily Rainfall Total >= Highest 120 Min Rainfall >= Highest 60 Min Rainfall >= Highest 30 Min Rainfall

	Parameters:
	- row (pd.Series): A row of the dataframe.

	Returns:
	- pd.Series: The transformed row.
	"""
	if row['Daily Rainfall Total (mm)'] == 0:
		row['Highest 30 Min Rainfall (mm)'] = 0
		row['Highest 60 Min Rainfall (mm)'] = 0
		row['Highest 120 Min Rainfall (mm)'] = 0
	if row['Highest 30 Min Rainfall (mm)'] == 0:
		row['Highest 60 Min Rainfall (mm)'] = 0
		row['Highest 120 Min Rainfall (mm)'] = 0
	if row['Highest 60 Min Rainfall (mm)'] == 0:
		row['Highest 120 Min Rainfall (mm)'] = 0
	if row['Highest 30 Min Rainfall (mm)'] == 0:
		row['Daily Rainfall Total (mm)'] = 0
		row['Highest 60 Min Rainfall (mm)'] = 0
		row['Highest 120 Min Rainfall (mm)'] = 0
	if row['Highest 30 Min Rainfall (mm)'] == row['Highest 120 Min Rainfall (mm)']:
		row['Highest 60 Min Rainfall (mm)'] = row['Highest 120 Min Rainfall (mm)']
	if row['Highest 60 Min Rainfall (mm)'] == row['Daily Rainfall Total (mm)']:
		row['Highest 120 Min Rainfall (mm)'] = row['Daily Rainfall Total (mm)']
	return row


def rainfall_zero_relationship(df):
	"""
	Apply the rainfall_zero_relationship_map function to each row of the dataframe.

	Parameters:
	- df (pd.DataFrame): The dataframe to process.

	Returns:
	- pd.DataFrame: The transformed dataframe.
	"""
	return df.apply(rainfall_zero_relationship_map, axis=1)


def absolute_value(df, col):
	"""
	Convert the specified column of the DataFrame to absolute values.

	Parameters:
	- df (pandas.DataFrame): The input DataFrame.
	- col (str): The name of the column to compute absolute values for.

	Returns:
	- pandas.DataFrame: The DataFrame with the absolute values of the specified column.
	"""
	df[col] = df[col].abs()
	return df


def convert_wet_bulb_temp_f_to_c(df, old_name, new_name):
	"""
	Convert Wet Bulb Temperature from Fahrenheit to Celsius and add it as a new column.

	Parameters:
	df (pd.DataFrame): The dataframe containing the temperature column in Fahrenheit.
	column_f (str): The name of the column containing temperatures in Fahrenheit.
	column_c (str): The name of the new column to store temperatures in Celsius.

	Returns:
	pd.DataFrame: The dataframe with the new temperature column in Celsius.
	"""
	df[new_name] = ((df.pop(old_name) - 32) * 5/9).round(1)
	return df


dewpointmap = {
	'VH': 5,
	'Very High': 5,
	'Low': 2,
	'High': 5,
	'Moderate': 3,
	'Extreme': 5,
	'Very Low': 1,
	'very low': 1,
	'LOW': 2,
	'VERY HIGH': 5,
	'High Level': 4,
	'very high': 5,
	'HIGH': 4,
	'H': 4,
	'M': 3,
	'moderate': 3,
	'VL': 1,
	'MODERATE': 3,
	'high': 4,
	'Below Average': 2,
	'VERY LOW': 1,
	'Minimal': 1,
	'low': 2,
	'Normal': 3,
	'L': 2
}


def map_dew_point_category(df, column):
	"""
	Map categorical 'Dew Point Category' to numerical values.

	Parameters:
	df (pd.DataFrame): The dataframe containing the 'Dew Point Category' column.
	column (str): The name of the column to map.
	mapping (dict): The dictionary containing the mapping of categorical values to numerical values.

	Returns:
	pd.DataFrame: The dataframe with the mapped 'Dew Point Category' column.
	"""
	df[column] = df[column].map(dewpointmap)
	return df


wind_direction_letters_map = {
	'W': 'W',
	'S': 'S',
	'E': 'E',
	'east': 'E',
	'NORTHEAST': 'NE',
	'NW': 'NW',
	'NE': 'NE',
	'SE': 'SE',
	'Southward': 'S',
	'W.': 'W',
	'southeast': 'SE',
	'SW': 'SW',
	'N': 'N',
	'Northward': 'N',
	'SOUTHEAST': 'SE',
	'northwest': 'NW',
	'west': 'W',
	'NORTH': 'N',
	'south': 'S',
	'NE.': 'NE',
	'SE.': 'SE',
	'NORTHWEST': 'NW',
	'northeast': 'NE',
	'SW.': 'SW',
	'north': 'N',
	'SOUTH': 'S',
	'E.': 'E',
	'S.': 'S',
	'NW.': 'NW',
	'WEST': 'W',
	'N.': 'N',
	'EAST': 'E'
}


def map_wind_direction(df, column):
	"""
	Map categorical 'Wind Direction' to standardized values.

	Parameters:
	df (pd.DataFrame): The dataframe containing the 'Wind Direction' column.
	column (str): The name of the column to map.
	mapping (dict): The dictionary containing the mapping of categorical values to standardized values.

	Returns:
	pd.DataFrame: The dataframe with the mapped 'Wind Direction' column.
	"""
	df[column] = df[column].map(wind_direction_letters_map)
	return df


efficiency_map = {
	'Low': 0,
	'Medium': 1,
	'High': 2
}

def map_solar_panel_efficiency(df, column):
	"""
	Map categorical 'Daily Solar Panel Efficiency' to numerical values.

	Parameters:
	- df (pd.DataFrame): The dataframe containing the 'Daily Solar Panel Efficiency' column.
	- column (str): The name of the column to map.
	- mapping (dict): The dictionary containing the mapping of categorical values to numerical values.

	Returns:
	- pd.DataFrame: The dataframe with the mapped 'Daily Solar Panel Efficiency' column.
	"""
	df[column] = df[column].map(efficiency_map)
	return df


def is_leap_year(year):
	"""
	Check if a year is a leap year.

	Args:
	- year (int): The year to be checked.

	Returns:
	- bool: True if the year is a leap year, False otherwise.
	"""
	return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def transform_date(df):
	"""
	Transforms the date column in the given DataFrame by converting it to a datetime object,
	sorting the DataFrame by date, and extracting additional features related to the date.

	Args:
	- df (pandas.DataFrame): The DataFrame containing the date column.

	Returns:
	- pandas.DataFrame: The transformed DataFrame with additional date-related features.
	"""
	# Convert date column to datetime object
	df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
	
	# Ensure that dataframe is sorted by date.
	df.sort_values(by='date', inplace=True)

	# Extract day of year information from date
	df['day_of_year'] = df['date'].dt.dayofyear

	# Extract year information from date
	df['year'] = df['date'].dt.year

	# Calculate number of days in a year
	df['total_days'] = df['year'].apply(lambda x:366 if is_leap_year(x) else 365)

	# Create sine and cosine transformation features
	df['day_sin'] = (np.sin(2 * np.pi * df['day_of_year'] / df['total_days'])).round(5)
	df['day_cos'] = (np.cos(2 * np.pi * df['day_of_year'] / df['total_days'])).round(5)

	# Drop intermediate columns
	df.drop(['date', 'day_of_year', 'year', 'total_days'], axis=1, inplace=True)
	return df


wind_direction_angles_map = {
	'N': 0,
	'NE': 45,
	'E': 90,
	'SE': 135,
	'S': 180,
	'SW': 225,
	'W': 270,
	'NW': 315
}

def transform_winddir(df):
	"""
	Transforms the wind direction column in the given DataFrame by converting it to angles,
	applying sine and cosine transformations, and dropping the original wind direction columns.

	Args:
		df (pandas.DataFrame): The DataFrame containing the wind direction column.

	Returns:
		pandas.DataFrame: The transformed DataFrame with wind direction columns replaced by
		wind direction sine and cosine columns.
	"""
	# Convert wind direction to angles
	df['wind_dir_angle'] = df['Wind Direction'].map(wind_direction_angles_map)

	# Convert angles to radians for sine and cosine transformations
	df['wind_dir_angle_rad'] = np.deg2rad(df['wind_dir_angle'])

	# Apply sine and cosine transformations
	df['wind_dir_sin'] = (np.sin(df['wind_dir_angle_rad'])).round(5)
	df['wind_dir_cos'] = (np.cos(df['wind_dir_angle_rad'])).round(5)

	# Drop the original wind direction columns
	df.drop(['Wind Direction', 'wind_dir_angle', 'wind_dir_angle_rad'], axis=1, inplace=True)

	return df


def interpolate_missing(df):
	"""
	Interpolates missing values in the given DataFrame.

	Parameters:
		df (pandas.DataFrame): The DataFrame containing the data to be processed.

	Returns:
		pandas.DataFrame: The DataFrame with missing values interpolated.

	"""
	# Drop Sunshine Duration and Cloud Cover (%) null values
	df.dropna(subset=['Sunshine Duration (hrs)', 'Cloud Cover (%)'], inplace=True)

	# Interpolate continuous data
	weather_cols_to_interpolate = [
		'Min Temperature (deg C)', 'Maximum Temperature (deg C)',
		'Min Wind Speed (km/h)', 'Max Wind Speed (km/h)'
	]

	df[weather_cols_to_interpolate] = df[weather_cols_to_interpolate].interpolate(method='linear')

	# Impute rainfall data using domain-specific rules
	for idx, row in df.iterrows():
		if pd.isna(row['Daily Rainfall Total (mm)']):
			df.at[idx, 'Daily Rainfall Total (mm)'] = row['Highest 120 Min Rainfall (mm)']
		if pd.isna(row['Highest 120 Min Rainfall (mm)']):
			df.at[idx, 'Highest 120 Min Rainfall (mm)'] = row['Daily Rainfall Total (mm)']
		if pd.isna(row['Highest 60 Min Rainfall (mm)']):
			df.at[idx, 'Highest 60 Min Rainfall (mm)'] = row['Highest 120 Min Rainfall (mm)']
		if pd.isna(row['Highest 30 Min Rainfall (mm)']):
			df.at[idx, 'Highest 30 Min Rainfall (mm)'] = row['Highest 60 Min Rainfall (mm)']

	# Interpolate continuous data
	air_quality_cols = ['pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 'pm25_central',
						'psi_north', 'psi_south', 'psi_east', 'psi_west', 'psi_central']

	df[air_quality_cols] = df[air_quality_cols].interpolate(method='linear')

	# Drop the remaining null values that can't be interpolated
	df.dropna(inplace=True)
	return df

 

###############################################################################################
