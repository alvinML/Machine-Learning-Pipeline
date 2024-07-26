from utils import create_main_logger, load_df, load_config
from helper.data_processing import data_processing
from helper.model_training import model1, model2, model3



def main():
	# 1. Initialise the logging
	logger = create_main_logger()
	logger.info("main.py has started.")

	# 2. Load the configuration file
	config = load_config("./config/config.yaml")

	# 3. Load and process the dataframes
	weather_df, air_quality_df = load_df(config)
	df_final = data_processing(weather_df, air_quality_df)

	# 4. Load the processed DataFrame into X and y
	X = df_final.drop(config['target_feature'], axis=1)
	y = df_final[config['target_feature']]

	# 5. Train and evaluate the models
	for model_name, model_config in config['models'].items():
		if not model_config['enabled']:
			continue

		print(f"Training {model_name}...")

		match model_name:
			case "logistic_regression":
				model1(X, y, model_config['params'], config['seed'])

			case "xgboost":
				model2(X, y, model_config['params'], config['seed'], model_config['cv'])

			case "autogluon":
				model3(df_final, model_config['params'], config['seed'], config['target_feature'])

			case _:
				raise ValueError(f"Model not found: {model_name}")

	logger.info("main.py has ended.\n")


if __name__ == "__main__":
	main()








