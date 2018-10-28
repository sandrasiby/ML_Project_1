from proj1_helpers_1 import *
from implementations import *
from logistic_regression import *

def exp_three_models():

	# ******************************************* INPUT PARAMS *************************************************************
	training_data_path = "train.csv"
	test_data_path = "test.csv"
	output_path = "output.csv"
	sd_limit_0, sd_limit_1, sd_limit_2, sd_limit_3 = 2.5, 2.75, 2.75, 2.6
	# list_sd_limit = [sd_limit_0,sd_limit_1,sd_limit_2,sd_limit_3]
	list_sd_limit = [sd_limit_0,sd_limit_1,sd_limit_2,sd_limit_3,sd_limit_0,sd_limit_1,sd_limit_2,sd_limit_3]
	maxiter, stepsize, lambda_, is_newton = 20000, 1e-01, 1, 1
	
	# ******************************************* READ DATA *************************************************************
	#Read training data
	training_y, training_tx, training_ids = load_csv_data(training_data_path, sub_sample=False)
	
	# Read test data
	test_y, test_tx, test_ids = load_csv_data(test_data_path)
	
	# ******************************************* SPLIT DATA *************************************************************
	#Split training data into different jets
	list_training_tx, list_training_y, list_training_ids = split_data_by_jet_num_feature(training_tx, training_y, training_ids)
	
	# Split test data into various jet numbers
	list_test_tx, list_test_y, list_test_ids = split_data_by_jet_num_feature(test_tx, test_y, test_ids)
	list_weights = []
	
	# Loop through jet numbers
	for i in range(8):
		training_tx_i, training_y_i, training_ids_i = list_training_tx[i], list_training_y[i], list_training_ids[i]
		test_tx_i, test_y_i, test_ids_i = list_test_tx[i], list_test_y[i], list_test_ids[i]
		sd_limit = list_sd_limit[i]
		
		# ******************************************* TRAINING *************************************************************		
		#Standardize the training data
		training_tx_i, training_tx_i_mean, training_tx_i_std = standardize_training(training_tx_i)
				
		#Remove outliers and -999 from the standardized training dataset
		training_tx_i, training_y_i, training_tx_i_out, training_y_i_out = remove_outliers(training_tx_i,training_y_i,sd_limit)

		#Redo standardization
		training_tx_i, training_tx_i_mean, training_tx_i_std = redo_standardization(training_tx_i, training_tx_i_mean, training_tx_i_std)
		
		#Create polynomial expansions
		training_tx_i = get_polynomial(training_tx_i, 2)
	
		#Get weights
		weights_i = logistic_regression(training_y_i, training_tx_i, is_newton, stepsize, maxiter, lambda_)
		# list_weights.append(weights_i)
	
		# ******************************************* PREDICTION *************************************************************
		#Standardize the test data using training mean and std
		test_tx_i = standardize_test(test_tx_i,training_tx_i_mean, training_tx_i_std)
	
		#Create polynomial expansion for the test data
		test_tx_i = get_polynomial(test_tx_i, 2)
		
		#Get predictions for all the jets 
		y_pred_i = predict_labels(weights_i, test_tx_i)

		#Collate data for all jets
		if(i == 0):
			y_pred_all = y_pred_i
			test_ids_all = test_ids_i
		else:
			y_pred_all = np.concatenate((y_pred_all,y_pred_i))
			test_ids_all = np.concatenate((test_ids_all,test_ids_i))
	
	# ******************************************* OUTPUT *************************************************************
	zipped_list = sorted(zip(test_ids_all, y_pred_all))
	test_ids_all, y_pred_all = zip(*zipped_list)
	test_ids_all = np.array(test_ids_all)
	print('length of test ids', len(test_ids_all))
	print('length of test ys', len(y_pred_all))
	y_pred_all = np.array(y_pred_all)
	
	# Create output file
	print("Write CSV output: START")
	create_csv_submission(test_ids_all, y_pred_all, output_path)
	print("Write CSV output: DONE")
	

if __name__ == "__main__":

	exp_three_models()

	
