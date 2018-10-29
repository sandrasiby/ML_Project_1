from proj1_helpers import *
from implementations import *

def run_model():

	# ******************************************* INPUT PARAMS *************************************************************
	training_data_path = "train.csv"
	test_data_path = "test.csv"
	output_path = "output.csv"
	sd_limit_0, sd_limit_1, sd_limit_2, sd_limit_3 = 3.0, 2.75, 2.75, 2.6 # SD above which outliers will be removed for each of the 4 jets
	list_sd_limit = [sd_limit_0,sd_limit_1,sd_limit_2,sd_limit_3,sd_limit_0,sd_limit_1,sd_limit_2,sd_limit_3]
	max_iters, stepsize, lambda_ = 20000, 1e-01, 1# Iterations for logistic regression
	
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
		
	# Loop through jet numbers (with and without valid DER_MASS_MMC
	for i in range(8):
		# Get current training, test data and SD limit
		training_tx_i, training_y_i, training_ids_i = list_training_tx[i], list_training_y[i], list_training_ids[i]
		test_tx_i, test_y_i, test_ids_i = list_test_tx[i], list_test_y[i], list_test_ids[i]
		sd_limit = list_sd_limit[i]
		
		# ************************* TRAINING FOR CURRENT JET NUMBER AND DER_MASS_MMC *********************************************		
		#Remove outliers and standardize the training data
		training_tx_i, training_y_i, training_tx_i_mean, training_tx_i_std = standardize_training(training_tx_i, training_y_i,sd_limit)
				
		#Create polynomial expansions for the training data
		training_tx_i = get_polynomial(training_tx_i, 2)
	
		#Get weights
		initial_w = np.zeros((training_tx_i.shape[1], ))
		weights_i, loss_i = reg_logistic_regression_newton(training_y_i, training_tx_i,lambda_, initial_w, max_iters, stepsize)
		# weights_i, loss_i = stochastic_gradient_descent(training_y_i, training_tx_i, initial_w, 1, max_iters, stepsize)
		print('weights are',weights_i)
			
		# ************************* PREDICTION FOR CURRENT JET NUMBER AND DER_MASS_MMC *********************************************		
		#Standardize the test data using training mean and std
		test_tx_i = standardize_test(test_tx_i,training_tx_i_mean, training_tx_i_std)
	
		#Create polynomial expansion for the test data
		test_tx_i = get_polynomial(test_tx_i, 2)
		
		#Get predictions for all the jets 
		y_pred_i = predict_labels(weights_i, test_tx_i,1) # 1 to indicate that we are using logistic regression and not linear

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

	run_model()

	
