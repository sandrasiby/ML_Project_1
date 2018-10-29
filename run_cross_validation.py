from proj1_helpers import *
from implementations import *

def run_model(train_ratio, stepsize, lambda_):

	training_data_path = "train.csv"
	test_data_path = "test.csv"
	output_path = "output.csv"
	n_trials = 10
	limit = int(train_ratio * 250000)
	sd_limit_0, sd_limit_1, sd_limit_2, sd_limit_3 = 3.0, 2.75, 2.75, 2.6
	list_sd_limit = [sd_limit_0,sd_limit_1,sd_limit_2,sd_limit_3, sd_limit_0,sd_limit_1,sd_limit_2,sd_limit_3]
	list_poly = [2,2,2,2,2,2,2,2] # List of polynomials  for jets 0 to 3, for DER_mass_MMC valid and invalid
	weights_average = np.array([0, 0 ,0 ,0, 0, 0 ,0 ,0]) # Average weights 
	max_iters = 10000
	
	print('stepsize = ', stepsize, '\t', 'lambda = ', lambda_)
	
	#Read training data	
	training_y_full, training_tx_full, training_ids = load_csv_data(training_data_path, sub_sample=False)
	accuracy_average = 0
	
	for i_trial in range(n_trials):
		
		#Split the data into test and training
		training_tx, training_y, test_tx, test_y = split_data(training_tx_full, training_y_full, train_ratio, i_trial)
					
		# Split the training data by jet numbers and validity of DER_mass_MMC
		list_training_tx, list_training_y, list_training_ids = split_data_by_jet_num_feature(training_tx, training_y, training_ids[:limit])
					
		# Split the test data by jet numbers and validity of DER_mass_MMC
		list_test_tx, list_test_y, list_test_ids = split_data_by_jet_num_feature(test_tx, test_y, training_ids[:250000-limit])
		
		# List that will contain all the norms of the weights for each jet
		list_weight_norms = np.array([0,0,0,0,0, 0 ,0 ,0])
		
		# Loop through jet numbers
		for i in range(8):
			
			training_tx_i, training_y_i, training_ids_i = list_training_tx[i], list_training_y[i], list_training_ids[i]
			test_tx_i, test_y_i, test_ids_i = list_test_tx[i], list_test_y[i], list_test_ids[i]
			sd_limit = list_sd_limit[i]
			
			# ******************************************* TRAINING *************************************************************		
			#Standardize the training data after removing outliers
			training_tx_i, training_y_i, training_tx_i_mean, training_tx_i_std = standardize_training(training_tx_i,training_y_i,sd_limit)
			
			#Create polynomial expansions
			training_tx_i = get_polynomial(training_tx_i, list_poly[i])
		
			#Get weights
			initial_w = np.zeros((training_tx_i.shape[1], ))
			weights_i, loss_i = reg_logistic_regression_newton(training_y_i, training_tx_i,lambda_, initial_w, max_iters, stepsize)
			
			# ******************************************* PREDICTION *************************************************************
			#Standardize the test data using training mean and std
			test_tx_i_standardized = standardize_test(test_tx_i,training_tx_i_mean, training_tx_i_std)
		
			#Create polynomial expansion for the test data
			test_tx_i_standardized = get_polynomial(test_tx_i_standardized, list_poly[i])
			
			#Get predictions and accuracy for the current jet
			y_pred_i = predict_labels(weights_i, test_tx_i_standardized, 1) # 1 for logistic regression, 0 for linear
			accuracy = verify_prediction(y_pred_i, test_y_i)
			
			#Collate data for all jets
			if(i == 0):
				y_pred_all = y_pred_i
				test_y_all = test_y_i
				test_ids_all = test_ids_i
				
			else:
				y_pred_all = np.concatenate((y_pred_all,y_pred_i))
				test_y_all = np.concatenate((test_y_all,test_y_i))
				test_ids_all = np.concatenate((test_ids_all,test_ids_i))
				
			list_weight_norms[i] = np.linalg.norm(weights_i)
		
		# Calculate average accuracy for the current test-train split using all jets
		accuracy = verify_prediction(y_pred_all, test_y_all)
		print(accuracy)
		
		# Add the current accuracy and weight norms to the average which will be outputted later
		accuracy_average = ( accuracy + (accuracy_average*i_trial) )/(i_trial+1)
		weights_average = ( list_weight_norms + (weights_average*i_trial) )/(i_trial+1)
	
	# Print the average accuracy and weight norm for the current test-train split after 10 fold cross validation
	print('Accuracy Average is:', accuracy_average)
	print('Weights Average Norms are:', weights_average)
		

if __name__ == "__main__":
    train_ratio = 0.9
    lambdas = 1.
    gammas = [1e-01]
    	
    for gamma in gammas:
        run_model(train_ratio,gamma,lambdas)
	

	
