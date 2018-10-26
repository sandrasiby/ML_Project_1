from proj1_helpers import *
from implementations import *
# from logistic_regression import *
from logistic_regression import *

def exp_three_models():

	training_data_path = "train.csv"
	test_data_path = "test.csv"
	output_path = "output.csv"
	train_ratio = 0.3
	n_trials = 10
	limit = int(train_ratio * 250000)
	sd_limit_0 = 2.3
	sd_limit_1 = 2.5
	sd_limit_2 = 2.75
	sd_limit_3 = 2.6
	list_sd_limit = [sd_limit_0,sd_limit_1,sd_limit_2,sd_limit_3]
	list_poly = [2,2,2,2]
	list_lambda = [0,0.1,0.5,1.0,5.0,10.0]
	weights_average = np.array([0, 0 ,0 ,0])
	maxiter, stepsize, lambda_, is_newton = 20000, 1e-05, 1, 0
	print('isNewton = ', is_newton)
	
	print('stepsize = ', stepsize)
	f = open('lambda_check.txt','w')
	f.write('lambda' + '\t' + 'average accuracy' +  '\t' + 'average weights' +'\n')
	#Read training data
	# print("Read training data: START")
	training_y_full, training_tx_full, training_ids = load_csv_data(training_data_path, sub_sample=False)
	# print("Read training data: DONE")


	for i_lam in range(len(list_lambda)):
		lambda_ = list_lambda[i_lam]
		accuracy_average = 0
		print('lambda = ', lambda_)		
		for i_trial in range(n_trials):
			
			#Split the data into test and training
			# print("Split data: START")
			training_tx, training_y, test_tx, test_y = split_data(training_tx_full, training_y_full, train_ratio, i_trial)
			# print("Split data: DONE")
			
			
			# print("Split data based on jet number (2 and 3 separate): START")
			list_training_tx, list_training_y, list_training_ids = split_data_by_jet_num_2(training_tx, training_y, training_ids[:limit])
			# print("Split data based on jet number (2 and 3 separate): DONE")
			
			# Split test data into various jet numbers
			list_test_tx, list_test_y, list_test_ids = split_data_by_jet_num_2(test_tx, test_y, training_ids[:250000-limit])
			list_weight_norms = np.array([0,0,0,0])
			
			# Loop through jet numbers
			for i in range(4):
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
				training_tx_i = get_polynomial(training_tx_i, list_poly[i])
			
				#Get weights
				weights_i = logistic_regression(training_y_i, training_tx_i, is_newton, stepsize, maxiter, lambda_)
				# list_weights.append(weights_i)
			
				# ******************************************* PREDICTION *************************************************************
				#Standardize the test data using training mean and std
				test_tx_i_standardized = standardize_test(test_tx_i,training_tx_i_mean, training_tx_i_std)
			
				#Create polynomial expansion for the test data
				test_tx_i_standardized = get_polynomial(test_tx_i_standardized, list_poly[i])
				
				#Get predictions for all the jets 
				y_pred_i = predict_labels(weights_i, test_tx_i_standardized)
				accuracy = verify_prediction(y_pred_i, test_y_i)
				# print(weights_i)
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
				# print('Accuracy 0 is:',accuracy)
				# print(list_weight_norms)
			accuracy = verify_prediction(y_pred_all, test_y_all)
			# print('Accuracy Cross is:',accuracy)
			accuracy_average = ( accuracy + (accuracy_average*i_trial) )/(i_trial+1)
			weights_average = ( list_weight_norms + (weights_average*i_trial) )/(i_trial+1)
		print('Accuracy Average is:', accuracy_average)
		print('Weights Average Norms are:', weights_average)
		f.write(str(lambda_) + '\t' + str(accuracy_average) + '\t' + str(weights_average) + '\n')
	
	f.close()

	
	
	
	

if __name__ == "__main__":

	exp_three_models()

	