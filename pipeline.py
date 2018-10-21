from proj1_helpers import *
from implementations import *
# from logistic_regression import *
from logistic_regression import *

def exp_three_models():

	training_data_path = "train.csv"
	test_data_path = "test.csv"
	output_path = "output.csv"
	train_ratio = 0.9
	n_trials = 1
	limit = int(train_ratio * 250000)
	sd_limit_0 = 2.3
	sd_limit_1 = 2.75
	sd_limit_others = 2.3
	#Read training data
	print("Read training data: START")
	training_y, training_tx, training_ids = load_csv_data(training_data_path, sub_sample=False)
	print("Read training data: DONE")
	
	for i in range(n_trials):
		#Split the data into test and training
		# print("Split data: START")
		# training_tx, training_y, test_tx, test_y = split_data(training_tx_full, training_y_full, train_ratio, i)
		# print("Split data: DONE")

		print("Split data based on jet number: START")
		training_tx_0, training_y_0, training_ids_0, training_tx_1, training_y_1, training_ids_1, \
		training_tx_others, training_y_others, training_ids_others \
		= split_data_by_jet_num(training_tx, training_y, training_ids)
		print("Split data based on jet number: DONE")

		#Standardize the training data
		training_tx_0, training_tx_0_mean, training_tx_0_std = standardize_training(training_tx_0)
		training_tx_1, training_tx_1_mean, training_tx_1_std = standardize_training(training_tx_1)
		training_tx_others, training_tx_others_mean, training_tx_others_std = standardize_training(training_tx_others)
		
		#Remove outliers and -999 from the standardized training dataset
		#training_tx, training_y = remove_outliers(training_tx,training_y,2.3)
		training_tx_0, training_y_0, training_tx_0_out, training_y_0_out = remove_outliers(training_tx_0,training_y_0,sd_limit_0)
		training_tx_1, training_y_1, training_tx_1_out, training_y_1_out = remove_outliers(training_tx_1,training_y_1,sd_limit_1)
		training_tx_others, training_y_others, training_tx_others_out, training_y_others_out = remove_outliers(training_tx_others,training_y_others,sd_limit_others)

		training_tx_0 = get_polynomial(training_tx_0, 2)
		training_tx_1 = get_polynomial(training_tx_1, 2)
		training_tx_others = get_polynomial(training_tx_others, 2)

		# weights_0 = least_squares(training_y_0, training_tx_0)
		# weights_1 = least_squares(training_y_1, training_tx_1)
		# weights_others = least_squares(training_y_others, training_tx_others)

		maxiter, stepsize = 50000, 5e-01
		lambda_ = 0
		weights_0 = logistic_regression(training_y_0, training_tx_0, 1, stepsize, maxiter, lambda_)
		weights_1 = logistic_regression(training_y_1, training_tx_1, 1, stepsize, maxiter, lambda_)
		weights_others = logistic_regression(training_y_others, training_tx_others, 1, stepsize, maxiter, lambda_)

		test_tx_0, test_y_0, test_ids_0, test_tx_1, test_y_1, test_ids_1, test_tx_others, test_y_others, \
		test_ids_others = split_data_by_jet_num(test_tx, test_y, training_ids)

		#Standardize the test data using training mean and std
		test_tx_0 = standardize_test(test_tx_0,training_tx_0_mean, training_tx_0_std)
		test_tx_1 = standardize_test(test_tx_1,training_tx_1_mean, training_tx_1_std)
		test_tx_others = standardize_test(test_tx_others,training_tx_others_mean, training_tx_others_std)

		test_tx_0, test_y_0, test_tx_0_out, test_y_0_out = remove_outliers(test_tx_0,test_y_0,sd_limit_0)
		#training_tx_1, training_y_1, training_tx_1_out, training_y_1_out = remove_outliers(training_tx_1,training_y_1,2.3)
		#training_tx_others, training_y_others, training_tx_others_out, training_y_others_out = remove_outliers(training_tx_others,training_y_others,2.3)


		test_tx_0 = get_polynomial(test_tx_0, 2)
		test_tx_1 = get_polynomial(test_tx_1, 2)
		test_tx_others = get_polynomial(test_tx_others, 2)

		y_pred_0 = predict_labels(weights_0, test_tx_0)
		y_pred_1 = predict_labels(weights_1, test_tx_1)
		y_pred_others = predict_labels(weights_others, test_tx_others)

		y_pred_0_out = np.array([-1] * test_y_0_out.shape[0])

		accuracy = verify_prediction(y_pred_0, test_y_0)
		print('Accuracy 0 is:',accuracy)
		accuracy = verify_prediction(y_pred_1, test_y_1)
		print('Accuracy 1 is:',accuracy)
		accuracy = verify_prediction(y_pred_others, test_y_others)
		print('Accuracy Others is:',accuracy)

		y_pred_all = np.concatenate((y_pred_0, y_pred_1, y_pred_others, y_pred_0_out))
		test_y_all = np.concatenate((test_y_0, test_y_1, test_y_others, test_y_0_out))
		accuracy = verify_prediction(y_pred_all, test_y_all)
		print('Accuracy Cross is:',accuracy)

	#Read test data
	print("Read test data: START")
	test_y, test_tx, test_ids = load_csv_data(test_data_path)
	print("Read test data: DONE")

	test_tx_0, test_y_0, test_ids_0, test_tx_1, test_y_1, test_ids_1, test_tx_others, test_y_others, \
		test_ids_others = split_data_by_jet_num(test_tx, test_y, test_ids)
	
	test_tx_0 = standardize_test(test_tx_0,training_tx_0_mean, training_tx_0_std)
	test_tx_1 = standardize_test(test_tx_1,training_tx_1_mean, training_tx_1_std)
	test_tx_others = standardize_test(test_tx_others,training_tx_others_mean, training_tx_others_std)
	
	test_tx_0, test_y_0, test_tx_0_out, test_y_0_out = remove_outliers(test_tx_0,test_y_0,sd_limit_0)

	#Use polynomial terms
	test_tx_0 = get_polynomial(test_tx_0, 2)
	test_tx_1 = get_polynomial(test_tx_1, 2)
	test_tx_others = get_polynomial(test_tx_others, 2)

	# Perform prediction
	print("Perform prediction: START")
	y_pred_0 = predict_labels(weights_0, test_tx_0)
	y_pred_1 = predict_labels(weights_1, test_tx_1)
	y_pred_others = predict_labels(weights_others, test_tx_others)
	y_pred_0_out = np.array([-1] * test_y_0_out.shape[0])
	print("Perform prediction: DONE")

	test_ids_all = np.concatenate((test_ids_0, test_ids_1, test_ids_others))
	y_pred_all = np.concatenate((y_pred_0, y_pred_1, y_pred_others, y_pred_0_out))	

	zipped_list = sorted(zip(test_ids_all, y_pred_all))
	test_ids_all, y_pred_all = zip(*zipped_list)
	test_ids_all = np.array(test_ids_all)
	y_pred_all = np.array(y_pred_all)
	# Create output file
	print("Write CSV output: START")
	create_csv_submission(test_ids_all, y_pred_all, output_path)
	print("Write CSV output: DONE")
	

if __name__ == "__main__":

	exp_three_models()

	# training_data_path = "train.csv"
	# test_data_path = "test.csv"
	# output_path = "output.csv"
    	
	# #Read training data
	# print("Read training data: START")
	# training_y, training_tx, training_ids = load_csv_data(training_data_path, sub_sample=False)
	# print("Read training data: DONE")
	
	# #Split the data into test and training
	# # print("Split data: START")
	# training_tx, training_y, test_tx, test_y = split_data(training_tx, training_y, 0.3, 5)
	# # print("Split data: DONE")

	# #Split training data into 3

	# #Use only DERivative or PRImitive variables
	# # training_tx = get_data_by_type(training_tx, 'DER')
    
	# #Remove -999 columns from the training set
	# # training_tx = remove_columns_invalid(training_tx, 50)
	# # training_tx = replace_999(training_tx)
	# # training_tx = remove_columns_colinear(training_tx, 80)
	# # training_tx, training_y = remove_rows_invalid(training_tx,training_y)
	
	# #Standardize the training data
	# training_tx, training_tx_mean, training_tx_std = standardize_training(training_tx)
	
	# #Remove outliers and -999 from the standardized training dataset
	# training_tx, training_y = remove_outliers(training_tx,training_y,4)
	
	# #Use polynomial terms
	# # training_tx = get_polynomial(training_tx,4)
			
	# #Get optimal weights
	# print("Compute least squares: START")
	# # weights = least_squares(training_y, training_tx)
	# maxiter, stepsize = 50000, 2e-07
	# # maxiter, stepsize = 50000, 5e-01
	# lambda_ = 0
	# weights = logistic_regression(training_y, training_tx,0,stepsize,maxiter,lambda_)
	# print("Compute least squares: DONE")
	# print("Weights are:",weights)
	
	# #Compute MSE
	# train_mse = compute_mse(training_y, training_tx, weights)
	# print("MSE:", train_mse)
	
	# #Read test data
	# # print("Read test data: START")
	# # test_y, test_tx, test_ids = load_csv_data(test_data_path)
	# # print("Read test data: DONE")
	
	# #Use only the DER or PRI columns of the test data
	# # test_tx = get_data_by_type(test_tx, 'DER')

	# #Remove -999 columns from the test dataset
	# # test_tx = remove_columns_invalid(test_tx, 50)
	# # test_tx = remove_columns_colinear(test_tx, 80)
	# # test_tx = replace_999(test_tx)
	
	# #Standardize the test data using training mean and std
	# test_tx = standardize_test(test_tx,training_tx_mean, training_tx_std)
	
	# #Use polynomial terms
	# # test_tx = get_polynomial(test_tx,4)
		
	# # Perform prediction
	# print("Perform prediction: START")
	# y_pred = predict_labels(weights, test_tx)
	# print("Perform prediction: DONE")
	
	# # Verify predictions with in house test data
	# print("Verify prediction: START")
	# accuracy = verify_prediction(y_pred, test_y)
	# print("Verify prediction: DONE")
	# print('Accuracy is:',accuracy)
	
	# # Create output file
	# # print("Write CSV output: START")
	# # create_csv_submission(test_ids, y_pred, output_path)
	# # print("Write CSV output: DONE")

