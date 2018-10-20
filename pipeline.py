from proj1_helpers import *
from implementations import *
# from logistic_regression import *
from logistic_regression import *
if __name__ == "__main__":

	training_data_path = "train.csv"
	test_data_path = "test.csv"
	output_path = "output.csv"
    	
	#Read training data
	print("Read training data: START")
	training_y, training_tx, training_ids = load_csv_data(training_data_path, sub_sample=False)
	print("Read training data: DONE")
	
	#Split the data into test and training
	print("Split data: START")
	training_tx, training_y, test_tx, test_y = split_data(training_tx, training_y, 0.3, 5)
	print("Split data: DONE")
	
	#Use only DERivative or PRImitive variables
	# training_tx = get_data_by_type(training_tx, 'DER')
    
	#Remove -999 columns from the training set
	# training_tx = remove_columns_invalid(training_tx, 50)
	# training_tx = replace_999(training_tx)
	# training_tx = remove_columns_colinear(training_tx, 80)
	# training_tx, training_y = remove_rows_invalid(training_tx,training_y)
	
	#Standardize the training data
	training_tx, training_tx_mean, training_tx_std = standardize_training(training_tx)
	
	#Remove outliers and -999 from the standardized training dataset
	training_tx, training_y = remove_outliers(training_tx,training_y,4)
	
	#Use polynomial terms
	# training_tx = get_polynomial(training_tx,4)
			
	#Get optimal weights
	print("Compute least squares: START")
	# weights = least_squares(training_y, training_tx)
	maxiter, stepsize = 50000, 2e-07
	# maxiter, stepsize = 50000, 5e-01
	lambda_ = 0
	weights = logistic_regression(training_y, training_tx,0,stepsize,maxiter,lambda_)
	print("Compute least squares: DONE")
	print("Weights are:",weights)
	
	#Compute MSE
	train_mse = compute_mse(training_y, training_tx, weights)
	print("MSE:", train_mse)
	
	#Read test data
	# print("Read test data: START")
	# test_y, test_tx, test_ids = load_csv_data(test_data_path)
	# print("Read test data: DONE")
	
	#Use only the DER or PRI columns of the test data
	# test_tx = get_data_by_type(test_tx, 'DER')

	#Remove -999 columns from the test dataset
	# test_tx = remove_columns_invalid(test_tx, 50)
	# test_tx = remove_columns_colinear(test_tx, 80)
	# test_tx = replace_999(test_tx)
	
	#Standardize the test data using training mean and std
	test_tx = standardize_test(test_tx,training_tx_mean, training_tx_std)
	
	#Use polynomial terms
	# test_tx = get_polynomial(test_tx,4)
		
	# Perform prediction
	print("Perform prediction: START")
	y_pred = predict_labels(weights, test_tx)
	print("Perform prediction: DONE")
	
	# Verify predictions with in house test data
	print("Verify prediction: START")
	accuracy = verify_prediction(y_pred, test_y)
	print("Verify prediction: DONE")
	print('Accuracy is:',accuracy)
	
	# Create output file
	# print("Write CSV output: START")
	# create_csv_submission(test_ids, y_pred, output_path)
	# print("Write CSV output: DONE")

