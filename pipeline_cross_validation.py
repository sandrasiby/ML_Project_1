from proj1_helpers import *
from implementations import *
from logistic_regression import *
if __name__ == "__main__":

	training_data_path = "train.csv"
	test_data_path = "test.csv"
	output_path = "output.csv"
	train_ratio = 0.6
	n_trials = 20	
	
	#Read training data
	print("Read training data: START")
	training_y_full, training_tx_full, training_ids = load_csv_data(training_data_path, sub_sample=False)
	print("Read training data: DONE")
	
	for i in range(n_trials):
		#Split the data into test and training
		print("Split data: START")
		training_tx, training_y, test_tx, test_y = split_data(training_tx_full, training_y_full, train_ratio, i)
		print("Split data: DONE")
		
		#Standardize the training data
		training_tx, training_tx_mean, training_tx_std = standardize_training(training_tx)
		
		#Remove outliers and -999 from the standardized training dataset
		training_tx, training_y = remove_outliers(training_tx,training_y,2.3)
		
		#Use polynomial terms
		training_tx = get_polynomial(training_tx,2)
				
		#Get optimal weights
		print("Compute least squares: START")
		maxiter, stepsize,lambda_ = 500, 1, 0
		weights = logistic_regression(training_y, training_tx,1,stepsize,maxiter,lambda_)
		print("Compute least squares: DONE")
		# print("Weights are:",weights)
				
		#Standardize the test data using training mean and std
		test_tx = standardize_test(test_tx,training_tx_mean, training_tx_std)
		
		#Use polynomial terms
		test_tx = get_polynomial(test_tx,2)
		
		# Perform prediction
		print("Perform prediction: START")
		y_pred = predict_labels(weights, test_tx)
		print("Perform prediction: DONE")

		
		# Verify predictions with in house test data
		print("Verify prediction: START")
		accuracy = verify_prediction(y_pred, test_y)
		print("Verify prediction: DONE")
		print('Accuracy is:',accuracy)
		
		# Average the weights
		if(i>0):
			weights_final = (weights_final*i + weights) / (i+1)
		else: 
			weights_final = weights
	
	#Read test data
	print("Read test data: START")
	test_y, test_tx, test_ids = load_csv_data(test_data_path)
	print("Read test data: DONE")
	
	#Standardize the test data using training mean and std
	test_tx = standardize_test(test_tx,training_tx_mean, training_tx_std)
	
	#Use polynomial terms
	test_tx = get_polynomial(test_tx,2)
	
	# Perform prediction
	print("Perform prediction: START")
	y_pred = predict_labels(weights, test_tx)
	print("Perform prediction: DONE")

	# Create output file
	print("Write CSV output: START")
	create_csv_submission(test_ids, y_pred, output_path)
	print("Write CSV output: DONE")

