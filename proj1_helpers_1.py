# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from logistic_regression import * 
from collections import Counter

def split_data(tx, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    ind = np.random.permutation(len(y))
    limit = int(np.floor(ratio * len(y)))
    train_tx = tx[ind[:limit]]
    train_y = y[ind[:limit]]
    test_tx = tx[ind[limit:]]
    test_y = y[ind[limit:]]
    return train_tx, train_y, test_tx, test_y

# Split the data by jet number, with 2 and 3 grouped as one
def split_data_by_jet_num(tx, y, ids):

    tx_jet_num_0 = tx[tx[:, 22] == 0]
    tx_jet_num_1 = tx[tx[:, 22] == 1]
    tx_jet_num_others = tx[tx[:, 22] > 1]
    y_jet_num_0 = y[tx[:, 22] == 0]
    y_jet_num_1 = y[tx[:, 22] == 1]
    y_jet_num_others = y[tx[:, 22] > 1]
    ids_jet_num_0 = ids[tx[:, 22] == 0]
    ids_jet_num_1 = ids[tx[:, 22] == 1]
    ids_jet_num_others = ids[tx[:, 22] > 1]

    tx_jet_num_0 = np.delete(tx_jet_num_0, [22, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29], axis=1)
    tx_jet_num_1 = np.delete(tx_jet_num_1, [22, 4, 5, 6, 12, 26, 27, 28], axis=1)
    tx_jet_num_others = np.delete(tx_jet_num_others, [22], axis=1)

    return tx_jet_num_0, y_jet_num_0, ids_jet_num_0, tx_jet_num_1, y_jet_num_1, ids_jet_num_1, \
     tx_jet_num_others, y_jet_num_others, ids_jet_num_others

def split_data_by_jet_num_1(tx, y, ids):
		
    tx_jet_num =  [  tx[tx[:, 22] == 0],  tx[tx[:, 22] == 1],  tx[tx[:, 22] >= 2]]
    
    y_jet_num =   [   y[tx[:, 22] == 0],   y[tx[:, 22] == 1],   y[tx[:, 22] >= 2]] 
    
    ids_jet_num = [ ids[tx[:, 22] == 0], ids[tx[:, 22] == 1], ids[tx[:, 22] >= 2]]
    	
    # tx_jet_num[0] = np.delete(tx_jet_num[0], [22, 8, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29], axis=1)
    # tx_jet_num[1] = np.delete(tx_jet_num[1], [22, 4, 5, 6, 12, 26, 27, 28, 29], axis=1)
    # tx_jet_num[2] = np.delete(tx_jet_num[2], [22, 29], axis=1)
    tx_jet_num[2] = np.delete(tx_jet_num[2], [22, 15, 18, 20, 25,28, 29], axis=1)
    # tx_jet_num[3] = np.delete(tx_jet_num[3], [22, 29], axis=1)

	#Removing some PHI as well
    tx_jet_num[0] = np.delete(tx_jet_num[0], [22, 8, 4, 5, 6, 12, 15, 18, 20, 23, 24, 25, 26, 27, 28, 29], axis=1)
    tx_jet_num[1] = np.delete(tx_jet_num[1], [22, 4, 5, 6, 12, 15, 18, 20, 25, 26, 27, 28, 29], axis=1)
    # tx_jet_num[2] = np.delete(tx_jet_num[2], [22, 15, 18, 20, 25, 27, 28, 29], axis=1)
    # tx_jet_num[3] = np.delete(tx_jet_num[3], [22, 15, 18, 20, 25, 27,  29], axis=1)
	
	#Removing only PHI
    # tx_jet_num[0] = np.delete(tx_jet_num[0], [22, 15, 18, 20, 25, 27, 28], axis=1)
    # tx_jet_num[1] = np.delete(tx_jet_num[1], [22, 15, 18, 20, 25, 27, 28], axis=1)
    # tx_jet_num[2] = np.delete(tx_jet_num[2], [22, 15, 18, 20, 25, 27, 28], axis=1)
    # tx_jet_num[3] = np.delete(tx_jet_num[3], [22, 15, 18, 20, 25, 27, 28], axis=1)

	
    return tx_jet_num, y_jet_num, ids_jet_num	 	 
	 
	 
# Split the data by jet number, with 2 and 3 separate	 
def split_data_by_jet_num_2(tx, y, ids):
		
    tx_jet_num =  [  tx[tx[:, 22] == 0],  tx[tx[:, 22] == 1],  tx[tx[:, 22] == 2],  tx[tx[:, 22] == 3]]
    
    y_jet_num =   [   y[tx[:, 22] == 0],   y[tx[:, 22] == 1],   y[tx[:, 22] == 2],   y[tx[:, 22] == 3]] 
    
    ids_jet_num = [ ids[tx[:, 22] == 0], ids[tx[:, 22] == 1], ids[tx[:, 22] == 2], ids[tx[:, 22] == 3] ]
    	
    # tx_jet_num[0] = np.delete(tx_jet_num[0], [22, 8, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29], axis=1)
    # tx_jet_num[1] = np.delete(tx_jet_num[1], [22, 4, 5, 6, 12, 26, 27, 28, 29], axis=1)
    # tx_jet_num[2] = np.delete(tx_jet_num[2], [22, 29], axis=1)
    tx_jet_num[2] = np.delete(tx_jet_num[2], [22, 15, 18, 20, 25,28, 29], axis=1)
    # tx_jet_num[3] = np.delete(tx_jet_num[3], [22, 29], axis=1)

	#Removing some PHI as well
    tx_jet_num[0] = np.delete(tx_jet_num[0], [22, 8, 4, 5, 6, 12, 15, 18, 20, 23, 24, 25, 26, 27, 28, 29], axis=1)
    tx_jet_num[1] = np.delete(tx_jet_num[1], [22, 4, 5, 6, 12, 15, 18, 20, 25, 26, 27, 28, 29], axis=1)
    # tx_jet_num[2] = np.delete(tx_jet_num[2], [22, 15, 18, 20, 25, 27, 28, 29], axis=1)
    tx_jet_num[3] = np.delete(tx_jet_num[3], [22, 15, 18, 20, 25, 27,  29], axis=1)
	
	#Removing only PHI
    # tx_jet_num[0] = np.delete(tx_jet_num[0], [22, 15, 18, 20, 25, 27, 28], axis=1)
    # tx_jet_num[1] = np.delete(tx_jet_num[1], [22, 15, 18, 20, 25, 27, 28], axis=1)
    # tx_jet_num[2] = np.delete(tx_jet_num[2], [22, 15, 18, 20, 25, 27, 28], axis=1)
    # tx_jet_num[3] = np.delete(tx_jet_num[3], [22, 15, 18, 20, 25, 27, 28], axis=1)

	
    return tx_jet_num, y_jet_num, ids_jet_num	 
	
def split_data_by_jet_num_feature(tx, y, ids):
		
    n_rows = tx.shape[0]
	# Calculate additional features reported in paper 
    eMHT = tx[:,29] - tx[:,26] - tx[:,23]
    assym = np.divide(tx[:,19] - eMHT, tx[:,19] + eMHT)
    delta_phi = tx[:,25] - tx[:,28]
    delta_phi_met = 0.5*(tx[:,25] + tx[:,28])
    special = np.divide( np.multiply(tx[:,0],tx[:,10]) , tx[:,9] )
	
	# Add the features to tx
    tx = np.append(tx,np.reshape(assym,(n_rows,1)),axis= 1)
    tx = np.append(tx,np.reshape(delta_phi,(n_rows,1)),axis= 1)
    tx = np.append(tx,np.reshape(delta_phi_met,(n_rows,1)),axis= 1)
    tx = np.append(tx,np.reshape(special,(n_rows,1)),axis= 1)
	
    tx_mass 	= tx[tx[:,0] > -999]
    tx_no_mass 	= tx[tx[:,0] == -999]
    y_mass = y[tx[:,0] > -999]
    y_no_mass = y[tx[:,0]== -999]
    ids_mass = ids[tx[:,0] > -999]
    ids_no_mass = ids[tx[:,0] == -999]
	
    tx_jet_num =  [ tx_mass[tx_mass[:, 22] == 0], tx_mass[tx_mass[:, 22] == 1], tx_mass[tx_mass[:, 22] == 2], tx_mass[tx_mass[:, 22] == 3], tx_no_mass[tx_no_mass[:, 22] == 0],  tx_no_mass[tx_no_mass[:, 22] == 1],  tx_no_mass[tx_no_mass[:, 22] == 2],  tx_no_mass[tx_no_mass[:, 22] == 3]]
    
    y_jet_num = [y_mass[tx_mass[:, 22] == 0],   y_mass[tx_mass[:, 22] == 1],   y_mass[tx_mass[:, 22] == 2],   y_mass[tx_mass[:, 22] == 3], y_no_mass[tx_no_mass[:, 22] == 0],   y_no_mass[tx_no_mass[:, 22] == 1],   y_no_mass[tx_no_mass[:, 22] == 2],   y_no_mass[tx_no_mass[:, 22] == 3]] 
	
    ids_jet_num = [ids_mass[tx_mass[:, 22] == 0],   ids_mass[tx_mass[:, 22] == 1],   ids_mass[tx_mass[:, 22] == 2],   ids_mass[tx_mass[:, 22] == 3], ids_no_mass[tx_no_mass[:, 22] == 0],   ids_no_mass[tx_no_mass[:, 22] == 1],   ids_no_mass[tx_no_mass[:, 22] == 2],   ids_no_mass[tx_no_mass[:, 22] == 3]] 
    
    # ids_jet_num = [ ids[tx[:, 22] == 0], ids[tx[:, 22] == 1], ids[tx[:, 22] == 2], ids[tx[:, 22] == 3] ]
    
	
	
    # tx_jet_num[0] = np.delete(tx_jet_num[0], [22, 8, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29], axis=1)
    # tx_jet_num[1] = np.delete(tx_jet_num[1], [22, 4, 5, 6, 12, 26, 27, 28, 29], axis=1)
    # tx_jet_num[2] = np.delete(tx_jet_num[2], [22, 29], axis=1)
    # tx_jet_num[2] = np.delete(tx_jet_num[2], [22, 15, 18, 20, 25,28, 29], axis=1)
	
    tx_jet_num[2] = np.delete(tx_jet_num[2], [22, 15, 18, 20, 25, 28], axis=1)
    tx_jet_num[6] = np.delete(tx_jet_num[6], [0, 22, 15, 18, 20, 25, 28], axis=1)
    
	# tx_jet_num[3] = np.delete(tx_jet_num[3], [22, 29], axis=1)

	#Removing some PHI as well
    tx_jet_num[0] = np.delete(tx_jet_num[0], [22, 8, 4, 5, 6, 12, 15, 18, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], axis=1)
    tx_jet_num[4] = np.delete(tx_jet_num[4], [0, 22, 8, 4, 5, 6, 12, 15, 18, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], axis=1)
    # tx_jet_num[1] = np.delete(tx_jet_num[1], [22, 4, 5, 6, 12, 15, 18, 20, 25, 26, 27, 28, 29, 30, 31,32], axis=1)
    tx_jet_num[1] = np.delete(tx_jet_num[1], [22, 4, 5, 6, 12, 15, 18, 20, 25, 26, 27, 28, 29, 31,32], axis=1)
    tx_jet_num[5] = np.delete(tx_jet_num[5], [0, 22, 4, 5, 6, 12, 15, 18, 20, 25, 26, 27, 28, 29, 31,32], axis=1)
    # tx_jet_num[2] = np.delete(tx_jet_num[2], [22, 15, 18, 20, 25, 27, 28, 29], axis=1)
    tx_jet_num[3] = np.delete(tx_jet_num[3], [22, 15, 18, 20, 25, 27,  29], axis=1)
    tx_jet_num[7] = np.delete(tx_jet_num[7], [0, 22, 15, 18, 20, 25, 27,  29], axis=1)
	
    # tx_jet_num[3] = np.delete(tx_jet_num[3], [22, 15, 18, 20, 25], axis=1)
	
	
	
    return tx_jet_num, y_jet_num, ids_jet_num	 
	 
def read_csv_headers(data_path):
    with open(data_path, 'r') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

    return fieldnames

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::20]
        input_data = input_data[::20]
        ids = ids[::20]

    return yb, input_data, ids

#Create polynomial terms for each column of tx 
def get_polynomial(tx,poly_term):
	tx_temp = tx.copy()
	tx_return = tx.copy()
	for i in range(poly_term-1):
		tx_temp = np.multiply(tx_temp,tx)
		tx_return = np.c_[ tx_return, tx_temp ]
	return tx_return

def get_data_by_type(input_data, type_tag):
    """Returns a sub-set of the input data based on tag - PRI or DER"""
    if type_tag == 'PRI':
        input_data_ss = input_data[:, :13]
    elif type_tag == 'DER':
        input_data_ss = input_data[:, 13:]
    return input_data_ss

# Standardize the training data using Z score standardization
def standardize_training(x):
	mean_x = np.mean(x,axis=0)
	x = x - mean_x
	std_x = np.std(x,axis=0)
	x = x / std_x
	return x, mean_x, std_x
	
# Standardize the test data using the mean and SD of the training data
def standardize_test(x,mean_x,std_x):
	"""Standardize the original data set."""
	x = x - mean_x
	x = x / std_x
	return x

# Standardize data using min-max standardization
# NOT REQUIRED
def standardize_minmax(tx):
	n_col = tx.shape[1]
	tx_return = tx.copy()
	for col in range(n_col):
		tx_return[:,col] = (tx[:,col] - min(tx[:,col]))/(max(tx[:,col]) - min(tx[:,col]))
	return tx_return

# Remove outliers from the dataset. i.e. points which are more than 2 SD away.
def remove_outliers(tx,y,threshold):
    #Author : BN 
	#Date: 17/10/2018
    shape_tx = np.shape(tx)
    n_columns = shape_tx[1]
    # print(n_columns)
    for i_col in range(n_columns):
        
        y = y[np.where(np.abs(tx[:,i_col]) < threshold)]
        # print(np.shape(y))
        tx = tx[np.where(np.abs(tx[:,i_col]) < threshold)]
        y_out = y[np.where(np.abs(tx[:,i_col]) >= threshold)]
        tx_out = tx[np.where(np.abs(tx[:,i_col]) >= threshold)]       
    return tx, y, tx_out, y_out

# In the test data, set the prediction for all samples with -999 in DER_MASS to background
def set_background(tx,y, sd):
	# der_mass_mmc = tx[:,0] # Get the der_mass_mmc column from tx
	# print('y before setting =', y[np.where(der_mass_mmc == -999)])
	# y[np.where(der_mass_mmc == -999)] = -1
	# print('y after setting =', y[np.where(der_mass_mmc == -999)])
	count = 0
	# Set background for outliers
	n_rows = tx.shape[0]
	for i in range(n_rows):
		row_vec = tx[i,:]
		n_outliers = np.where(row_vec > sd)
		# print(np.shape(n_outliers)[1])
		if((np.shape(n_outliers)[1]) > 5):
			count +=1
			y[i] = -1
		else:
			n_outliers = np.where(row_vec < -1*sd)
			if((np.shape(n_outliers)[1]) > 5):
				count +=1
				y[i] = -1
		
	print('number changed =', count)
	
	return y

def redo_standardization(tx, mean, sd):
	tx = tx*sd
	tx = tx + mean
	tx, mean_tx,std_tx = standardize_training(tx)
	return tx, mean_tx, std_tx
	
# Replace -999 in a column with mean of remaining elements
def replace_999(tx):
	n_col = tx.shape[1]
	tx_return = tx.copy()
	
	for col in range(n_col):
		vec_temp = tx[:,col]
		vec = vec_temp[vec_temp > -999]
		if(len(vec) > 0):
			# mean_vec = np.mean(vec)
			mean_vec = np.median(vec)
			# vec_temp[vec_temp == -999] = mean_vec
			vec_temp[vec_temp == -999] = mean_vec
		tx_return[:,col] = vec_temp
	return tx_return

# Replace -999 in der_mass_mmc with the mean of the background
def replace_999_mass(tx,y):
	der_mass_mmc = tx[:,0] # Get the der_mass_mmc column from tx
	vec_bg = der_mass_mmc[np.where(y == -1)] # vector containing mass entries for the background only
	# print('No. of background = ', np.shape(vec_bg)) 
	vec_bg = vec_bg[vec_bg > -999]  # get non -999 entries to calculate their mean and median
	# print('No. of valid background = ', np.shape(vec_bg))
	mean_bg = np.mean(vec_bg)
	# print('Mean. of valid background = ', mean_bg)
	median_bg = np.median(vec_bg)
	
	# Replace all -999 in der_mass_mmc with the median or mean of the background data with proper values
	# print('der_mass BEFORE', tx[:,0])
	der_mass_mmc[np.where(tx[:,0] == -999)] = median_bg
	
	# difference_vector = der_mass_mmc - tx[:,0]
	# print('difference vector = ', difference_vector)
	
	tx[:,0] = der_mass_mmc
	# print('der_mass AFTER', tx[:,0])
	return tx
	
	
#remove columns with large 999
def remove_columns_invalid(input_data, thresh):
    input_data_ss = input_data
    ct = 0
    for column in input_data.T:
        num_invalid = len(column[np.where(column == -999)])
        # print(float(num_invalid) / len(column) * 100)
        if float(num_invalid) / len(column) * 100 > thresh:
            input_data_ss = np.delete(input_data_ss, ct, 1)
            # print("Shape:", np.shape(input_data_ss))
            # print('ct is:',ct)            
#         ct += 1
        else: ct += 1 # BN edit to account for renumbering of columns
    return input_data_ss

#Generates class predictions given weights, and a test data matrix
def predict_labels(weights, data):
    
    # y_pred = np.dot(data, weights)
    y_pred = sigmoid(np.dot(data, weights))
	# sigm = sigmoid(y_pred)
    # print(y_pred)
    # print(sigm[sigm <= 0.5])
    y_pred[np.where(y_pred < 0.5)] = -1
    y_pred[np.where(y_pred >= 0.5)] = 1
    
    return y_pred

#Verify the predicted y against the test y for cross validation data sets 
def verify_prediction(y_pred, y_test):
    y_diff = y_pred - y_test
    # print(len(y_diff))
    nFalse = len(y_diff[y_diff !=0 ])
    # accuracy = 1 - np.sum(y_diff)/len(y_diff)
    accuracy = 1 - nFalse/len(y_diff)
    return accuracy
	
def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

if __name__ == "__main__":

    data_path = "/Users/sandra/Downloads/all/train.csv"
    print(read_csv_headers(data_path))
    yb, input_data, ids = load_csv_data(data_path, sub_sample=True)
    idss = get_data_by_type(input_data, "PRI")
    print(np.shape(idss))
    idss1 = remove_columns_invalid(idss, 50)
    print(np.shape(idss1))
