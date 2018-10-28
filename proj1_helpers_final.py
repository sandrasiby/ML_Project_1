# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from logistic_regression import * 
from collections import Counter

# Split data into training and test
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

# Split data by jet number after adding the 4 extra features 	
def split_data_by_jet_num_feature(tx, y, ids):
    '''We add 4 additional features:
    1. Assymetric Energy
    2. delta phi
    3. Avg_phi
    4. A special variable which is (Der_Mass_MMC * Der_pt_ratio) / Der_sum_pt
    
    We then split the tx into the 4 jet numbers and remove the columns with -999 (these are the values that are not calculated for the current jet
    Next, the variables deemed unnecessary, using the histogram plots, are removed for each jet
    '''	
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
	
	# Split the data into those rows with DER_MASS_MMC = -999 and those having valid DER_MASS_MMC
    tx_mass 	= tx[tx[:,0] > -999]
    tx_no_mass 	= tx[tx[:,0] == -999]
    y_mass = y[tx[:,0] > -999]
    y_no_mass = y[tx[:,0]== -999]
    ids_mass = ids[tx[:,0] > -999]
    ids_no_mass = ids[tx[:,0] == -999]
	
	# Split the data by jet numbers and create a list of tx, y and ids for the jet numbers
    tx_jet_num =  [ tx_mass[tx_mass[:, 22] == 0], tx_mass[tx_mass[:, 22] == 1], tx_mass[tx_mass[:, 22] == 2], tx_mass[tx_mass[:, 22] == 3], 
	tx_no_mass[tx_no_mass[:, 22] == 0],  tx_no_mass[tx_no_mass[:, 22] == 1],  tx_no_mass[tx_no_mass[:, 22] == 2],  tx_no_mass[tx_no_mass[:, 22] == 3]]
    
    y_jet_num = [y_mass[tx_mass[:, 22] == 0],   y_mass[tx_mass[:, 22] == 1],   y_mass[tx_mass[:, 22] == 2],   y_mass[tx_mass[:, 22] == 3], 
	y_no_mass[tx_no_mass[:, 22] == 0],   y_no_mass[tx_no_mass[:, 22] == 1],   y_no_mass[tx_no_mass[:, 22] == 2],   y_no_mass[tx_no_mass[:, 22] == 3]] 
	
    ids_jet_num = [ids_mass[tx_mass[:, 22] == 0],   ids_mass[tx_mass[:, 22] == 1],   ids_mass[tx_mass[:, 22] == 2],   ids_mass[tx_mass[:, 22] == 3], 
	ids_no_mass[tx_no_mass[:, 22] == 0],   ids_no_mass[tx_no_mass[:, 22] == 1],   ids_no_mass[tx_no_mass[:, 22] == 2],   ids_no_mass[tx_no_mass[:, 22] == 3]] 
	
	# Remove -999 features and features which are deemed unnecessary (like the PHI features)
    tx_jet_num[2] = np.delete(tx_jet_num[2], [22, 15, 18, 20, 25, 28], axis=1)    # Jet 2, valid DER_MASS_MMC
    tx_jet_num[6] = np.delete(tx_jet_num[6], [0, 22, 15, 18, 20, 25, 28], axis=1) # Jet 2, invalid DER_MASS_MMC
    tx_jet_num[0] = np.delete(tx_jet_num[0], [22, 8, 4, 5, 6, 12, 15, 18, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], axis=1) # Jet 0, valid DER_MASS_MMC
    tx_jet_num[4] = np.delete(tx_jet_num[4], [0, 22, 8, 4, 5, 6, 12, 15, 18, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], axis=1) # Jet 0, invalid DER_MASS_MMC
    tx_jet_num[1] = np.delete(tx_jet_num[1], [22, 4, 5, 6, 12, 15, 18, 20, 25, 26, 27, 28, 29, 31,32], axis=1) # Jet 1, valid DER_MASS_MMC
    tx_jet_num[5] = np.delete(tx_jet_num[5], [0, 22, 4, 5, 6, 12, 15, 18, 20, 25, 26, 27, 28, 29, 31,32], axis=1) # Jet 1, invalid DER_MASS_MMC
    tx_jet_num[3] = np.delete(tx_jet_num[3], [22, 15, 18, 20, 25, 27,  29], axis=1) # Jet 3, valid DER_MASS_MMC
    tx_jet_num[7] = np.delete(tx_jet_num[7], [0, 22, 15, 18, 20, 25, 27,  29], axis=1) # Jet 3, invalid DER_MASS_MMC
	    
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
	
# Standardize the test data using the mean and SD of the training data
def standardize_test(x,mean_x,std_x):
	x = x - mean_x
	x = x / std_x
	return x

# Remove outliers from the dataset. i.e. points which are more than a certain SD away.
def remove_outliers(tx,y,threshold):
    shape_tx = np.shape(tx)
    n_columns = shape_tx[1]
    for i_col in range(n_columns):
        y = y[np.where(np.abs(tx[:,i_col]) < threshold)]
        tx = tx[np.where(np.abs(tx[:,i_col]) < threshold)]
        
    return tx, y

# Remove outliers and standardize
def standardize_training(tx, y, threshold):
	
	# Perform an initial standardization of the data 
	mean_tx = np.mean(tx,axis=0)
	tx = tx - mean_tx
	std_tx = np.std(tx,axis=0)
	tx = tx / std_tx
	
	# Remove the outliers using this standardized data
	tx, y = remove_outliers(tx,y,threshold)
	
	# Recover original values of the cleaned dataset using the std and mean calculated earlier
	tx = tx*std_tx
	tx = tx + mean_tx
	
	# Standardize the cleaned dataset to obtain the correct mean and std
	mean_tx = np.mean(tx,axis=0)
	tx = tx - mean_tx
	std_tx = np.std(tx,axis=0)
	tx = tx / std_tx
	
	return tx, y, mean_tx, std_tx

#Generates class predictions given weights, and a test data matrix
def predict_labels(weights, data, isLogistic):
    
	# If the classification used is logistic regression
	if(isLogistic):
		y_pred = sigmoid(np.dot(data, weights)) # Obtain the log function values
		y_pred[np.where(y_pred < 0.5)] = -1
		y_pred[np.where(y_pred >= 0.5)] = 1
	
	else: # If we use linear regression
		y_pred = np.dot(data, weights)
		y_pred[np.where(y_pred < 0)] = -1
		y_pred[np.where(y_pred >= 0)] = 1
	return y_pred

#Verify the predicted y against the test y for cross validation data sets 
def verify_prediction(y_pred, y_test):
    y_diff = y_pred - y_test 		  # difference between the predicted and actual test values
    nFalse = len(y_diff[y_diff !=0 ]) # Number of elements whose difference is non-zero
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
