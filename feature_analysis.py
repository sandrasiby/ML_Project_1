import seaborn as sns
from proj1_helpers import *
from implementations import *
import pandas as pd
from matplotlib import pyplot as plt

def visualize_correlation(df):

	# calculate the correlation matrix
	#df_der = df[df.columns.drop(list(df.filter(regex='DER')))]
	corr = df.corr()

	# plot the heatmap
	sns.heatmap(corr, 
		xticklabels=corr.columns,
		yticklabels=corr.columns)
	plt.show()

if __name__ == "__main__":

	training_data_path = "train.csv"
	# print("Read training data: START")
	# training_y, training_tx, training_ids = load_csv_data(training_data_path, sub_sample=True)
	# print("Read training data: DONE")

	df = pd.read_csv(training_data_path)
	visualize_correlation(df)

