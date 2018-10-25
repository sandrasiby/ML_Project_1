import seaborn as sns
from proj1_helpers import *
from implementations import *
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter

def plot_hist_all_features(df, jetnum):

	fnames = list(df)

	for feature in fnames[1:]:
		f_sig = df.loc[df['Prediction'] == 's'][feature]
		f_sig = list(f_sig)
		f_bg = df.loc[df['Prediction'] == 'b'][feature]
		f_bg = list(f_bg)
		plt.title(feature)
		plt.hist(f_bg, bins=40, alpha=0.5, label="Background")
		plt.hist(f_sig, bins=40, alpha=0.5, label="Signal")
		plt.legend()
		plt.savefig(str(jetnum) + "/" + feature + ".png")
		plt.clf()
		#plt.show()

def plot_hist(df):

	mass_signal = df.loc[df['Prediction'] == 's']['DER_mass_MMC']
	mass_signal = list(mass_signal)
	#mass_signal = [x for x in mass_signal if x != -999]
	print(max(mass_signal))

	mass_bg = df.loc[df['Prediction'] == 'b']['DER_mass_MMC']
	#mass_bg = [x for x in list(mass_bg) if x != -999]
	print(max(mass_bg))

	plt.hist(mass_bg, bins=40, alpha=0.5, label="Background")
	plt.hist(mass_signal, bins=40, alpha=0.5, label="Signal")
	plt.legend()
	plt.show()

def get_df_by_jet(df, jetnum):

	df = df.loc[df['PRI_jet_num'] == jetnum]
	print(df.shape)
	if jetnum == 0:
		df = df.drop(['PRI_jet_num', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', \
		 'DER_prodeta_jet_jet', 'DER_lep_eta_centrality', 'PRI_jet_leading_pt', \
		 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',\
		 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi'], axis=1)
	elif jetnum == 1:
		df = df.drop(['PRI_jet_num', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', \
		 'DER_prodeta_jet_jet', 'DER_lep_eta_centrality', 'PRI_jet_subleading_pt',\
		 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi'], axis=1)
	elif jetnum == 2:
		df = df.drop(['PRI_jet_num'], axis=1)
	elif jetnum == 3:
		df = df.drop(['PRI_jet_num'], axis=1)
	else:
		print("Error: invalid jet number")
	print(df.shape)

	return df

def get_df_filtered(df, fl):

	df = df[df.columns.drop(list(df.filter(regex=fl)))]
	return df

def visualize_correlation(df):

	# calculate the correlation matrix
	corr = df.corr()

	# plot the heatmap
	ax = sns.heatmap(corr, 
		xticklabels=corr.columns,
		yticklabels=corr.columns)
	#ax.tick_params(axis='x', rotation=45)
	#xtexts = [t.get_text()  for t in ax.get_xticklabels()]
	# xtexts = [x[4:] for x in xtexts]
	#ax.set_xticklabels(xtexts, rotation='30')
	# ytexts = [t.get_text()  for t in ax.get_yticklabels()]
	# ytexts = [x[4:] for x in ytexts]
	# ax.set_xticklabels(xtexts)
	# ax.set_yticklabels(ytexts)
	plt.show()
	#plt.savefig("figs/pri_features_full_labels.png")

if __name__ == "__main__":

	training_data_path = "train.csv"

	df = pd.read_csv(training_data_path)
	
	#print(list(df))
	#df.Prediction.replace(['s', 'b'], [1, 0], inplace=True)
	df = df.drop(['Id'], axis=1)

	#filter_label = 'DER'
	#df = get_df_filtered(df, filter_label)
	jetnum = 2
	for jetnum in range(0, 4):
		df1 = get_df_by_jet(df, jetnum)
		plot_hist_all_features(df1, jetnum)

	#plot_hist(df)
	#visualize_correlation(df)

