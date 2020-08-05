import os
import argparse
from sklearn import model_selection
import pandas as pd

def get_args():
	parser=argparse.ArgumentParser('Folds generator')
	parser.add_argument('-f', '--csv_filename', type=str, default=None, help='csv file path')
	parser.add_argument('-t', '--target_column', type=str, default=None, help='target column for kfold')
	parser.add_argument('s', '--n_splits', type=int, default=5, help='number of kfold')
	args=parser.parse_args()
	return args

def create_folds(opt):
	df=pd.read_csv(os.path.join('../input/', opt.filename))
	df['kfold']=-1
	df=df.sample(frac=1).reset_index(drop=True)
	y=df[opt.target_column]
	kf=model_selection.StratifiedKFold(n_splits=opt.n_splits)
	for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
		df.loc[v_, 'kfold']=f

	filename=opt.filename.split('.')[0]+'_folds.csv'
	df.to_csv(os.path.join('../input/', filename), index=False)

if __name__=='__main__':
	opt=get_args()
	create_folds(opt)