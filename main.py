import numpy as np
import pandas as pd

## the block for importing DL related packages.
import tensorflow as tf

## the block for importing Sklearn related packages
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from scipy import stats

from copy import deepcopy
import random

import os
import os.path as op
import gc
import sys

## load the model definition
from feature_extraction_backbone import ModelSettings
from feature_extraction_backbone import start_training_process, training_process_caller

## use this line to debug with numpy deprecation warning.
np.warnings.filterwarnings('error', category = np.VisibleDeprecationWarning)


## toggle if disable the CUDA device is necessary.
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class DataProcessor:
    def __init__(self, pattern_val, server_val, user_ratio_val, split_ratio_val):
        print('initiate DataProcessor')

        pattern_val = int(pattern_val)
        print('the pattern being accessed is', str(pattern_val))

        self.root_dir = op.abspath(os.sep)

        self.extend_load_bs = 7200
        self.limited_load_bs = 3600

        # self.da_iter_val = da_iter_val

        if server_val == 'featurize' or server_val.startswith('f'):
            self.batch_size = self.extend_load_bs
            self.workspace_dir = str('/home/featurize/data/')
        else:
            self.batch_size = self.limited_load_bs
            self.workspace_dir = str(r'./dataset/d1/full')

        self.CSV_FILE_PATH = op.join(self.workspace_dir + '/' +
                                     str(pattern_val) + '_train_proc.csv')
        self.VALID_FILE_PATH = op.join(self.workspace_dir + '/' +
                                       str(pattern_val) + '_valid_proc.csv')

        print('the data being used is under', self.workspace_dir)

        np.random.seed(13)
        tf.random.set_seed(1313)

        user_ratio_val = float(user_ratio_val)
        self.load_data(user_ratio_val, split_ratio_val)

        ## reshaping the data
        self.train_x, self.train_y = self.data_shaper(self.train_x, self.train_y,
                                                      self.timestep_val, self.timestep_val)
        self.test_x, self.test_y = self.data_shaper(self.test_x, self.test_y,
                                                    self.timestep_val, self.timestep_val)
        ## doing the one-hot encoding here.
        self.train_y_oh = self.ohenc.transform(self.train_y)
        self.test_y_oh = self.ohenc.transform(self.test_y)

        self.train_data_shape = self.train_x.shape
        self.train_label_shape = self.train_y_oh.shape

        print('the shape of train set is', self.train_data_shape, self.train_label_shape)
        print('the shape of test set is', self.test_x.shape, self.test_y_oh.shape)

    def __del__(self):
        print('the instance of DataProcessor has been recycled. ')

        record_dir = './parameters/'
        if not os.path.exists(record_dir):
            os.mkdir(record_dir)

        feature_col_arr = np.array(self.feature_col).reshape(1, -1)
        feature_col_df = pd.DataFrame(feature_col_arr)
        feature_col_df.to_csv(record_dir + 'feature_col.csv', index = False)

    def load_data(self, user_ratio_val, split_ratio_val):
        data = pd.read_csv(self.CSV_FILE_PATH)

        ## in this sub function, the data is loaded, o-h encoded and normalized
        ## we use dedicated structure to ensure that padding values will not be scaled here.
        ## and, in case necessary, we persistent this normalizer for later use.
        ## use another function for data shaping.

        ## get the timestep length
        self.timestep_val = data.loc[((data['tag'] == 0) & (data['user'] == 0)
                                      & (data['input'] == 1)), :].shape[0]

        ## separate the labels apart from the data content
        feature_col = data.columns.to_list()
        feature_remove_list = ['tag', 'user', 'input', 'size', 'oriX', 'oriY', 'oriZ', 'rvX', 'rvY', 'rvZ', 'scalar']
        feature_col = [ele for ele in feature_col if ele not in feature_remove_list]
        # print(feature_col)
        self.feature_col = feature_col

        print('scaling the data. ')
        self.scaler = MinMaxScaler()
        scaler_mask = np.any(data.loc[:, self.feature_col].values, -1)
        # self.scaler = self.scaler.fit(data_content.values[scaler_mask])
        ## avoid repeat call the function.
        self.scaler.fit(data.loc[:, self.feature_col].values[scaler_mask])
        data.loc[scaler_mask, self.feature_col] = self.scaler.transform(
            data.loc[:, self.feature_col].values[scaler_mask])

        ## create an instance of one_hot encoder.
        self.ohenc = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
        # self.ohenc = self.ohenc.fit(data['user'].values.reshape(-1, 1))
        ## avoid repeat call the function.
        self.ohenc.fit(data['user'].values.reshape(-1, 1))

        ## separate one user aside to simulate the transfer learning protocol.
        if bool(user_ratio_val):
            print('the dataset will be split to simulate the transfer learning protocol. ')
            ## based on the user_chosen_ratio_var, the dataset would thus be trimmed.
            user_chosen = random.sample(range(0, int(data['user'].max() + 1)),
                                        int(user_ratio_val * (data['user'].max() + 1)))
            print('the data from user', sorted(user_chosen, reverse = False),
                  'will be used for training. ')

        else:
            print('user trimming is not activated. ')

        ## splitting train set and test set
        remaining_tag = np.array(sorted(data['tag'].value_counts().index, reverse = False))
        trainset_chosen = random.sample(list(remaining_tag),
                                        int(split_ratio_val * remaining_tag.shape[0]))
        self.trainset_chosen = trainset_chosen
        print('the train set will include', int(split_ratio_val * remaining_tag.shape[0]),
              'tags. ')
        ## generating the set
        # self.train_x = data.loc[data['tag'].isin(trainset_chosen), self.feature_col]
        # self.train_y = data.loc[data['tag'].isin(trainset_chosen), 'user']
        # self.test_x = data.loc[~data['tag'].isin(trainset_chosen), self.feature_col]
        # self.test_y = data.loc[~data['tag'].isin(trainset_chosen), 'user']
        train_set = data.loc[data['tag'].isin(trainset_chosen), :]
        test_set = data.loc[~data['tag'].isin(trainset_chosen), :]
        if bool(user_ratio_val):
            self.train_x = train_set.loc[train_set['user'].isin(user_chosen), self.feature_col]
            self.train_y = train_set.loc[train_set['user'].isin(user_chosen), 'user']
            self.test_x = test_set.loc[test_set['user'].isin(user_chosen), self.feature_col]
            self.test_y = test_set.loc[test_set['user'].isin(user_chosen), 'user']
        else:
            self.train_x = train_set.loc[:, self.feature_col]
            self.train_y = train_set.loc[:, 'user']
            self.test_x = test_set.loc[:, self.feature_col]
            self.test_y = test_set.loc[:, 'user']

        ## rearranging the columns so as for a better split in the keras model
        re_col = ['siteX', 'siteY', 'baro', 'laccX', 'laccY', 'laccZ', 'gyrX', 'gyrY', 'gyrZ']
        self.train_x = self.train_x[re_col]
        self.test_x = self.test_x[re_col]
        del data, train_set, test_set
        gc.collect()

    def data_shaper(self, x, y, time_steps, stride):
        xs, ys = [], []
        for i in range(0, len(x), stride):
            v = x.iloc[i:(i + time_steps)].values
            labels = y.iloc[i: i + time_steps]
            xs.append(v)
            ys.append(stats.mode(labels)[0][0])
        return np.array(xs), np.array(ys).reshape(-1, 1)

    def load_data_rep_ex(self):
        ## the data loaded here are for the representation extraction and transfer learning.
        ## representation learning requires to separate the label from the untrimmed data part
        ## transfer learning requires to skip the user samples process.

        data = pd.read_csv(self.VALID_FILE_PATH)

        scaler_mask = np.any(data.loc[:, self.feature_col].values, -1)
        ## directly call the trained data scaler.
        data.loc[scaler_mask, self.feature_col] = self.scaler.transform(
            data.loc[:, self.feature_col].values[scaler_mask])

        ## making instance of data for representation extraction.
        # row_limit = int(data.shape[0] / (self.da_iter_val + 1))
        row_limit = int(data.shape[0] / 2)
        self.rep_ex_feature = data.iloc[:row_limit, :].loc[:, self.feature_col]
        self.rep_ex_label = data.iloc[:row_limit, :].loc[:, 'user']
        ## also they need reshape
        self.rep_ex_feature, self.rep_ex_label = self.data_shaper(
            self.rep_ex_feature, self.rep_ex_label,
            self.timestep_val, self.timestep_val)

        del data
        gc.collect()

    def load_data_tr_lr(self):
        ## this function is for the purpose of untrimmed data for transfer learning.
        ## we would first delete the vals of self.train_x etc. cause they not not needed in the protocol.
        del self.train_x, self.train_y, self.train_y_oh
        del self.test_x, self.test_y, self.test_y_oh
        gc.collect()

        data = pd.read_csv(self.CSV_FILE_PATH)

        scaler_mask = np.any(data.loc[:, self.feature_col].values, -1)
        ## directly call the trained data scaler.
        data.loc[scaler_mask, self.feature_col] = self.scaler.transform(
            data.loc[:, self.feature_col].values[scaler_mask])

        self.train_x_full = data.loc[data['tag'].isin(self.trainset_chosen), self.feature_col]
        train_y = data.loc[data['tag'].isin(self.trainset_chosen), 'user']
        self.test_x_full = data.loc[~data['tag'].isin(self.trainset_chosen), self.feature_col]
        test_y = data.loc[~data['tag'].isin(self.trainset_chosen), 'user']

        self.train_x_full, train_y = self.data_shaper(self.train_x_full, train_y,
                                                      self.timestep_val, self.timestep_val)
        self.test_x_full, test_y = self.data_shaper(self.test_x_full, test_y,
                                                    self.timestep_val, self.timestep_val)
        self.train_y_full_oh = self.ohenc.transform(train_y)
        self.test_y_full_oh = self.ohenc.transform(test_y)

        del train_y, test_y, data
        gc.collect()


if __name__ == '__main__':
    for current_pattern in ['90', '91', '92', '93']:
        ## setting the variables.
        user_ratio_val_in, split_ratio_val_in = 0.99, 0.8

        server_flag = 'local'

        dp = DataProcessor(current_pattern, server_flag,
                           user_ratio_val_in, split_ratio_val_in)
        rep_lr = ModelSettings(dp)
        ## the model is patented, and should only be used for academic purpose only
        ## if you are interested in getting the model
        ## please contact with me using email

        start_training_process(current_pattern, dp, rep_lr)

        del dp, rep_lr

        print(current_pattern, 'finished. ')

    # catch arguments from terminals. currently not enabled.
    commands = sys.argv[1:]
    pass
