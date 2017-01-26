# Utils for handling mixed type data

import pandas as pd
from pandas.core.categorical import CategoricalDtype
import numpy as np
from collections import OrderedDict
import cPickle
import gzip

from architecture.abstract.data_type import InputType, BroadInputType

# Ratio of train over all data
def separate_train_test(label, train_ratio, shuffle=True):
    if len(label.shape) > 1 and not (np.issubdtype(label.dtype, int) or np.issubdtype(label.dtype, np.object_)):
        raise ValueError("Only vectors of type int or object are supported")

    values, counts = np.unique(label, return_counts=True)

    value_map = {values[i]: i for i in xrange(len(values))}

    train_counts = train_ratio * counts
    train_fills = np.zeros(len(values), dtype=np.int32)

    indices = np.arange(len(label))
    if shuffle:
        np.random.shuffle(indices)

    train_indices = []
    test_indices = []

    for idx in indices:
        val_id = value_map.get(label[idx])
        if train_fills[val_id] < train_counts[val_id]:
            train_indices.append(idx)
            train_fills[val_id] += 1
        else:
            test_indices.append(idx)
    return train_indices, test_indices


def print_values_proportion(x, return_values=False):
    if len(x.shape) > 1 and not (np.issubdtype(x.dtype, int) or np.issubdtype(x.dtype, np.object_)):
        raise ValueError("Only vectors of type int or object are supported")
    values, counts = np.unique(x, return_counts=True)

    total_count = counts.sum()
    for i in range(len(values)):
        print "{0}: {1} ({2})".format(values[i], 1. * counts[i] / total_count, counts[i])

    if return_values:
        return values, 1. * counts/total_count


# Bring data in range [0,1]
# data is a numpy 2-D array
# Not good for outlier as it is brought to close to other data
# Input is numpy array
def normalize(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)

    pos_diff = col_max != col_min
    zero_diff = np.logical_and(col_max == col_min, col_max != 0)
    zero = np.logical_and(col_max == col_min, col_max == 0)

    # Apply over rows
    normalized_data = np.zeros(data.shape)
    normalized_data[:, pos_diff] =\
        (data[:, pos_diff] - col_min[pos_diff]) * 1.0/(col_max[pos_diff] - col_min[pos_diff])
    normalized_data[:, zero_diff] = 1.0
    normalized_data[:, zero] = 0.0
    # normalized_data = np.apply_along_axis(lambda x: 1.0*(x-col_min)/(col_max-col_min), 1, data)
    return normalized_data


# Bring data to have zero mean an unit variance
# Input is numpy array
# Account for the case of zero std
def standardize(data):

    col_mean = np.mean(data, axis=0)
    col_std = np.std(data, axis=0, ddof=1)
    # print "Length col mean is: " + str(len(col_mean))
    # print "Length col std is: " + str(len(col_std))

    new_data = np.zeros(data.shape)

    pos_std = col_std > 0
    zero_std = col_std == 0

    new_data[:, pos_std] = (data[:, pos_std] - col_mean[pos_std])/col_std[pos_std]
    new_data[:, zero_std] = data[:, zero_std] - col_mean[zero_std]

    return new_data

    # standardized_data = np.apply_along_axis(lambda x: 1.0*(x-col_mean)/col_std, 1, data)
    # return standardized_data


def log_count(data, base=None):
    if base is None:
        return np.log(1 + data)
    else:
        return np.log10(1 + data)/np.log10(base)
# The correct data type of a data frame for processing is below:
# Binary: boolean
# Categorical: category (pandas type: pandas.core.dtypes.CategoricalDtype)
# Poisson: integer
# Gaussian: float

# Note that binary and category can be grouped in the broad type of categorical
# Poisson and Gaussian can be grouped in the broad type of numeric

# If you load data from a mix source, you must write utility (handler)
# to turn them into this format

# cat_as_bin: Consider categorical (dummy values) as binary data
# poiss_as_gauss: Consider poisson as gaussian


def separate_clean_categorical(df):
    # Take the categorical data away from original data frame
    cat_col_ids = [i for i in range(len(df.dtypes)) if isinstance(df.dtypes[i], CategoricalDtype)]

    # No need to care about delete-all problem because it will results in empty data frame
    cat_df = df.ix[:, cat_col_ids]
    noncat_df = df.drop(df.columns[cat_col_ids], axis=1)

    return noncat_df, cat_df


# cat_map is a map with key is categorical column index(name)
# and value is an Index object that specify how categorical data is map to its code
# Save data_df, label_df, v_types, v_ranges
def process_clean_mixed_data(data_df, label_df=None, cat_as_bin=False, poiss_as_gauss=False,
                             save_file=None, as_matrix=False, use_gzip=False):
    # Dictionary with key is type and value is columns index

    noncat_df, cat_df = separate_clean_categorical(data_df)

    # noncat_type_dict = noncat_df.columns.to_series().groupby(df.dtypes).groups
    noncat_type_dict = noncat_df.columns.to_series().groupby(data_df.dtypes).groups

    bin_ranges = []
    gau_ranges = []
    cat_ranges = []
    poi_ranges = []

    # Non catgorical data
    for col_type, cols in noncat_type_dict.iteritems():
        # Binary data
        # print type(col_type)
        # print col_type
        print "Column type {0} includes {1}".format(col_type, cols)
        if np.issubdtype(col_type, np.bool_):
            # Dua data ve dang 0,1. No need to copy to new memeory region
            noncat_df.ix[:, cols] = noncat_df.ix[:, cols].astype(np.int32, copy=False)
            #bin_ranges.extend(cols)

            # We use column indices, not column names
            # Note that here we have to use noncat_df, not df
            col_ixs = [noncat_df.columns.get_loc(col) for col in cols]
            bin_ranges.extend(col_ixs)

        # Gaussian data
        elif np.issubdtype(col_type, float):
            #gau_ranges.extend(cols)
            col_ixs = [noncat_df.columns.get_loc(col) for col in cols]
            gau_ranges.extend(col_ixs)

        # Poisson data
        elif np.issubdtype(col_type, int):
            col_ixs = [noncat_df.columns.get_loc(col) for col in cols]
            if poiss_as_gauss:
                gau_ranges.extend(col_ixs)
            else:
                poi_ranges.extend(col_ixs)

        else:
            raise ValueError("Our mixed type architecture cannot recognize {}".format(col_type))

    # Handle categorical data
    new_col_id = len(noncat_df.columns)

    for col in cat_df.columns.values:
        dummy_df = pd.get_dummies(data_df.ix[:, col])

        if cat_as_bin: # Neu coi dummy cua cat nhu binary, add vao binary
            bin_ranges.extend(range(new_col_id, new_col_id + len(dummy_df.columns)))
        else:
            # Note that with cat_ranges we have to use append because we cannot merge them
            cat_ranges.append(range(new_col_id, new_col_id + len(dummy_df.columns)))

        # concat dummy_df to noncat_df (not cat_df)
        noncat_df = pd.concat([noncat_df, dummy_df], axis=1) # Append dummy_df vao noncat_df
        new_col_id += len(dummy_df.columns)

    # Tat ca dummy_df da duoc add vao noncat_df, ta co the vut cat_df di duoc
    v_types = []
    v_ranges = []

    if len(bin_ranges) > 0:
        v_types.append(InputType.binary)
        v_ranges.append(bin_ranges)

    if len(gau_ranges) > 0:
        v_types.append(InputType.gaussian)
        v_ranges.append(gau_ranges)

    # Type poisson
    if len(poi_ranges) > 0:
        v_types.append(InputType.poisson)
        v_ranges.append(poi_ranges)

    # Type categorical
    if len(v_ranges) > 0:
        for i in xrange(len(cat_ranges)):
            v_types.append(InputType.categorical)
            v_ranges.append(cat_ranges[i])

    if as_matrix:
        data = noncat_df.as_matrix()
        if label_df is not None:
            label = label_df.as_matrix()
    else:
        data = noncat_df
        label = label_df

    # If save_file is provided, save to file
    if save_file is not None:
        if use_gzip:
            with gzip.open(save_file, 'wb') as f:
                cPickle.dump((data, label, v_types, v_ranges), f) # Note here is noncat_df
        else:
            with open(save_file, 'wb') as f:
                cPickle.dump((data, label, v_types, v_ranges), f) # Note here is noncat_df

    return data, label, v_types, v_ranges


def load_rawdata_with_finetypes(file_or_df, input_types, last_class_col=False,
                                bin_map=None, cat_map=None, save_file=None, use_gzip=False, **kwargs):
    # If it is dataframe or series
    if isinstance(file_or_df, pd.DataFrame) or isinstance(file_or_df, pd.Series):
        df = file_or_df  # Gan luon vao df
    elif isinstance(file_or_df, str):
        df = pd.read_csv(file_or_df, **kwargs)
    else:
        TypeError('You should pass file or dataframe as input')

    if last_class_col:
        class_col = df.iloc[:, -1]
        df = df.drop(df.columns[-1], axis=1)  # Note that axis=1 for column dropped
    else:
        class_col = None

    if len(input_types) == len(df.columns):
        if bin_map is not None:
            print "Using previous mapping for binary values of this dataframe"
            binary_map = bin_map
        else:
            print "There is no previous mapping for binary values of this dataframe"
            binary_map = OrderedDict()

        if cat_map is not None:
            print "Using previous mapping for categorical values of this dataframe"
            category_map = cat_map
        else:
            print "There is no previous mapping for categorical values of this dataframe"
            category_map = OrderedDict()

        for i in xrange(len(input_types)):
            if input_types[i] == InputType.binary:
                unique_vals = df.iloc[:, i].unique()
                if len(unique_vals) <= 2:  # La binary
                    # Chuyen ve dang chuan binary
                    val = binary_map.get(df.columns.values[i])  # Gia tri de false
                    if val is None:
                        val = unique_vals[0]
                        # binary_map.update((df.columns.values[i], val))
                        binary_map[df.columns.values[i]] = val
                    # Gia tri false
                    df.iloc[:, i] = df.iloc[:, i].apply(lambda x: not (x == val))
                else:
                    raise ValueError("Column {} is binary but has more than 2 unique values.".format(i))

            elif input_types[i] == InputType.categorical:
                categories = category_map.get(df.columns.values[i])

                # Chuyen ve type cateogry
                df.iloc[:, i] = df.iloc[:, i].astype('category')

                if categories is None:  # category_map moi tao
                    # category_map.update((df.columns.values[i], df.iloc[:, i].cat.categories))
                    category_map[df.columns.values[i]] = df.iloc[:, i].cat.categories
                else:
                    df.iloc[:, i].cat.set_categories(categories, inplace=True)

            elif input_types[i] == InputType.gaussian:
                if not (np.issubdtype(df.dtypes[i], float)):
                    raise ValueError("Column {} is gaussian but does not have type float ({})".format(i, df.dtypes[i]))

            elif input_types[i] == InputType.poisson:
                if not (np.issubdtype(df.dtypes[i], int)):
                    raise ValueError("Column {} is poisson but does not have type int ({})".format(i, df.dtypes[i]))

    else:
        raise ValueError('Difference in length, broad types: {0} but columns: {1}'.format(
            len(input_types), len(df.columns)
        ))

    # Note that it must be binary_map and category_map (not bin_map and cat_map)
    result = (df, class_col, binary_map, category_map)
    if save_file is not None:
        if use_gzip:
            with gzip.open(save_file, 'wb') as f:
                cPickle.dump(result, f)
        else:
            with open(save_file, 'wb') as f:
                cPickle.dump(result, f)
    # Return clean dataframe va binary_map (value in each column that is assigned to true)
    return result

# Handle raw data to get clean data
# Parameters:
# - broad_types la mot list cac BroadInputType instances
# - bin_map la map voi key la binary column index (name), value la value trong column
# se duoc gan la true
# Dataframe duoc load tu file csv va khong co constraint gi dac biet cho data
def load_rawdata_with_broadtypes(file_or_df, broad_types, last_class_col=False, bin_map=None, cat_map=None,
                                 save_file=None, use_gzip=False, **kwargs):
    # If it is dataframe or series
    if isinstance(file_or_df, pd.DataFrame) or isinstance(file_or_df, pd.Series):
        df = file_or_df  # Gan luon vao df
    elif isinstance(file_or_df, str):
        df = pd.read_csv(file_or_df, **kwargs)
    else:
        TypeError('You should pass file or dataframe as input')

    if last_class_col:
        class_col = df.iloc[:, -1]
        df = df.drop(df.columns[-1], axis=1)  # Note that axis=1 for column dropped
    else:
        class_col = None

    if len(broad_types) == len(df.columns):
        if bin_map is not None:
            print "Using previous mapping for binary values of this dataframe"
            binary_map = bin_map
        else:
            print "There is no previous mapping for binary values of this dataframe"
            binary_map = OrderedDict()

        if cat_map is not None:
            print "Using previous mapping for categorical values of this dataframe"
            category_map = cat_map
        else:
            print "There is no previous mapping for categorical values of this dataframe"
            category_map = OrderedDict()

        for i in xrange(len(broad_types)):
            # Neu broad_types la categorical
            # Type chi co the la binary hoac categorical
            if broad_types[i] == BroadInputType.nominal:
                unique_vals = df.iloc[:, i].unique()
                if len(unique_vals) <= 2: # La binary
                    # Chuyen ve dang chuan binary
                    val = binary_map.get(df.columns.values[i]) # Gia tri de false
                    if val is None:
                        val = unique_vals[0]
                        #binary_map.update((df.columns.values[i], val))
                        binary_map[df.columns.values[i]] = val
                    # Gia tri false
                    df.iloc[:, i] = df.iloc[:, i].apply(lambda x: not(x == val))
                else: # La categorical
                    categories = category_map.get(df.columns.values[i])

                    # Chuyen ve type cateogry
                    df.iloc[:, i] = df.iloc[:, i].astype('category')

                    if categories is None:  # category_map moi tao
                        #category_map.update((df.columns.values[i], df.iloc[:, i].cat.categories))
                        category_map[df.columns.values[i]] = df.iloc[:, i].cat.categories
                    else:
                        df.iloc[:, i].cat.set_categories(categories, inplace=True)

            # Neu broad_types la numeric
            # Type chi co the la poisson hoac gaussian
            # Voi phan nay ta chi check xem co type nao lech khong thoi
            if broad_types[i] == BroadInputType.numeric:
                if not (np.issubdtype(df.dtypes[i], float) or np.issubdtype(df.dtypes[i], int)):
                    raise ValueError("Do not support type {0} for numeric data".format(df.dtypes[i]))
    else:
        raise ValueError('Difference in length, broad types: {0} but columns: {1}'.format(
            len(broad_types), len(df.columns)
        ))

    # Note that it must be binary_map and category_map (not bin_map and cat_map)
    result = (df, class_col, binary_map, category_map)

    if save_file is not None:
        if use_gzip:
            with gzip.open(save_file, 'wb') as f:
                cPickle.dump(result, f)
        else:
            with open(save_file, 'wb') as f:
                cPickle.dump(result, f)
    # Return clean dataframe va binary_map (value in each column that is assigned to true)
    return result


# Without broad_type, binary and category must have object type
# Ve sau neu object type la string co loai khac (datetime chang han),
# thi phai provide pattern de detect

# gaussian must have type float
# poisson must have type int
def load_rawdata_without_type(file_or_df, last_class_col=False, bin_map=None, cat_map=None,
                              save_file=None, use_gzip=False, **kwargs):
    # If it is dataframe or series
    if isinstance(file_or_df, pd.DataFrame) or isinstance(file_or_df, pd.Series):
        df = file_or_df  # Gan luon vao df
    elif isinstance(file_or_df, str):
        df = pd.read_csv(file_or_df, **kwargs)
    else:
        TypeError('You should pass file or dataframe as input')

    if last_class_col:
        class_col = df.iloc[:, -1]
        df = df.drop(df.columns[-1], axis=1)  # Note that axis=1 for column dropped
    else:
        class_col = None

    if bin_map is not None:
        print "Using previous mapping for binary values of this dataframe"
        binary_map = bin_map
    else:
        print "There is no previous mapping for binary values of this dataframe"
        binary_map = OrderedDict()

    if cat_map is not None:
        print "Using previous mapping for categorical values of this dataframe"
        category_map = cat_map
    else:
        print "There is no previous mapping for categorical values of this dataframe"
        category_map = OrderedDict()

    for i in xrange(len(df.columns)):
        # Neu broad_types la categorical
        # Type chi co the la binary hoac categorical
        if np.issubdtype(df.dtypes[i], np.object_):
            unique_vals = df.iloc[:, i].unique()
            if len(unique_vals) <= 2:
                val = binary_map.get(df.columns.values[i])
                if val is None:
                    val = unique_vals[0]
                    binary_map[df.columns.values[i]] = val
                # Return 1 neu khong giong, 0 neu giong gia tri mau
                df.iloc[:, i] = df.iloc[:, i].apply(lambda x: not (x == val))
            else:  # La categorical
                categories = category_map.get(df.columns.values[i])

                # Chuyen ve type cateogry
                df.iloc[:, i] = df.iloc[:, i].astype('category')

                if categories is None:  # category_map moi tao
                    # category_map.update((df.columns.values[i], df.iloc[:, i].cat.categories))
                    category_map[df.columns.values[i]] = df.iloc[:, i].cat.categories
                else:
                    df.iloc[:, i].cat.set_categories(categories, inplace=True)
        # Neu la int thi van co the la binary (0, 1)
        elif np.issubdtype(df.dtypes[i], int):
            unique_vals = df.iloc[:, i].unique()
            if len(unique_vals) <= 2 and abs(unique_vals.max()) <= 1: # La binary roi
                val = binary_map.get(df.columns.values[i])
                if val is None:
                    val = unique_vals[0]
                    binary_map[df.columns.values[i]] = val
                # Return 1 neu khong giong, 0 neu giong gia tri mau
                df.iloc[:, i] = df.iloc[:, i].apply(lambda x: not (x == val))
        else:
            if not np.issubdtype(df.dtypes[i], float):
                raise ValueError("Do not support type {0}".format(df.dtypes[i]))

    # Note that it must be binary_map and category_map (not bin_map and cat_map)
    result = (df, class_col, binary_map, category_map)

    if save_file is not None:
        if use_gzip:
            with gzip.open(save_file, 'wb') as f:
                cPickle.dump(result, f)
        else:
            with open(save_file, 'wb') as f:
                cPickle.dump(result, f)
    # Return clean dataframe va binary_map (value in each column that is assigned to true)
    return result

# Separate hierarchical multi-labeled data into different level
