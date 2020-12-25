from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer

def column_change(encoder, input_columns, change_feature_map):
    """
    Get change of the encoder which changed the columns
    :param encoder: encoder
    :param input_columns: the columns before chenged
    :param change_feature_map: changed map of features
    :return: the changed cloumns ,and feature map
    """

    final_cols = list()
    if isinstance(encoder, OneHotEncoder):
        for i in range(len(input_columns)):
            tmp_cols = ['%s_%s' % (input_columns[i], x) for x in encoder.categories_[i]]
            change_feature_map[input_columns[i]] = tmp_cols
            final_cols.extend(tmp_cols)
    elif isinstance(encoder, KBinsDiscretizer) and (
            encoder.encode == 'onehot' or encoder.encode == 'onehot-dense'):
        k = 1
        if isinstance(encoder, int):
            k = encoder.n_bins
        if isinstance(encoder, list) or isinstance(encoder, tuple):
            k = len(encoder.n_bins)
        for col in input_columns:
            tmp_cols = list()
            for i in range(k):
                tmp_cols.append('%s_%s' % (col, i))
            change_feature_map[col] = tmp_cols
            final_cols.extend(tmp_cols)
    else:
        final_cols = input_columns
    return final_cols, change_feature_map