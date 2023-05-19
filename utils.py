import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def null_value_rows_remove(df):
    df = df.dropna(axis=0)
    return df


def random_over_sampling(df, subject_name):
    b_plus_class = df[df[subject_name] == 'B+']
    c_plus_class = df[df[subject_name] == 'C+']
    a_class = df[df[subject_name] == 'A']
    b_class = df[df[subject_name] == 'B']
    b_minus_class = df[df[subject_name] == 'B-']

    a_minus_class = df[df[subject_name] == 'A-']

    a_plus_class = df[df[subject_name] == 'A+']
    c_minus_class = df[df[subject_name] == 'C-']
    c_class = df[df[subject_name] == 'C']
    d_class = df[df[subject_name] == 'D']

    count = df[subject_name].value_counts()[0]

    b_plus_class_over = b_plus_class.sample(count, replace=True)
    c_plus_class_over = c_plus_class.sample(count, replace=True)
    a_class_over = a_class.sample(count, replace=True)
    b_class_over = b_class.sample(count, replace=True)
    b_minus_class_over = b_minus_class.sample(count, replace=True)

    a_minus_class_over = a_minus_class.sample(count, replace=True)

    a_plus_class_over = a_plus_class.sample(count, replace=True)
    c_minus_class_over = c_minus_class.sample(count, replace=True)
    c_class_over = c_class.sample(count, replace=True)
    d_class_over = d_class.sample(count, replace=True)

    new_df = pd.concat([
        b_plus_class_over,
        c_plus_class_over,
        a_class_over,
        b_class_over,
        b_minus_class_over,

        a_minus_class_over,

        a_plus_class_over,
        c_minus_class_over,
        c_class_over,
        d_class_over
    ], axis=0)

    return new_df


def write_json(data, filename='data.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def label_encoder(df):
    encoder_result = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            encoder_result[col] = le_name_mapping

    encoder_result = {k: str(v) for k, v in encoder_result.items()}
    write_json(encoder_result)
    return df


def find_best_features(features, target, num_features):
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(features, target)

    feature_importances = pd.DataFrame(rfc.feature_importances_, index=features.columns,
                                       columns=['importance'])
    feature_importances.sort_values('importance', ascending=False, inplace=True)
    return feature_importances.index[:num_features]


def save_sklearn_model(model, model_name):
    with open('./weights/' + model_name, 'wb') as f:
        pickle.dump(model, f)


def load_sklearn_model(model_name):
    with open('./weights/' + model_name, 'rb') as f:
        model = pickle.load(f)
    return model


def save_list_to_file(list_name, file_name):
    with open(file_name, 'w') as file:
        for item in list_name:
            file.write(str(item) + '\n')


def read_list_from_file(file_name):
    with open(file_name, 'r') as file:
        list_name = file.read().splitlines()
    return list_name


def read_json(file_name):
    with open(file_name) as f_obj:
        return json.load(f_obj)


def change_to_dict(x):
    x = x.replace('{', '')
    x = x.replace('}', '')
    x = x.replace('\'', '')
    x = x.split(',')
    x = [i.split(':') for i in x]
    x = {i[0]: int(i[1]) for i in x}
    return x


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key
