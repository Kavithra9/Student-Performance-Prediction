import streamlit as st
import numpy as np
import pandas as pd
import json
from utils import read_list_from_file, load_sklearn_model, read_json, change_to_dict, get_key

best_features = read_list_from_file('features_columns.txt')
encoded_data = read_json('data.json')

user_inputs = []

model_name = "random_forest"
subject_name = '29. Results (IS3440)'

st.header("Result prediction of IS3440 ")

for i in best_features:
    if i in encoded_data.keys():
        encod_dict = change_to_dict(encoded_data[i])
        option = st.selectbox(i, tuple(encod_dict.keys()))
        user_inputs.append(encod_dict[option])

form = st.form(key='my-form')

submit = form.form_submit_button('Submit')

if submit:
    print(user_inputs)
    clf = load_sklearn_model(model_name)

    res = clf.predict(np.expand_dims(np.array(user_inputs), axis=0))

    print(res)

    result_dict = change_to_dict(encoded_data[subject_name])

    encoded_value = get_key(result_dict, res)

    print(encoded_value)

    st.write("Your result will be "+encoded_value)

