import asyncio
#from dataprep.datasets import load_dataset
#from dataprep.eda import create_report
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport



import streamlit as st
import os
import json
#import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
# visualize=Data_visualization()
# remove_waste_cols = removing_unwanted_columns()
## 1)Get the data from the user
## 2)Removing the unwanted columns

# remove the unwanted columns is not None

# remove the unwanted columns

# save the data to the data folder and UI for user folders

# remove the unwanted columns is None

# save the data to the data folder and UI for user folders


## 3)Select the target column and problem_type.


## 4) Selecting Metrics

# if the problem type is regression

# Display only R2 score

# if the problem type is classification

# Display "auc","f1","Precision","Recall"
### EDA
# user have to select the EDA type
# if the EDA type is 1 0r 2
# report will be generated as html file and stored in the UI folder
# visualize as html file for user when he click visualize button
# if the EDA type is 3
# It automatically visualizes
#streamlite heading
st.header("All in One ML")
datafile = st.file_uploader("Upload CSV", type=['csv'])


def save_uploadedfile(uploadedfile):
    with open(os.path.join("data", "data.csv"), "wb") as f:
        f.write(uploadedfile.getbuffer())


def save_data_files(df, filename, directory):
    if os.path.isdir(directory):
        if os.path.isfile(directory + '/' + filename + '.csv'):
            os.remove(directory + '/' + filename + '.csv')
        df.to_csv(directory + '/' + filename + '.csv', index=None, header=True)
    else:
        os.makedirs(directory)
        df.to_csv(directory + '/' + filename + '.csv', index=None, header=True)


def remove_unwanted_columns(data, waste_columns, cols_remove):
    directory1 = 'Data_files'
    directory2 = 'UI for user'

    filename = 'preprocessed_data'

    if cols_remove == 'no':
        save_data_files(df, filename, directory1)
        save_data_files(df, filename, directory2)
    else:
        data = data.drop(waste_columns, axis=1)
        save_data_files(data, filename, directory1)
        save_data_files(data, filename, directory2)


if datafile is not None:


    file_details = {"FileName": datafile.name, "FileType": datafile.type}
    df = pd.read_csv(datafile)
    profile = ProfileReport(df, title="Pandas Profiling Report")
    report = profile.to_file("UI for user/report.html")
    if st.button('Visualize data'):
        # load the report.html file and display it
        st.header("EDA")

        HtmlFile = open("UI for user/report.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        # print(source_code)
        components.html(source_code, height=800, width=800, scrolling=True)
    #report = create_report(df)
    #report.to_html('UI for user/report.html')

    st.dataframe(df)
    save_uploadedfile(datafile)
    df = pd.read_csv('data/data.csv')
    cols = df.columns
    unwanted_columns = st.multiselect('Select the columns to remove', cols)
    if unwanted_columns is not None:
        remove_cols = 'yes'
    else:
        remove_cols = 'no'

    print(remove_cols)
    target_variable = st.selectbox('Select the column to be used as the target variable', cols)
    remove_unwanted_columns(df, unwanted_columns, remove_cols)
    problem_type = st.selectbox('Select the problem type', ['Classification', 'Regression'])

    if problem_type == 'Classification':
        metrics = st.multiselect('Select the metrics to be used for evaluation', ['auc', 'f1', 'Precision', 'Recall'])
    else:
        metrics = st.multiselect('Select the metrics to be used for evaluation', ['R2'])

    # store target column, problem type and metrics in a json file
    try:
        with open('data/data.json', 'w') as f:
            # clear the data in the json file
            f.write('')
            json.dump({"target_variable": target_variable, "problem_type": problem_type, "metrics": metrics[0]}, f)
        with open('UI for user/data.json', 'w') as f:
            f.write('')
            json.dump({"target_variable": target_variable, "problem_type": problem_type, "metrics": metrics[0]}, f)
        # create a submit button
    except Exception as e:
        print(e)

    if st.button('Train'):
        # load the json file
        with open('data/data.json', 'r') as f:
            jss = json.load(f)
        target_variable = jss['target_variable']
        problem_type = jss['problem_type']
        metrics = jss['metrics']
        df = pd.read_csv('Data_files/preprocessed_data.csv')

        # js=pd.read_json('data/data.json')
        # print(js)
        target_variable1 = jss['target_variable']
        problem_type1 = jss['problem_type']
        metrics1 = jss['metrics']
        df1 = pd.read_csv('Data_files/preprocessed_data.csv')

        #import evalml
        from evalml.automl import AutoMLSearch

        X = df.drop([target_variable1], axis=1)
        y = df[target_variable1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       # X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y, problem_type=problem_type1)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if problem_type1 == 'Classification':
            automl_auc = AutoMLSearch(X_train=X_train, y_train=y_train,
                                      problem_type=problem_type1,
                                      objective=str(metrics1),
                                      additional_objectives=['f1', 'precision', 'recall', 'auc'],
                                      max_batches=1,
                                      optimize_thresholds=True)
        else:
            automl_auc = AutoMLSearch(X_train=X_train, y_train=y_train,
                                      problem_type=problem_type1,
                                      objective=str(metrics1),
                                      additional_objectives=["R2", "Root Mean Squared Error", "MaxError", "MedianAE",
                                                             "MSE",
                                                             "MAE"],
                                      max_batches=1,
                                      optimize_thresholds=True)
        automl_auc.search()
        best_pipeline = automl_auc.best_pipeline
        st.write(automl_auc.describe_pipeline(automl_auc.rankings.iloc[0]["id"],return_dict=True))  # describe pipeline
        st.write(automl_auc.rankings)  # ranking of pipelines
        best_pipeline.save('Data_files/model.pkl')
        best_pipeline.save('UI for user/model.pkl')
        if problem_type1 == 'Classification':
            st.write(
                best_pipeline.score(X_test, y_test, objectives=["auc", "f1", "Precision", "Recall"]))
        else:
            st.write(best_pipeline.score(X_test, y_test, objectives=["R2", "Root Mean Squared Error", "MaxError"]))
        #st.write(automl_auc.best_pipeline)  # best pipeline

    if st.button('Download Model'):
        st.download_button('UI for user/model.pkl')
    if st.button('Download EDA Report'):
        st.download_button('UI for user/report.html')










