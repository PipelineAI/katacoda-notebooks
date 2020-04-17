#!/usr/bin/env python
# coding: utf-8

'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import pandas as pd
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

import mlflow
import mlflow.tensorflow
from mlflow import pyfunc


# # Setup Experiment Tracker

tracking_uri='file:///root/mlflow'
mlflow.set_tracking_uri(tracking_uri)

experiment_name = 'boston'
mlflow.set_experiment(experiment_name)  

import mlflow.tensorflow
mlflow.tensorflow.autolog()


# # Import Training Data 

# Builds, trains and evaluates a tf.estimator. Then, exports it for inference, logs the exported model
# with MLflow, and loads the fitted model back as a PyFunc to make predictions.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
# There are 13 features we are using for inference.
feat_cols = [tf.feature_column.numeric_column(key="features", shape=(x_train.shape[1],))]
feat_spec = {
    "features": tf.placeholder("float", name="features", shape=[None, x_train.shape[1]])}
hidden_units = [50, 20]
steps = 1000
regressor = tf.estimator.DNNRegressor(hidden_units=hidden_units, feature_columns=feat_cols)
train_input_fn = tf.estimator.inputs.numpy_input_fn({"features": x_train}, y_train,
                                                    num_epochs=None, shuffle=True)


# # Start Training Run

with mlflow.start_run() as run:
    mlflow.log_param("Hidden Units", hidden_units)
    mlflow.log_param("Steps", steps)
    regressor.train(train_input_fn, steps=steps)
    test_input_fn = tf.estimator.inputs.numpy_input_fn({"features": x_test}, y_test,
                                                       num_epochs=None, shuffle=True)
    # Compute mean squared error
    mse = regressor.evaluate(test_input_fn, steps=steps)
    mlflow.log_metric("Mean Square Error", mse['average_loss'])
    # Building a receiver function for exporting
    receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feat_spec)
    
    saved_estimator_path = regressor.export_savedmodel('./saved-models/boston', receiver_fn).decode("utf-8")
    # Logging the saved model
    mlflow.tensorflow.log_model(tf_saved_model_dir=saved_estimator_path,
                                tf_meta_graph_tags=[tag_constants.SERVING],
                                tf_signature_def_key="predict",
                                artifact_path="model")    


# # Predict with the Model

# Resume the run by passing run id from above
with mlflow.start_run(run_id=run.info.run_id) as run:
    # Reload the model and predict
    pyfunc_model = mlflow.pyfunc.load_model(mlflow.get_artifact_uri('model'))

    # Predict with the loaded Python Function
    df = pd.DataFrame(data=x_test, columns=["features"] * x_train.shape[1])
    predict_df = pyfunc_model.predict(df)
    predict_df['original_labels'] = y_test
    print(predict_df)
