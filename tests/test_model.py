
import pytest
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import inference, compute_model_metrics

@pytest.fixture(scope="module")
def data():
    df = pd.read_csv("data/census_clean.csv")
    return df

@pytest.fixture(scope="module")
def model():
    model = joblib.load('model/model.pkl')
    return model

@pytest.fixture(scope="module")
def data_transformers():
    encoder = joblib.load('model/encoder.pkl')
    lb = joblib.load('model/lb.pkl')
    return encoder, lb

@pytest.fixture(scope="module")
def train_test_data(data):
    train, test = train_test_split(data, test_size=0.20)
    return train, test

@pytest.fixture(scope="module")
def cat_features():
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]
    return cat_features

@pytest.fixture(scope="module")
def x_y_split_process(train_test_data, data_transformers, cat_features):
    train, test = train_test_data
    encoder, lb = data_transformers
    
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )

    return X_train, y_train, X_test, y_test



def test_data(data):
    """
    Test if the data is empty or not
    """
    assert data.shape[0] > 0


def test_process_data(train_test_data, data_transformers, cat_features):
    """
    Test if process data split X & y with same the shape
    """
    encoder, lb = data_transformers
    train, _ = train_test_data

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )

    assert X_train.shape[0] == y_train.shape[0]


def test_compute_model_metrics(model, x_y_split_process):
    """
    Test if model metrics is computed 
    """
    _, _, X_test, y_test = x_y_split_process
    y_preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

    assert precision is not None
    assert recall is not None
    assert fbeta is not None


def test_inference(model, x_y_split_process):
    """
    Test if the model inference return the expected output shape
    """
    _, _, X_test, y_test = x_y_split_process
    
    y_preds = inference(model, X_test)
    print(y_test.shape)
    assert y_test.shape == y_preds.shape
