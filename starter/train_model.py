"""
Script to train machine learning model.
"""
# Add the necessary imports
import joblib
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()


# Add code to load in the data.
logger.info('Reading census clean data..')
data = pd.read_csv('data/census_clean.csv')

# train-test split.
logger.info(f'Splitting data to train and test (test_size=0.2)..')
train, test = train_test_split(data, test_size=0.20)

# Process the train data.
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

logger.info(f'Processing train data..')
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data.
logger.info(f'Processing test data..')
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train a model.
logger.info('Training a model..')
model = train_model(X_train, y_train)

# Score the model.
logger.info('Scoring the model in test data..')
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logger.info(f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta(1): {fbeta: .2f}")

# Save the model.
model_path = 'model/model.pkl'
encoder_path = 'model/encoder.pkl'
lb_path = 'model/lb.pkl'
logger.info('Saving the Model, OneHotEncoder and LabelBinarizer inside model/ folder..')
joblib.dump(model, model_path)
joblib.dump(encoder, encoder_path)
joblib.dump(lb, lb_path)

