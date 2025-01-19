from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    """ Test the root welcome endpoint """
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {'message': 'Welcome to the Income Census Model Inference API'}


def test_prediction_above():
    """ Test prediction endpoint output where salary (target feature) is >50k """

    json_data = {
        'age': 53,
        'workclass': 'Private',
        'fnlgt': 126592,
        'education': 'Some-college',
        'education-num': 10,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Craft-repair',
        'relationship': 'Husband',
        'race': 'Black',
        'sex': 'Male',
        'capital-gain': 7688,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'
        }

    response = client.post('/prediction', json=json_data)

    assert response.status_code == 200
    assert response.json() == {'Predicted Income': '>50K'}

def test_prediction_below():
    """ Test prediction endpoint output where salary (target feature) is <=50K """

    json_data = {
        'age': 54,
        'workclass': 'Local-gov',
        'fnlgt': 163557,
        'education': 'HS-grad',
        'education-num': 9,
        'marital-status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Female',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 30,
        'native-country': 'United-States'
        }

    response = client.post('/prediction', json=json_data)

    assert response.status_code == 200
    assert response.json() == {'Predicted Income': '<=50K'}