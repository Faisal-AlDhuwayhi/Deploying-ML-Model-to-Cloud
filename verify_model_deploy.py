import requests


input_data = {
    "age": 53,
    "workclass": "Private",
    "fnlgt": 126592,
    "education": "Some-college",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Craft-repair",
    "relationship": "Husband",
    "race": "Black",
    "sex": "Female",
    "capital-gain": 7688,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
    }

url = 'https://deploying-ml-model-to-cloud-cb3a15297af5.herokuapp.com/prediction'

response = requests.post(url, json=input_data)

print('Response Status Code:', response.status_code)
print('Response Output (model inference):', response.json())