import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference

cat_features = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
    ]

class IndividualInfo(BaseModel):
    age: int = Field(examples=[52])
    workclass: str = Field(examples=['Private'])
    fnlgt: int = Field(examples=[198863])
    education: str = Field(examples=['Prof-school'])
    education_num: int = Field(alias='education-num', examples=[15])
    marital_status: str = Field(alias='marital-status', examples=['Divorced'])
    occupation: str = Field(examples=['Exec-managerial'])
    relationship: str = Field(examples=['Not-in-family'])
    race: str = Field(examples=['White'])
    sex: str = Field(examples=['Male'])
    capital_gain: int = Field(alias='capital-gain', examples=[0])
    capital_loss: int = Field(alias='capital-loss', examples=[2559])
    hours_per_week: int = Field(alias='hours-per-week', examples=[60])
    native_country: str = Field(alias='native-country', examples=['United-States'])


# load model and data transformers
model = joblib.load('model/model.pkl')
encoder = joblib.load('model/encoder.pkl')
lb = joblib.load('model/lb.pkl')

# create app
app = FastAPI(title="Income Census Model Inference API",
              description="An Income Census Model Inference\
                API Application that is used for\
                inference given an individual information input.",
                version="1.0.0")


@app.get('/')
async def root():
    return {'message': 'Welcome to the Income Census Model Inference API'}


@app.post('/prediction')
async def predict(request: IndividualInfo):
    input_data = pd.DataFrame({k: v for k,v in request.model_dump(by_alias=True).items()}, index=[0])
    X, _, _, _ = process_data(
                    input_data, categorical_features=cat_features, label=None, training=False,
                    encoder=encoder, lb=lb
                )
    
    y_pred = inference(model, X)

    return {'Predicted Income': lb.inverse_transform(y_pred)[0]}

if __name__ == "__main__":
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)