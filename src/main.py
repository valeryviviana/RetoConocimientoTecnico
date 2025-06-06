import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from models_predicts import predict

app = FastAPI()


class PatientData(BaseModel):
    Clinical_Note: str = Field(..., alias="Clinical Note")
    Age: int
    Sex: str


class PatientDataList(BaseModel):
    data: List[PatientData]


@app.post("/predict")
async def predict2(data_list: PatientDataList):
    # Convertir la lista de datos a DataFrame
    df = pd.DataFrame([d.dict(by_alias=True) for d in data_list.data])

    # Ejecutar predicción
    predictions = predict(df)

    results = []
    for record, pred in zip(data_list.data, predictions):
        results.append({
            "Clinical Note": record.Clinical_Note,
            "Age": record.Age,
            "Sex": record.Sex,
            "Prediction": pred
        })

    return {"results": results}
