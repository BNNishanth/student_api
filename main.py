from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
import joblib
from model import StudentModel

app = FastAPI()

# Load scaler, trained column names, and model
scaler = joblib.load("scaler.pkl")
trained_columns = joblib.load("trained_columns.pkl")
input_dim = len(trained_columns)
model = StudentModel(input_dim)
model.load_state_dict(torch.load("student_model.pt", map_location=torch.device("cpu")))
model.eval()


# Define input schema (only the fields you're accepting)
class StudentInput(BaseModel):
    age: int
    Medu: int
    Fedu: int
    traveltime: int
    studytime: int
    failures: int
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    absences: int
    guardian_mother: int
    guardian_other: int
    schoolsup_yes: int
    famsup_yes: int
    paid_yes: int
    activities_yes: int
    nursery_yes: int
    higher_yes: int
    internet_yes: int
    romantic_yes: int
    G1: int
    G2: int
    school_GP: int
    school_MS: int
    sex_F: int
    sex_M: int
    address_U: int
    address_R: int
    famsize_LE3: int
    famsize_GT3: int
    Pstatus_T: int
    Pstatus_A: int
    Mjob_at_home: int
    Mjob_health: int
    Mjob_other: int
    Mjob_services: int
    Mjob_teacher: int
    Fjob_at_home: int
    Fjob_health: int
    Fjob_other: int
    Fjob_services: int
    Fjob_teacher: int
    reason_course: int
    reason_home: int
    reason_other: int
    reason_reputation: int


@app.post("/predict")
def predict(input: StudentInput):
    try:
        input_dict = input.dict()
        input_df = pd.DataFrame([input_dict])

        # Ensure all columns from training are present
        for col in trained_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Align column order to match training
        input_df = input_df[trained_columns]

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor).item()

        return {"predicted_G3": round(prediction, 2)}

    except Exception as e:
        return {"error": str(e)}
