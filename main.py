from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import joblib
from model import StudentModel

app = FastAPI()

# Load scaler and model
scaler = joblib.load("scaler.pkl")
input_dim = len(scaler.mean_)
model = StudentModel(input_dim)
model.load_state_dict(torch.load("student_model.pt", map_location=torch.device("cpu")))
model.eval()


# Define input schema
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


@app.post("/predict")
def predict(input: StudentInput):
    input_data = np.array([list(input.dict().values())])
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return {"predicted_G3": round(prediction, 2)}
