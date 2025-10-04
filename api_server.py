from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI(title="Little Grow ML API")

# เพิ่ม CORS เพื่อให้ Flutter เรียกใช้ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ในการใช้งานจริงควรระบุ origin ที่แน่นอน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดลและเครื่องมือ
try:
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    model = joblib.load("rf_weight_height.pkl")
    print("✅ โหลดโมเดลสำเร็จ")
except Exception as e:
    print(f"❌ ไม่สามารถโหลดโมเดล: {e}")
    print("กรุณาตรวจสอบว่าไฟล์ scaler.pkl, encoders.pkl, rf_weight_height.pkl อยู่ในโฟลเดอร์เดียวกัน")
    model = None


class PredictionRequest(BaseModel):
    weight: float
    height: float
    birth_date: str
    measurement_date: str
    gender: str
    birth_weight: float = 3000
    breastfeeding_duration: int = 6


@app.get("/")
def root():
    return {
        "message": "Little Grow ML API",
        "model_loaded": model is not None,
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict(data: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="โมเดลยังไม่พร้อมใช้งาน - กรุณาตรวจสอบไฟล์ .pkl"
        )

    try:
        # คำนวณอายุเป็นเดือน
        birth = datetime.fromisoformat(data.birth_date.replace("Z", "+00:00"))
        measurement = datetime.fromisoformat(
            data.measurement_date.replace("Z", "+00:00")
        )
        age_months = (measurement - birth).days / 30.44

        # เตรียมฟีเจอร์ทั้งหมด (ต้องตรงกับตอนเทรนโมเดล)
        features = {
            "Gender": 0 if data.gender.lower() == "male" else 1,
            "Age/month": age_months,
            "Birth Weight (grams)": data.birth_weight,
            "Weight (kg)": data.weight,
            "Height (cm)": data.height,
            "Pregnancy Complications": 0,
            "Breastfeeding Duration": data.breastfeeding_duration,
            "Birth Order": 1,
            "Father's Age": 30,
            "Mother's Age": 28,
            "Father's Occupation": 1,
            "Mother's Occupation": 1,
            "Father's Smoking Status": 0,
            "Mother's Smoking Status": 0,
            "Father's Alcohol Consumption": 0,
            "Mother's Alcohol Consumption": 0,
            "Father's Substance Use": 0,
            "Mother's Substance Use": 0,
            "Child's Living Situation": 1,
        }

        # แปลงเป็น DataFrame
        X = pd.DataFrame([features])

        # Encode categorical features (ถ้ามี)
        for col, encoder in encoders.items():
            if col in X.columns and X[col].dtype == "object":
                try:
                    X[col] = encoder.transform(X[col].astype(str))
                except Exception as e:
                    print(f"Warning: ไม่สามารถ encode {col}: {e}")

        # Scale features
        X_scaled = scaler.transform(X)

        # ทำนาย
        prediction = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0].tolist()

        # แมปผลลัพธ์เป็นภาษาไทย
        status_map = {
            0: "ผอมแห้ง",
            1: "ผอม",
            2: "สมส่วน",
            3: "ท้วม",
            4: "อ้วน",
        }

        return {
            "success": True,
            "prediction": prediction,
            "status": status_map.get(prediction, "ไม่ทราบ"),
            "probabilities": probabilities,
            "confidence": float(max(probabilities)),
            "age_months": round(age_months, 1),
        }

    except Exception as e:
        import traceback
        print(f"Error in predict: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"เกิดข้อผิดพลาด: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("🚀 กำลังเริ่มเซิร์ฟเวอร์...")
    print("📝 เปิด http://127.0.0.1:8000/docs เพื่อดู API Documentation")