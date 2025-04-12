from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.pos_classifier.model.predict import predict_label, decode_fasttext_label
from src.pos_classifier.config.config import get_prediction_output_path

app = FastAPI()


class ProductInput(BaseModel):
    product_description: str


@app.post("/predict")
def get_prediction(data: ProductInput):
    label, probability = predict_label(data.product_description)
    category = decode_fasttext_label(label)
    return {
        "prediction": category,
        "confidence": probability[0]
    }


@app.post("/predict-batch")
async def batch_prediction(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        df = pd.read_csv(file.file)

        if 'product_description' not in df.columns:
            raise HTTPException(status_code=400, detail="Missing 'product_description' column in CSV.")

        results = []
        for _, row in df.iterrows():
            label, probability = predict_label(row["product_description"])
            category = decode_fasttext_label(label)
            results.append({
                "product_description": row["product_description"],
                "predicted_category": category,
                "confidence": probability[0]
            })

        result_df = pd.DataFrame(results)

        output_path = get_prediction_output_path()
        result_df.to_csv(output_path, index=False)

        return {
            "message": "Batch prediction complete.",
            "output_file": str(output_path),
            "rows_processed": len(result_df)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
