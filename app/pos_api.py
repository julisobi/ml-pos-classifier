"""API file.

This module provides FastAPI endpoints for product category prediction.
"""

import pandas as pd
import logging
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from monitoring.json_monitor import update_monitoring_json, update_prediction_time
from pos_classifier.data.postprocessing import decode_fasttext_label
from pos_classifier.config.config import get_prediction_output_path, FASTTEXT_MODEL_PATH
from pos_classifier.config.logging_config import setup_logging
from pos_classifier.model.fasttext_wrapper import FastTextModelWrapper

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI()

params = {"model_location": str(FASTTEXT_MODEL_PATH)}
fasttext_model = FastTextModelWrapper(params)
fasttext_model.load_model()


class ProductInput(BaseModel):
    """Input model for product data.

    Attributes:
        product_description (str): A description of the product.

    """

    product_description: str


@app.post("/predict")
def get_prediction(data: ProductInput):
    """Get a category prediction for a given product description."""
    logger.info(
        f"Received prediction request for product description: {data.product_description}"
    )

    label, probability = fasttext_model.predict(data.product_description)
    category = decode_fasttext_label(label)

    logger.info(f"Prediction result: {category} with probability: {probability[0]}")

    return {"prediction": category, "probability": probability[0]}


@app.post("/predict_batch")
async def batch_prediction(file: UploadFile = File(...)):
    """Handle batch prediction requests from a CSV file."""
    logger.info(f"Received batch prediction request with file: {file.filename}")

    if not file.filename.endswith(".csv"):
        logger.error("Invalid file format received. Only CSV files are supported.")
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        df = pd.read_csv(file.file)
        logger.info(f"Loaded {len(df)} rows from CSV file.")

        if "product_description" not in df.columns:
            logger.error("Missing 'product_description' column in CSV.")
            raise HTTPException(
                status_code=400, detail="Missing 'product_description' column in CSV."
            )

        has_labels = "HUMAN_VERIFIED_Category" in df.columns
        correct_predictions = 0
        total_predictions = 0

        results = []
        for _, row in df.iterrows():
            start_time = time.perf_counter()
            label, probability = fasttext_model.predict(row["product_description"])
            category = decode_fasttext_label(label)
            elapsed = time.perf_counter() - start_time
            update_prediction_time(elapsed)
            update_monitoring_json(category)
            if has_labels:
                true_label = row["HUMAN_VERIFIED_Category"]
                if pd.isna(true_label):
                    continue
                total_predictions += 1
                update_monitoring_json("total_predictions")
                if true_label == category:
                    correct_predictions += 1
                    update_monitoring_json("correct_predictions")

            results.append(
                {
                    "product_description": row["product_description"],
                    "predicted_category": category,
                    "probability": probability[0],
                }
            )
        result_df = pd.DataFrame(results)

        output_path = get_prediction_output_path()
        result_df.to_csv(output_path, index=False)

        logger.info(f"Batch prediction completed. Results saved to {output_path}.")

        return {
            "message": "Batch prediction complete.",
            "output_file": str(output_path),
            "rows_processed": len(result_df),
        }

    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
