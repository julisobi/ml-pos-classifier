"""API file.

This module provides FastAPI endpoints for product category prediction.
"""

import pandas as pd
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.pos_classifier.model.predict import predict_label, decode_fasttext_label
from src.pos_classifier.config.config import get_prediction_output_path
from src.pos_classifier.config.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI()


class ProductInput(BaseModel):
    """Input model for product data.

    Attributes:
        product_description (str): A description of the product.

    """

    product_description: str


@app.post("/predict")
def get_prediction(data: ProductInput):
    """Predict the product category from a single product description.

    Args:
        data (ProductInput): An object containing the product description.

    Returns:
        dict: A dictionary with the predicted category and its probability.

    """
    logger.info(
        f"Received prediction request for product description: {data.product_description}"
    )

    label, probability = predict_label(data.product_description)
    category = decode_fasttext_label(label)

    logger.info(f"Prediction result: {category} with probability: {probability[0]}")

    return {"prediction": category, "probability": probability[0]}


@app.post("/predict-batch")
async def batch_prediction(file: UploadFile = File(...)):
    """Perform batch predictions from a CSV file containing product descriptions.

    Args:
        file (UploadFile): A CSV file with a 'product_description' column.

    Returns:
        dict: A summary of the batch prediction including output file path and rows processed.

    Raises:
        HTTPException: If the file format is invalid, the required column is missing,
                        or any other error occurs during processing.

    """
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

        results = []
        for _, row in df.iterrows():
            label, probability = predict_label(row["product_description"])
            category = decode_fasttext_label(label)
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
