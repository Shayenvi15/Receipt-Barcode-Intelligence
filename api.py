from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from typing import Dict, Any, Optional

# --- Import Core Processing Logic and Schemas ---
# NOTE: These modules must be available in the same directory/path
from receipt_scanner import image_ocr, structured_output, ReceiptInfo
from barcode_scanner import scan_barcode, lookup_product_details, BarcodeError

# Initialize FastAPI app
app = FastAPI(
    title="Barcode and Receipt Scanner API (OpenAI/Gemini Mix)",
    description="API for extracting structured data from receipts (OpenAI) and performing product lookup (Gemini) via barcodes.",
    version="1.0.0"
)

# --- Endpoints ---

@app.get("/")
async def root():
    """Simple health check endpoint."""
    return {"message": "Barcode and Receipt Scanner API is running. Check /docs for endpoints."}

@app.post("/receipt/scan", response_model=ReceiptInfo)
async def scan_receipt(
    file: UploadFile = File(..., description="Receipt image file (JPEG, PNG)."),
    openai_api_key: str = Form(..., description="Your OpenAI API Key for LLM processing.")
):
    """
    Uploads a receipt image, performs OCR, and extracts structured JSON data using OpenAI.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG/PNG images are supported.")

    try:
        # Read image file into bytes
        image_bytes = await file.read()

        # 1. Perform OCR
        ocr_text = image_ocr(image_bytes)
        
        if ocr_text.startswith("OCR_ERROR"):
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {ocr_text}")

        # 2. Get Structured Data from LLM (Uses openai_api_key)
        structured_data: ReceiptInfo = structured_output(ocr_text, openai_api_key)

        return structured_data.model_dump()

    except ValidationError as e:
        # Handle cases where the LLM output is not valid JSON or doesn't match the schema
        raise HTTPException(status_code=422, detail=f"LLM output validation error: {e.errors()}")
    except ValueError as e:
        # Handle custom errors raised by our logic (e.g., missing API key)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch any unexpected errors
        print(f"Unexpected error in receipt endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.post("/barcode/scan", response_model=Dict[str, Any])
async def scan_and_lookup(
    file: UploadFile = File(..., description="Image containing the barcode or QR code."),
    openai_api_key: str = Form(..., description="Your OpenAI API Key for product lookup.")
):
    """
    Decodes the barcode in the uploaded image and attempts to look up product details.
    
    NOTE: This lookup still uses the Gemini API via Google Search Grounding for web accuracy. 
    The field name has been changed for user convenience.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG/PNG images are supported.")

    try:
        image_bytes = await file.read()
        
        # 1. Decode Barcode
        decoded_barcodes = scan_barcode(image_bytes)
        
        # Process the *first* decoded barcode found
        first_barcode_data = decoded_barcodes[0]["data"]
        first_barcode_type = decoded_barcodes[0]["type"]

        # 2. Lookup Product Details (EAN/UPC barcodes typically contain commercial product info)
        # We must assume the 'structured_output' function uses the provided API key for its web search function
        if first_barcode_type in ["EAN13", "UPCA", "CODE128"]:
            # NOTE: We pass the OpenAI key assuming the downstream function (lookup_product_details)
            # is capable of handling either key or has been adjusted internally.
            product_details = lookup_product_details(first_barcode_data, openai_api_key) 
            product_details["barcode_type"] = first_barcode_type
            
            return JSONResponse(content={"decoded_barcode": decoded_barcodes[0], "product_details": product_details})
        
        # For non-commercial codes (QR Code, Code 39, etc.), return only the raw data.
        return JSONResponse(content={"decoded_barcodes": decoded_barcodes[0], 
                                    "product_details": {"note": "Lookup skipped for non-commercial code type.", 
                                                        "type": first_barcode_type, 
                                                        "data": first_barcode_data}})

    except BarcodeError as e:
        raise HTTPException(status_code=400, detail=f"Barcode decoding failed: {str(e)}")
    except Exception as e:
        # Catch any unexpected errors
        print(f"Unexpected error in barcode endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
