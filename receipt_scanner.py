from PIL import Image
import pandas as pd
import json
import uuid
import os
import io
import cv2
import numpy as np
import pytesseract # <-- Used for stable local OCR
import requests    # <-- NEW: Used for Gemini API calls
import time        # <-- NEW: Used for exponential backoff in API calls
from pydantic import BaseModel, Field, ValidationError
from typing import List, Union
from datetime import datetime
from enum import Enum

# External ML/AI dependencies
# Note: Keeping PaddleOCR import to prevent import errors in api.py, 
# but disabling initialization due to reported environmental issues.
from paddleocr import PaddleOCR 

# --- Tesseract Configuration (Adjust path if needed) ---
try:
    # Ensure this path is correct for your Windows Tesseract installation
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    # Fallback for Linux/Mac or if Tesseract is already in PATH
    pass

######################################################################
### CONFIGURATION AND EXAMPLES (Pydantic Schemas and Few-Shot Data)
######################################################################

# Pydantic Schemas and Few-Shot Examples (UNMODIFIED)
class ProductCategory(str, Enum):
    fruits = 'fruits'
    vegetables = 'vegetables'
    protein_foods = 'protein_foods'
    seafood = 'seafood'
    dairy = 'dairy'
    grains = 'grains'
    nuts_and_seeds = 'nuts_and_seeds'
    sweets = 'sweets'
    spices = 'spices'
    beverages = 'beverages'
    snacks = 'snacks'
    condiments = 'condiments'
    frozen_foods = 'frozen_foods'
    bakery = 'bakery'
    canned_goods = 'canned_goods'
    household = 'household'
    personal_care = 'personal_care'
    pet_supplies = 'pet_supplies'
    other = 'other'

class ItemInfo(BaseModel):
    name: str = Field(..., description="Name of the item")
    unit: float = Field(..., description="Quantity of the item")
    price: float = Field(..., description="Price per unit of the item")
    amount: float = Field(..., description="Total amount for the item")
    category: ProductCategory = Field(..., description="Category of the item")

class PaymentMethodEnum(str, Enum):
    tarjeta = 'tarjeta'
    efectivo = 'efectivo'

class ReceiptInfo(BaseModel):
    store: str = Field(..., description="Store name")
    address: str = Field(..., description="Address of the store")
    city: str = Field(..., description="City where the store is located")
    phone: str = Field(..., description="Phone number of the store")
    receipt_no: str = Field(..., description="Receipt number")
    date: str = Field(..., description="Date of the receipt in DD/MM/YYYY format")
    time: str = Field(..., description="Time of the transaction")
    items: List[ItemInfo] = Field(..., description="List of items purchased")
    total: float = Field(..., description="Total amount of the receipt")
    number_items: int = Field(..., description="Number of items in the receipt")
    payment_method: PaymentMethodEnum = Field(..., description="Payment method used")

# --- Few-Shot Examples (Hardcoded for Gemini Prompt) ---
example_cat_1 = {"store": "HiperDino", "address": "9238-SD Bernardo de la torre", "city": "Tafira Baja", "phone": "928493638", "receipt_no": "2024/923813-00060866", "date": "15/04/2024", "time": "16:01", "items": [{"name": "FRESA TARINA 500 GR", "unit": 1, "price": 1.59, "amount": 1.59, "category": "fruits"}, {"name": "HIPERDINO ACEITUNA R/ANCHOA LATA 350", "unit": 1, "price": 0.95, "amount": 0.95, "category": "canned_goods"}], "total": 9.96, "number_items": 5, "payment_method": "tarjeta"}
receipt_texts_1 = ['HiperDino', 'Las mcjores precios de Canarias', 'DINOSOL SUPERMERCADOS. S.L', 'C.I.F.B61742565', '9238-SD BERNARD0 DE LA T0RRE', 'Te1éfono:928493638', 'TOTAL COMPRA: 9,96']

example_cat_2 = {"store": "SPAR TAFIRA", "address": "C/. Bruno Naranjo DIAZ 9A-9B", "city": "Tafira Baja", "phone": "928 351 616", "receipt_no": "014\\002-18965", "date": "06/04/2024", "time": "15:23", "items": [{"name": "CLIPPER MANZ.1.5L.", "unit": 1, "price": 1.49, "amount": 1.49, "category": "beverages"}, {"name": "LECHE.GRNJ.FLR.UHT.", "unit": 1, "price": 1.15, "amount": 1.15, "category": "dairy"}], "total": 15.12, "number_items": 7, "payment_method": "tarjeta"}
receipt_texts_2 = ['SPAR TAFIRA', 'C/.BRUNO NARANJO DIAZ9A-B', 'TLF.:928351616-FAX:928351004', 'Nro.014002-18965', 'Fecha:06-04-202415:23', 'CLIPPER MANZ.1.5L. 1,49', 'LECHE.GRNJ.FLR.UHT. 1,15', 'Total F 15,12']

example_cat_4 = {"store": "MERCADONA", "address": "AVDA. PINTOR FELO MONZON (C.C. 7 PALMAS) S/N", "city": "35019 LAS PALMAS DE GRAN CANARIA", "phone": "928411755", "receipt_no": "2185-013-6970Z2", "date": "03/04/2024", "time": "21:22", "items": [ { "name": "DETERG HIPO COLONIA", "unit": 1, "price": 3.30, "amount": 3.30, "category": "household"}, { "name": "POLLO ENTERO LIMPIO", "unit": 1, "price": 6.52, "amount": 6.52, "category": "protein_foods"}, { "name": "BOLSA PLASTICO", "unit": 1, "price": 0.15, "amount": 0.15, "category": "household"} ], "total": 43.95, "number_items": 15, "payment_method": "tarjeta"}
receipt_texts_4 = ['MERCADONA.', 'A-46103834', 'AVDA. PINTOR FELO MONZON (C.C. 7 PALMAS) S/N', '35019 LAS PALMAS DE GRAN CANARIA', 'FACTURA SIMPLIFICADA:2185-013-6970Z2', '1 DETERG HIPO COLONIA 3,30', '1 POLLO ENTERO LIMPIO 6,52', '1 BOLSA PLASTICO 0,15', 'TOTAL @) 43,95']

######################################################################
### CORE API FUNCTIONS ###
######################################################################

# Global placeholder for OCR instance/status
_ocr_instance = True 

def get_ocr_instance():
    """Returns True if Tesseract path is configured, otherwise triggers an error."""
    # Checks if tesseract is configured in the python environment
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        return False
    except Exception:
        return False

def image_ocr(image_bytes: bytes) -> str:
    """
    Performs OCR on image bytes using Pytesseract and returns a single string of extracted text.
    """
    if not get_ocr_instance():
        return "OCR_ERROR: Tesseract executable not found. Check installation/path."

    try:
        # 1. Load image using PIL
        img_pil = Image.open(io.BytesIO(image_bytes))
        
        # 2. Preprocess (Convert to grayscale)
        img_gray = img_pil.convert('L') 
        
        # 3. Perform OCR (Using Spanish + English for robust receipt scanning)
        ocr_text = pytesseract.image_to_string(img_gray, lang='spa+eng').strip()

        if not ocr_text:
            return "OCR_WARNING: No text or result found in image."
            
        # Clean up text into list of lines for consistency with the prompt structure
        text_lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
        return " ".join(text_lines)
    
    except Exception as e:
        print(f"Error during Tesseract OCR processing: {e}")
        return f"OCR_ERROR: {str(e)}"


# Define a variable to hold the Gemini API endpoint
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

def structured_output(ocr_text: str, gemini_api_key: str) -> ReceiptInfo:
    """
    Calls the Gemini API to parse OCR text into the ReceiptInfo Pydantic model
    using JSON Schema and few-shot examples.
    """
    if not gemini_api_key:
        raise ValueError("Gemini API key is required.")
        
    # --- 1. Construct API Prompt and Examples ---
    system_instruction = (
        "You are a POS receipt data expert. Your task is to parse, detect, "
        "recognize, and convert the provided OCR text from a receipt image into "
        "a structured JSON object that strictly conforms to the given JSON schema. "
        "Assign a category to each item from the ProductCategory enum. "
        "Do not invent categories, fields, or values not explicitly present in the input text. "
        "Ensure all float values use a dot (.) as the decimal separator."
    )

    contents = [
        {"role": "system", "parts": [{"text": system_instruction}]}
    ]

    def create_example_turn(ocr_lines, json_output):
        return [
            {"role": "user", "parts": [{"text": "OCR Input:\n" + " ".join(ocr_lines)}]},
            {"role": "model", "parts": [{"text": json_output}]}
        ]

    contents.extend(create_example_turn(receipt_texts_1, json.dumps(example_cat_1)))
    contents.extend(create_example_turn(receipt_texts_2, json.dumps(example_cat_2)))
    contents.extend(create_example_turn(receipt_texts_4, json.dumps(example_cat_4)))
    contents.append({"role": "user", "parts": [{"text": "OCR Input:\n" + ocr_text}]})

    # 2. Get the JSON schema from the Pydantic model
    schema = ReceiptInfo.model_json_schema()

    # 3. Construct the API Payload (no config section)
    payload = {
        "contents": contents
    }

    # --- 4. Make the API Call with Backoff ---
    max_retries = 3
    delay = 1

    for attempt in range(max_retries):
        response = None
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={gemini_api_key}",
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()
            response_json = response.json()
            json_text = response_json['candidates'][0]['content']['parts'][0]['text']
            structured_data = ReceiptInfo.model_validate_json(json_text)
            return structured_data

        except requests.exceptions.RequestException as e:
            status_code = response.status_code if response is not None else 0
            if attempt < max_retries - 1 and (status_code >= 500 or isinstance(e, requests.exceptions.Timeout)):
                print(f"Request failed (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Final Gemini API request failed (Status: {status_code}): {e}")
                raise ValueError(f"Gemini API Request Failed (Status: {status_code})")
        except ValidationError as e:
            print(f"Validation Error: {e}")
            raise ValidationError(f"LLM output failed Pydantic validation: {e.errors()}")
        except Exception as e:
            print(f"Unexpected error in API call: {e}")
            raise ValueError(f"Gemini API Call Error: {e}")


# --- Utility Functions (Kept for completeness) ---

def initialize_session():
    return str(uuid.uuid4())

def parse_dates(date_str: str) -> datetime:
    """Parses date string with various formats."""
    date_formats = ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%Y.%m.%d", "%d.%m.%Y", "%m.%d.%Y"]
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    return pd.NaT 

def get_empty_dataframe() -> pd.DataFrame:
    """Creates an empty DataFrame matching the full expected schema."""
    return pd.DataFrame(columns=[
        'store', 'address', 'city', 'phone', 'receipt_no', 'date', 'time',
        'total', 'number_items', 'payment_method', 'week', 'month', 'name',
        'unit', 'price', 'amount', 'category'
    ])
