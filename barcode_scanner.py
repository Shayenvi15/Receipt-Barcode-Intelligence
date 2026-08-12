import cv2
import numpy as np
from PIL import Image
import io
from pyzbar.pyzbar import decode, ZBarSymbol
import json
import requests  # <-- Add this import

# Assuming the Google Search tool is available via the environment
# We will model the search operation using a mock interface.

class BarcodeError(Exception):
    """Custom exception for barcode processing errors."""
    pass

def scan_barcode(image_bytes: bytes):
    """
    Decodes barcodes and QR codes from an image byte stream using pyzbar.
    
    Args:
        image_bytes: Raw byte stream of the image file.

    Returns:
        A list of decoded barcode dictionaries or raises BarcodeError.
    """
    try:
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        np_img = np.array(img)
        barcodes = decode(np_img, symbols=[ZBarSymbol.EAN13, ZBarSymbol.UPCA, ZBarSymbol.CODE128, ZBarSymbol.QRCODE])
        if not barcodes:
            raise BarcodeError("No barcode found in image.")
        results = []
        for barcode in barcodes:
            results.append({
                "data": barcode.data.decode("utf-8"),
                "type": barcode.type
            })
        return results
    except Exception as e:
        raise BarcodeError(f"Image processing failed: {e}")


def lookup_product_details(barcode_data: str, gemini_api_key: str):
    """
    Looks up product information using the Gemini API with Google Search grounding.
    
    Args:
        barcode_data: The decoded EAN or UPC string.
        gemini_api_key: The API key for Gemini.

    Returns:
        A dictionary containing product details or an error message.
    """
    if not barcode_data:
        raise ValueError("Barcode data cannot be empty.")
        
    print(f"INFO: Looking up product details for barcode: {barcode_data}")

    user_query = f"Find the official name, manufacturer, and basic category for the product with barcode number: {barcode_data}. Respond ONLY with the requested product details in a concise summary."
    
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
    
    # Define the desired structured output schema for product details
    product_schema = {
        "type": "OBJECT",
        "properties": {
            "barcode": {"type": "STRING", "description": "The original barcode number."},
            "product_name": {"type": "STRING", "description": "The official or common name of the product."},
            "manufacturer": {"type": "STRING", "description": "The company that manufactures the product."},
            "category": {"type": "STRING", "description": "The general product category (e.g., 'Dairy', 'Snack Foods', 'Cleaning Supplies')."},
            "source_found": {"type": "BOOLEAN", "description": "True if product details were found, False otherwise."}
        }
    }

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "config": {
            "responseMimeType": "application/json",
            "responseSchema": product_schema
        },
        "tools": [{"google_search": {}}] # Enable Google Search grounding
    }
    
    # We use a simple attempt structure here; exponential backoff can be added if needed.
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={gemini_api_key}",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload),
            timeout=15 
        )
        response.raise_for_status()
        
        response_json = response.json()
        json_text = response_json['candidates'][0]['content']['parts'][0]['text']
        
        # Manually decode the JSON string returned by the LLM
        product_details = json.loads(json_text)
        
        # Ensure the original barcode is included
        product_details['barcode'] = barcode_data 
        
        return product_details

    except requests.exceptions.RequestException as e:
        error_msg = f"External API lookup failed: {e}"
        print(error_msg)
        return {"barcode": barcode_data, "source_found": False, "error": error_msg}
    except Exception as e:
        error_msg = f"Error parsing LLM response or unexpected error: {e}"
        print(error_msg)
        return {"barcode": barcode_data, "source_found": False, "error": error_msg}
