# 🧾 Receipt & Barcode Intelligence API

An AI-powered **Receipt and Barcode Intelligence API** built with Python and FastAPI.

The project combines **OCR, computer vision, barcode decoding, Gemini-based structured extraction, Google Search grounding, and Pydantic validation** to transform receipt and barcode images into structured information.

---

## ✨ Overview

The application provides two primary processing pipelines:

### 🧾 Receipt Intelligence

Upload a receipt image and the system:

1. Validates the image format
2. Preprocesses the image
3. Performs OCR using **Tesseract**
4. Extracts text from the receipt
5. Sends the OCR output to **Google Gemini**
6. Converts the result into structured receipt information
7. Validates the response using **Pydantic**
8. Returns structured JSON data

### 🏷️ Barcode Intelligence

Upload an image containing a barcode or QR code and the system:

1. Reads the uploaded image
2. Detects and decodes supported barcodes
3. Extracts the barcode value and type
4. Identifies commercial barcode formats
5. Uses **Gemini with Google Search grounding** for product lookup
6. Returns structured product information

---

## 🧠 Core Capabilities

- 📷 Image-based receipt processing
- 🔎 OCR-based text extraction
- 🤖 Gemini-powered structured information extraction
- 🧾 Structured receipt representation
- 🏷️ Barcode and QR code decoding
- 🔍 AI-assisted product lookup
- 🌐 Google Search grounding for product information
- ✅ Pydantic-based response validation
- 🚀 FastAPI REST API
- 📦 JSON-based structured responses
- ⚠️ Input validation and exception handling

---

# 🏗️ System Architecture

The application consists of two primary pipelines.

```text
                         IMAGE INPUT
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
          RECEIPT IMAGE             BARCODE / QR IMAGE
                │                         │
                ▼                         ▼
          Tesseract OCR              pyzbar Decoder
                │                         │
                ▼                         ▼
            OCR Text                 Barcode Data
                │                         │
                ▼                         │
             Gemini                      │
                │                         │
                ▼                         ▼
       Structured Receipt          Gemini + Google
            Information             Search Grounding
                │                         │
                └────────────┬────────────┘
                             │
                             ▼
                          FastAPI
                             │
                             ▼
                       JSON Response
