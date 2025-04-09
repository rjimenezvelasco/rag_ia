from fastapi import FastAPI, File, UploadFile
import fitz  # PyMuPDF
import requests
import json

app = FastAPI()

OLLAMA_API_URL = "http://localhost:11434/api/chat"

def extract_text_from_pdf(file_data: bytes) -> str:
    text = ""
    with fitz.open(stream=file_data, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def query_mistral_for_summary(context: str) -> str:
    prompt = "Por favor, proporciona un resumen claro y conciso del siguiente documento:"
    full_prompt = f"Contexto: {context}\n\nPregunta: {prompt}"

    headers = {"Content-Type": "application/json"}
    data = {
        "model": "mistral",
        "messages": [{"role": "user", "content": full_prompt}]
    }

    try:
        with requests.post(OLLAMA_API_URL, json=data, headers=headers, stream=True) as response:
            response.raise_for_status()
            result = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_line = json.loads(line)
                        result += json_line.get("message", {}).get("content", "")
                    except json.JSONDecodeError:
                        continue
            return result
    except requests.exceptions.RequestException as e:
        return f"Error al consultar Mistral: {e}"

@app.post("/resumir/")
async def resumir_pdf(file: UploadFile = File(...)):
    content = await file.read()
    extracted_text = extract_text_from_pdf(content)

    resumen = query_mistral_for_summary(extracted_text[:3000])  # Limitar tamaño si es necesario
    return {"resumen": resumen}
