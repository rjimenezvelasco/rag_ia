import PyPDF2
import requests
import json

OLLAMA_API_URL = "http://localhost:11434/api/chat"

def extraer_texto_pdf(ruta_pdf):
    texto = ""
    with open(ruta_pdf, 'rb') as archivo:
        lector = PyPDF2.PdfReader(archivo)
        for pagina in lector.pages:
            texto += pagina.extract_text() + "\n"
    return texto

def resumir_con_mistral(texto):
    prompt = (
        "Por favor, resume el siguiente documento PDF en español con los puntos más importantes:\n\n"
        f"{texto[:3000]}"  # Limitar por tamaño si es muy grande
    )

    data = {
        "model": "mistral",
        "messages": [{"role": "user", "content": prompt}]
    }

    headers = {"Content-Type": "application/json"}

    try:
        with requests.post(OLLAMA_API_URL, json=data, headers=headers, stream=True) as response:
            response.raise_for_status()
            resultado = ""
            for linea in response.iter_lines():
                if linea:
                    try:
                        json_line = json.loads(linea)
                        resultado += json_line.get("message", {}).get("content", "")
                    except json.JSONDecodeError as e:
                        print("Error al decodificar JSON:", e)
            return resultado
    except requests.exceptions.RequestException as e:
        print("Error al conectar con Mistral:", e)
        return None

if __name__ == "__main__":
    ruta_pdf = "archivo.pdf"  # Cambia por la ruta de tu archivo
    texto_pdf = extraer_texto_pdf(ruta_pdf)
    
    if texto_pdf:
        resumen = resumir_con_mistral(texto_pdf)
        print("\n Resumen del documento:")
        print(resumen)
    else:
        print("No se pudo extraer texto del PDF.")
