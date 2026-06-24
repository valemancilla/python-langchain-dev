"""Diagnóstico de OPENAI_API_KEY"""
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[1] / ".env"
print(f"Cargando .env desde: {env_path}")
print(f"Archivo existe: {env_path.is_file()}")

load_dotenv(env_path, override=True)

key = os.getenv("OPENAI_API_KEY", "")

print(f"\n--- Diagnóstico OPENAI_API_KEY ---")
print(f"Key existe: {bool(key)}")
print(f"Longitud: {len(key)}")
print(f"Primeros 10 chars: {key[:10]}...")
print(f"Últimos 4 chars: ...{key[-4:]}")
print(f"Tiene espacios al inicio/final: {key != key.strip()}")
print(f"Contiene \\r o \\n: {chr(13) in key or chr(10) in key}")
print(f"Empieza con 'sk-': {key.startswith('sk-')}")

# Prueba real contra OpenAI
print("\n--- Prueba de conexión con OpenAI ---")
try:
    from openai import OpenAI
    client = OpenAI(api_key=key.strip())
    response = client.models.list()
    models = [m.id for m in response.data[:3]]
    print(f"✅ Conexión EXITOSA. Modelos disponibles (muestra): {models}")
except Exception as e:
    print(f"❌ Error de conexión: {type(e).__name__}: {e}")
