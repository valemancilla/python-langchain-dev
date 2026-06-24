import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("✅ API key registrada en el Proyecto. (No la compartas con nadie).")
else:
    print("❌ Aún no has pegado tu API key. Actualiza la variable api_key.")

client = chromadb.Client()
print("✅ Chroma inicializado correctamente en modo local.")

test_collection = client.get_or_create_collection(name="prueba_entorno")

print("✅ Colección creada o recuperada correctamente:", test_collection.name)