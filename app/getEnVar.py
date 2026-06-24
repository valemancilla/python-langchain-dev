import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

my_server = os.getenv("POSTGRES_HOST")

if not my_server:
    raise ValueError("No se encontro servidor en el .env")
else:
    print(my_server)