import os
from pathlib import Path
import chromadb
from openai import OpenAI
from dotenv import load_dotenv


# Paso 4 – Función para generar embeddings usando OpenAI
def embed_openai(textos):
    response = client_openai.embeddings.create(
        input=textos,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]


def imprimir_resultados(results, titulo="Resultados"):
    print("=" * 90)
    print(titulo)
    print("=" * 90)
    # results['ids'], ['documents'], ['distances'] vienen anidados por consulta
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for i, (rid, rdoc, rdist) in enumerate(zip(ids, docs, dists), start=1):
        print(f"Top {i} | ID: {rid:<28} | Distancia: {rdist:.4f}")
        snippet = (rdoc[:180] + "...") if len(rdoc) > 180 else rdoc
        print(f"      Snippet: {snippet}\n")

    print("Nota: menor distancia = mayor similitud semántica.")


env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key or api_key.startswith("tu_api_key"):
    raise RuntimeError(
        "OPENAI_API_KEY no esta configurada con una clave real. "
        f"Revise el archivo {env_path} o la variable de entorno OPENAI_API_KEY."
    )

# Paso 1 – Crear el directorio para la base vectorial generada con OpenAI
PERSIST_DIR_OPENAI = "./db_chroma_banco_andino_openai"
os.makedirs(PERSIST_DIR_OPENAI, exist_ok=True)

print("Carpeta de persistencia (OpenAI):", os.path.abspath(PERSIST_DIR_OPENAI))

# Paso 2 – Crear cliente persistente de Chroma para esta base
client_chroma_openai = chromadb.PersistentClient(path=PERSIST_DIR_OPENAI)

# Colección específica para estos embeddings
collection_openai = client_chroma_openai.get_or_create_collection(
    name="banco_andino_openai"
)

# Paso 3 – Configurar el cliente de OpenAI
OPENAI_API_KEY = api_key
client_openai = OpenAI(api_key=OPENAI_API_KEY)

documentos_banco_andino = [
    {
        "id": "consumo_clasico",
        "texto": """
Crédito de consumo clásico dirigido a personas asalariadas con al menos 12 meses de estabilidad laboral.
Monto: USD 1,000 a 10,000. Plazo: 6 a 36 meses.
Requisitos: comprobante de ingresos, sin mora mayor a 30 días en los últimos 12 meses.
La cuota mensual no debe superar el 35% del ingreso neto del cliente.
"""
    },
    {
        "id": "nomina_convenio",
        "texto": """
Crédito con descuento por nómina para empleados de empresas con convenio vigente con el banco.
No requiere codeudor. Antigüedad mínima: 6 meses.
Monto máximo: hasta 8 veces el salario neto mensual.
El pago se realiza vía deducción automática en planilla.
"""
    },
    {
        "id": "hipotecario_primera_vivienda",
        "texto": """
Crédito hipotecario para primera vivienda.
Financia hasta el 80% del valor de tasación del inmueble. Plazo hasta 20 años.
Requisitos: enganche mínimo del 20%, sin registros negativos en los últimos 24 meses.
Las obligaciones mensuales totales (incluida la hipoteca) no deben superar el 40% del ingreso familiar neto.
"""
    },
    {
        "id": "pyme_capital_trabajo",
        "texto": """
Crédito PYME Capital de Trabajo para empresas con al menos 2 años de operación formal.
Montos desde USD 5,000 hasta USD 200,000. Plazo hasta 24 meses.
Requiere estados financieros, flujo de caja proyectado y, según el riesgo, garantías
reales o fideicomisos.
"""
    },
    {
        "id": "politica_riesgo_general",
        "texto": """
Política general de riesgo de crédito.
Se evalúan estabilidad laboral o del negocio, nivel de endeudamiento, score interno y externo,
y comportamiento histórico con el banco.
Solicitudes con endeudamiento total superior al 45% del ingreso neto se consideran solo
de forma excepcional.
No se aprueban créditos con moras activas mayores a 90 días al momento de la evaluación.
"""
    },
]

docs_banco_andino = {
    "ids": [d["id"] for d in documentos_banco_andino],
    "documents": [d["texto"] for d in documentos_banco_andino],
}

# Paso 8 – Embeddings para todos los documentos
embeddings_docs = embed_openai(docs_banco_andino["documents"])

print("Documentos:", len(docs_banco_andino["documents"]))
print("Embeddings generados:", len(embeddings_docs))
print("Dimensión de embedding:", len(embeddings_docs[0]))

# Paso 9 – Insertar documentos + embeddings en Chroma
collection_openai.upsert(
    ids=docs_banco_andino["ids"],
    documents=docs_banco_andino["documents"],
    embeddings=embeddings_docs,
)

print("Documentos almacenados en Chroma con embeddings de OpenAI.")

# Paso 10 – Pregunta de prueba (usted puede cambiarla)
pregunta = "¿Cuál es el monto máximo que puedo solicitar con descuento por nómina?"

embedding_pregunta = embed_openai([pregunta])[0]

resultado_openai = collection_openai.query(
    query_embeddings=[embedding_pregunta],
    n_results=3
)

imprimir_resultados(resultado_openai)