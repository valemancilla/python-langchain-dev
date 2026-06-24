import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path


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


# Carpeta local donde se persistirá la base de Chroma
PERSIST_DIR = "./db_chroma_banco_andino"

os.makedirs(PERSIST_DIR, exist_ok=True)

print("Carpeta de persistencia:", os.path.abspath(PERSIST_DIR))

# Cliente persistente: guarda en disco (SQLite)
client = chromadb.PersistentClient(path=PERSIST_DIR)

# Embeddings por defecto de Chroma
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Nombre de la colección de Banco Andino (manténlo estable entre sesiones)
COLLECTION_NAME = "banco_andino_v1"

# Crear u obtener la colección persistente
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=default_ef
)

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

# ------------------------------------------------------
# Convertimos la lista de diccionarios al formato que requiere Chroma
docs_banco_andino = {
    "ids": [d["id"] for d in documentos_banco_andino],
    "documents": [d["texto"] for d in documentos_banco_andino],
}

collection.upsert(
    ids=docs_banco_andino["ids"],
    documents=docs_banco_andino["documents"],
    metadatas=docs_banco_andino.get("metadatas")  # puede ser None
)

print("Documentos almacenados (persistentes).")
consulta_rel = "¿Cuál es el monto máximo que puedo solicitar con descuento por nómina?"
res_rel = collection.query(query_texts=[consulta_rel], n_results=3)
imprimir_resultados(res_rel, titulo="(Persistente) Consulta relacionada – Monto máximo por nómina")

db_files = list(Path(PERSIST_DIR).glob("**/*"))
print("Archivos en la carpeta de persistencia:")
for p in db_files[:15]:
    print(" -", p.relative_to(PERSIST_DIR))

print(f"\nTotal de archivos/carpetas detectados: {len(db_files)}")

# Simular que "cerramos y reabrimos" el proyecto: re-instanciamos cliente y colección
client2 = chromadb.PersistentClient(path=PERSIST_DIR)
collection2 = client2.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=default_ef  # misma EF para consultas consistentes
)
# Consulta idéntica: deberían salir resultados coherentes (misma base persistida)
res_rel_2 = collection2.query(query_texts=[consulta_rel], n_results=3)
imprimir_resultados(res_rel_2, titulo="(Reapertura) Consulta relacionada – Monto máximo por nómina")