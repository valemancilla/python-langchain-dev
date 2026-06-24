"""
Taller: Asistente Interno BCO ANDINO
Parte 1: Colección ChromaDB con políticas de crédito.
Parte 2: Búsqueda semántica y presentación de resultados.
"""

import chromadb

# ---------------------------------------------------------------------------
# Parte 1 — Cliente y colección
# ---------------------------------------------------------------------------

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="banco_andino_politicas_credito",
    metadata={
        "tema": "politicas_credito",
        "idioma": "es",
    },
)

# Documentos internos de políticas de crédito del BCO ANDINO
documentos_banco_andino = [
    {
        "id": "consumo_clasico",
        "texto": (
            "Crédito de consumo clásico BCO ANDINO. Producto destinado a financiar "
            "bienes y servicios personales como electrodomésticos, educación o "
            "viajes. Monto mínimo $5.000.000 y máximo $80.000.000. Plazo entre "
            "12 y 60 meses. La tasa es variable según el perfil de riesgo del "
            "cliente. Requiere demostrar capacidad de pago y un historial crediticio "
            "favorable en centrales de riesgo."
        ),
    },
    {
        "id": "nomina_convenio",
        "texto": (
            "Crédito con descuento por nómina BCO ANDINO. Disponible para empleados "
            "de empresas que mantengan un convenio de nómina vigente con el banco. "
            "El monto máximo que puede solicitar el cliente con descuento por nómina "
            "es hasta 10 veces su salario mensual neto, sin superar $100.000.000. "
            "El descuento se aplica automáticamente de la nómina en cada periodo de "
            "pago. Tasa preferencial EA del 18% y plazo máximo de 48 meses."
        ),
    },
    {
        "id": "hipotecario_primera_vivienda",
        "texto": (
            "Crédito hipotecario de primera vivienda BCO ANDINO. Financia la compra "
            "de vivienda nueva o usada para uso propio. Financiación hasta el 70% "
            "del valor comercial del inmueble. Plazo máximo 20 años. Requiere "
            "aporte mínimo del 30% como cuota inicial. Se exige seguro de vida "
            "deudor y avalúo del inmueble aprobado por el banco."
        ),
    },
    {
        "id": "pyme_capital_trabajo",
        "texto": (
            "Crédito PYME capital de trabajo BCO ANDINO. Dirigido a pequeñas y "
            "medianas empresas para cubrir necesidades de liquidez operativa, "
            "inventarios y pago a proveedores. Monto entre $20.000.000 y "
            "$500.000.000 según flujo de caja y antigüedad del negocio. Plazo "
            "de 6 a 36 meses. Puede requerir codeudor o garantía según el monto "
            "solicitado."
        ),
    },
    {
        "id": "politica_riesgo_general",
        "texto": (
            "Política general de riesgo crediticio BCO ANDINO. Regla de "
            "endeudamiento: el porcentaje máximo del ingreso que puede destinarse "
            "a la cuota mensual de obligaciones crediticias, incluyendo la nueva "
            "operación, es del 40% del ingreso mensual neto del solicitante. "
            "Para créditos hipotecarios de vivienda el límite sobre ingresos es "
            "del 30%. Se evalúa el endeudamiento total reportado en centrales de "
            "riesgo antes de aprobar cualquier desembolso."
        ),
    },
]

# Cargar documentos solo si la colección está vacía (evita IDs duplicados)
if collection.count() == 0:
    ids = [doc["id"] for doc in documentos_banco_andino]
    texts = [doc["texto"] for doc in documentos_banco_andino]

    collection.add(
        ids=ids,
        documents=texts,
    )
    print("Documentos cargados en la colección.")
else:
    print(f"Colección existente reutilizada ({collection.count()} documentos).")

# Verificación del contenido almacenado
print("\n--- Verificación de la colección ---")
print(f"Total de documentos: {collection.count()}")
print("Vista previa (peek):")
print(collection.peek())

# ---------------------------------------------------------------------------
# Parte 2 — Función para mostrar resultados de búsqueda semántica
# ---------------------------------------------------------------------------


def imprimir_resultados(resultados, titulo="Resultados"):
    """
    Muestra los resultados de collection.query() de forma organizada.
    Una menor distancia indica mayor similitud semántica con la consulta.
    """
    print(f"\n{'=' * 60}")
    print(titulo)
    print("=" * 60)
    print(
        "Nota: una menor distancia representa una mayor similitud semántica.\n"
    )

    ids = resultados["ids"][0]
    documentos = resultados["documents"][0]
    distancias = resultados["distances"][0]

    etiquetas_ranking = ["Top 1", "Top 2", "Top 3"]

    for i, (doc_id, texto, distancia) in enumerate(
        zip(ids, documentos, distancias)
    ):
        ranking = etiquetas_ranking[i] if i < len(etiquetas_ranking) else f"Top {i + 1}"
        snippet = texto[:120] + ("..." if len(texto) > 120 else "")

        print(f"{ranking}")
        print(f"  ID:       {doc_id}")
        print(f"  Distancia: {distancia:.4f}")
        print(f"  Snippet:  {snippet}")
        print("-" * 40)


# ---------------------------------------------------------------------------
# Consultas semánticas
# ---------------------------------------------------------------------------

print("\n\n" + "#" * 60)
print("# CONSULTAS SEMÁNTICAS — ASISTENTE INTERNO BCO ANDINO")
print("#" * 60)

# Consulta relacionada #1: política de endeudamiento sobre ingresos
query_1 = (
    "Cual es el porcentaje máximo del ingreso que puede destinarse "
    "a la cuota mensual?"
)
resultados_1 = collection.query(
    query_texts=[query_1],
    n_results=3,
)
imprimir_resultados(
    resultados_1,
    titulo="Consulta relacionada #1 — Cuota máxima sobre ingresos",
)

# Consulta relacionada #2: monto máximo con descuento por nómina
query_2 = "¿Cuál es el monto máximo que puedo solicitar con descuento por nómina?"
resultados_2 = collection.query(
    query_texts=[query_2],
    n_results=3,
)
imprimir_resultados(
    resultados_2,
    titulo="Consulta relacionada #2 — Monto máximo crédito por nómina",
)

# Consulta no relacionada: fuera del dominio bancario
query_3 = "¿Cuántas unidades de zapatillas vendimos en la tienda el mes pasado?"
resultados_3 = collection.query(
    query_texts=[query_3],
    n_results=3,
)
imprimir_resultados(
    resultados_3,
    titulo="Consulta NO relacionada — Ventas de zapatillas (fuera de dominio)",
)
print(
    "\nObservación: esta consulta no pertenece al dominio de políticas de crédito "
    "del BCO ANDINO. Los resultados muestran los documentos más cercanos en el "
    "espacio semántico, pero no responden la pregunta sobre ventas minoristas."
)
