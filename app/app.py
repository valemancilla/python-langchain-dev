import chromadb

# Crea una base de datos persistente.
client = chromadb.PersistentClient(path="./chroma_data")

# Obtiene la colección o la crea si no existe.
collection = client.get_or_create_collection(
    name="conceptos_ia"
)

# Inserta o actualiza documentos.
collection.upsert(
    ids=["doc-1", "doc-2", "doc-3"],
    documents=[
        "ChromaDB es una base de datos vectorial.",
        "Los embeddings representan información mediante vectores.",
        "Los modelos LLM generan y procesan lenguaje natural."
    ],
    metadatas=[
        {"tema": "chromadb"},
        {"tema": "embeddings"},
        {"tema": "llm"}
    ]
)

# Realiza una búsqueda semántica.
resultado = collection.query(
    query_texts=["¿Cómo se representa numéricamente la información?"],
    n_results=2,
    include=["documents", "metadatas", "distances"]
)

print("Documentos encontrados:")
print(resultado["documents"])

print("\nMetadatos:")
print(resultado["metadatas"])

print("\nDistancias:")
print(resultado["distances"])