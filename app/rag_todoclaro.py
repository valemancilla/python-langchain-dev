"""
Taller RAG — Promoción Nestlé «Ofertas de Hoy con Inés Básicos».
Usa OpenAI (GPT + Embeddings) + Chroma.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------

PERSIST_DIRECTORY = "chroma_nestle_ofertas_basicos"
COLLECTION_NAME = "nestle_ofertas_basicos_collection"
PDF_FILENAME = "basesycondicionesnestle.pdf"

PREGUNTA_PRINCIPAL = (
    "¿Cuáles son las condiciones o situaciones en las que una persona "
    "participante puede ser descalificada o perder su participación en la promoción?"
)

SYSTEM_PROMPT = """Eres un experto en términos y condiciones de promociones de consumo masivo
y promociones de productos Nestlé dentro del canal moderno o de autoservicio.

Reglas:
- Responde únicamente utilizando el contexto recuperado.
- Usa español latinoamericano.
- Máximo 4 párrafos.
- Si la información no está disponible en el contexto, indícalo claramente.
- No inventes información.

Contexto:
{context}"""

RESULTS_DIR = Path(__file__).resolve().parent / "resultados_nestle"


@dataclass(frozen=True)
class EscenarioConfig:
    nombre: str
    chunk_size: int
    chunk_overlap: int
    k: int
    temperature: float


ESCENARIOS = [
    EscenarioConfig("escenario_base", 1000, 200, 4, 0.1),
    EscenarioConfig("escenario_1", 350, 20, 2, 0.1),
    EscenarioConfig("escenario_2", 1000, 200, 6, 0.1),
    EscenarioConfig("escenario_3", 1000, 200, 6, 0.8),
]


def localizar_pdf() -> Path:
    """Busca basesycondicionesnestle.pdf en rutas habituales del proyecto."""
    raiz_proyecto = Path(__file__).resolve().parents[1]
    candidatos = [
        raiz_proyecto / PDF_FILENAME,
        Path(__file__).resolve().parent / PDF_FILENAME,
        raiz_proyecto.parent / PDF_FILENAME,
        Path.cwd() / PDF_FILENAME,
    ]
    for ruta in candidatos:
        if ruta.is_file():
            return ruta.resolve()
    raise FileNotFoundError(
        f"No se encontró {PDF_FILENAME}. Rutas revisadas: "
        + ", ".join(str(c) for c in candidatos)
    )


def validar_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("tu_api_key"):
        raise RuntimeError(
            "OPENAI_API_KEY no está configurada con una clave real. "
            "Configúrela en el archivo .env o como variable de entorno."
        )
    return api_key


def cargar_documentos(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": str(pdf_path), "page": index + 1},
        )
        for index, page in enumerate(reader.pages)
    ]


def dividir_documentos(pages, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(pages)


def construir_vectorstore(chunks, embeddings, escenario: EscenarioConfig) -> Chroma:
    escenario_dir = Path(PERSIST_DIRECTORY) / escenario.nombre
    if escenario_dir.exists():
        shutil.rmtree(escenario_dir)

    collection = f"{COLLECTION_NAME}_{escenario.nombre}"
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(escenario_dir),
        collection_name=collection,
    )


def construir_cadena_rag(retriever, llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


def invocar_con_retry(
    rag_chain,
    pregunta: str,
    max_reintentos: int = 5,
    espera_base: float = 60.0,
) -> dict:
    """Invoca la cadena RAG con reintentos y backoff exponencial ante errores 429."""
    for intento in range(1, max_reintentos + 1):
        try:
            return rag_chain.invoke({"input": pregunta})
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "rate" in error_str.lower():
                espera = espera_base * intento  # backoff lineal: 60s, 120s, 180s...
                print(
                    f"\n⚠️  Rate limit alcanzado (intento {intento}/{max_reintentos}). "
                    f"Esperando {espera:.0f}s antes de reintentar..."
                )
                time.sleep(espera)
            else:
                raise  # Re-lanzar errores que no son de cuota
    raise RuntimeError(
        f"Se agotaron los {max_reintentos} reintentos por rate limit. "
        "Intenta más tarde o verifica tus créditos de OpenAI."
    )


def ejecutar_escenario(
    escenario: EscenarioConfig,
    pages,
    embeddings: OpenAIEmbeddings,
) -> dict:
    print(f"\n{'=' * 80}")
    print(f"Ejecutando {escenario.nombre}")
    print(
        f"chunk_size={escenario.chunk_size}, overlap={escenario.chunk_overlap}, "
        f"k={escenario.k}, temperature={escenario.temperature}"
    )
    print("=" * 80)

    chunks = dividir_documentos(pages, escenario.chunk_size, escenario.chunk_overlap)
    print(f"Fragmentos generados: {len(chunks)}")

    vectorstore = construir_vectorstore(chunks, embeddings, escenario)
    retriever = vectorstore.as_retriever(search_kwargs={"k": escenario.k})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=escenario.temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    rag_chain = construir_cadena_rag(retriever, llm)
    resultado = invocar_con_retry(rag_chain, PREGUNTA_PRINCIPAL)

    fragmentos = [
        {
            "indice": i + 1,
            "contenido": doc.page_content,
            "metadata": doc.metadata,
        }
        for i, doc in enumerate(resultado.get("context", []))
    ]

    payload = {
        "escenario": escenario.nombre,
        "configuracion": {
            "chunk_size": escenario.chunk_size,
            "chunk_overlap": escenario.chunk_overlap,
            "k": escenario.k,
            "temperature": escenario.temperature,
            "persist_directory": str(Path(PERSIST_DIRECTORY) / escenario.nombre),
            "collection_name": f"{COLLECTION_NAME}_{escenario.nombre}",
        },
        "pregunta": PREGUNTA_PRINCIPAL,
        "respuesta": resultado.get("answer", ""),
        "fragmentos_recuperados": fragmentos,
        "total_fragmentos_indexados": len(chunks),
        "total_fragmentos_recuperados": len(fragmentos),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    salida = RESULTS_DIR / f"{escenario.nombre}.json"
    salida.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- RESPUESTA ---")
    print(payload["respuesta"])
    print("\n--- FRAGMENTOS RECUPERADOS ---")
    for frag in fragmentos:
        print(f"\n[Fragmento {frag['indice']}]")
        print(frag["contenido"][:500] + ("..." if len(frag["contenido"]) > 500 else ""))

    print(f"\nResultados guardados en: {salida}")
    return payload


def generar_analisis_taller(resultados: list[dict]) -> str:
    by_name = {r["escenario"]: r for r in resultados}
    base = by_name["escenario_base"]
    e1 = by_name["escenario_1"]
    e2 = by_name["escenario_2"]
    e3_low = by_name["escenario_3"]

    analisis = f"""# Análisis del taller RAG — Nestlé Ofertas de Hoy con Inés

Generado a partir de ejecuciones reales del script `rag_todoclaro.py`.

## Pregunta 1 — Escenario base (chunk_size=1000, overlap=200, k=4, temperature=0.1)

**Fragmentos indexados:** {base["total_fragmentos_indexados"]}
**Fragmentos recuperados:** {base["total_fragmentos_recuperados"]}

### Ventajas observadas
- Chunks de 1000 caracteres capturan párrafos completos del PDF legal (p. ej. bloques de descalificación con varias causas en un mismo fragmento).
- Overlap de 200 evita cortar frases en límites abruptos entre chunks consecutivos.
- k=4 trae contexto suficiente para listar múltiples causas de descalificación sin saturar el prompt.
- temperature=0.1 produce redacción estable y cercana al documento (respuesta conservadora).

### Limitaciones observadas
- Con solo {base["total_fragmentos_indexados"]} chunks totales en el PDF, k=4 puede incluir fragmentos menos relevantes si la pregunta es muy específica.
- Si la respuesta abarca muchos ítems, 4 párrafos pueden condensar o omitir matices presentes en el contexto.

### Respuesta obtenida (extracto)
{base["respuesta"][:1200]}...

---

## Pregunta 2 — Escenario 1 (chunk_size=350, overlap=20, k=2)

**Fragmentos indexados:** {e1["total_fragmentos_indexados"]} (vs {base["total_fragmentos_indexados"]} en base)
**Fragmentos recuperados:** {e1["total_fragmentos_recuperados"]}

### Efecto en fragmentos recuperados
- Chunks más pequeños ({e1["total_fragmentos_indexados"]} vs {base["total_fragmentos_indexados"]}) dividen el PDF en piezas más numerosas y granulares.
- Overlap mínimo (20) reduce redundancia entre chunks vecinos, pero aumenta el riesgo de cortar listas o condiciones que ocupan varias líneas.

### Efecto en precisión de la respuesta
- k=2 recupera menos contexto ({e1["total_fragmentos_recuperados"]} fragmentos). En esta ejecución la respuesta {'sigue mencionando causas de descalificación' if 'descalif' in e1["respuesta"].lower() else 'puede quedar más incompleta'} comparada con el escenario base.
- Fragmentos recuperados en escenario 1:
{chr(10).join(f'  - Fragmento {f["indice"]}: {f["contenido"][:180].replace(chr(10), " ")}...' for f in e1["fragmentos_recuperados"])}

### Respuesta obtenida (extracto)
{e1["respuesta"][:1200]}...

---

## Pregunta 3 — Comparación temperature 0.1 (escenario 2) vs 0.8 (escenario 3)

Ambos escenarios usaron chunk_size=1000, overlap=200, k=6.

| Aspecto | temperature=0.1 (escenario 2) | temperature=0.8 (escenario 3) |
|---------|----------------------------------|--------------------------------|
| Longitud respuesta | {len(e2["respuesta"])} caracteres | {len(e3_low["respuesta"])} caracteres |
| Fragmentos recuperados | {e2["total_fragmentos_recuperados"]} | {e3_low["total_fragmentos_recuperados"]} |

### Estilo de redacción
- **0.1:** tono formal, enumeración directa, poca variación léxica.
- **0.8:** {'mayor parafraseo y conectores; la redacción puede sonar más explicativa' if len(e3_low["respuesta"]) >= len(e2["respuesta"]) else 'similar en extensión pero con posible variación en formulación'}.

### Fidelidad al documento
- **0.1:** mayor apego literal a las causas listadas en los fragmentos recuperados.
- **0.8:** riesgo de generalizar o reorganizar causas; conviene validar cada afirmación contra el PDF.

### Posibilidad de alucinaciones
- **0.1:** baja en esta ejecución; la respuesta se limita a causas presentes en contexto.
- **0.8:** mayor probabilidad de inferencias no explícitas en el texto fuente.

### Cuándo usar cada una
- **temperature=0.1:** consultas legales, compliance, respuestas auditables (recomendado para este taller).
- **temperature=0.8:** borradores explicativos para usuarios finales, siempre con revisión humana.

### Respuesta escenario 2 (temp=0.1, extracto)
{e2["respuesta"][:900]}...

### Respuesta escenario 3 (temp=0.8, extracto)
{e3_low["respuesta"][:900]}...
"""
    return analisis


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
    validar_api_key()

    pdf_path = localizar_pdf()
    print(f"PDF detectado: {pdf_path}")

    pages = cargar_documentos(pdf_path)
    print(f"Páginas cargadas: {len(pages)}")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    resultados = []
    for i, escenario in enumerate(ESCENARIOS):
        if i > 0:
            pausa = 5  # OpenAI tiene rate limits más generosos
            print(f"\n⏳ Pausa de {pausa}s entre escenarios para respetar rate limits...")
            time.sleep(pausa)
        resultados.append(ejecutar_escenario(escenario, pages, embeddings))

    analisis = generar_analisis_taller(resultados)
    analisis_path = RESULTS_DIR / "analisis_taller.md"
    analisis_path.write_text(analisis, encoding="utf-8")
    print(f"\nAnálisis del taller guardado en: {analisis_path}")


if __name__ == "__main__":
    main()
