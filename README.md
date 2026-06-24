# Taller RAG — Promoción Nestlé «Ofertas de Hoy con Inés Básicos»

Sistema de **Retrieval-Augmented Generation (RAG)** que responde preguntas sobre las bases
y mecánicas de la promoción *«Ofertas de Hoy con Inés Básicos»* de Nestlé México, a partir
del documento legal `basesycondicionesnestle.pdf`.

Este proyecto replica el flujo del caso «Todo Claro» pero adaptado a un documento real de
consumo masivo, y experimenta con los parámetros del modelo (temperatura) y del split de
texto (`chunk_size`, `chunk_overlap`, `k`) para observar cómo cambian los resultados.

---

## 📋 Descripción del proyecto

El asistente recibe el PDF con los términos y condiciones de la promoción, lo divide en
fragmentos, genera *embeddings* con OpenAI, los almacena en una base vectorial **Chroma** y
construye una cadena RAG que recupera los fragmentos más relevantes para responder una
pregunta. Todas las respuestas se generan **únicamente con base en el contexto recuperado**,
en español latinoamericano y con un máximo de 4 párrafos.

El flujo completo se ejecuta en **cuatro escenarios** con configuraciones distintas para
comparar el impacto de los parámetros sobre la calidad, precisión y estilo de las respuestas.
La pregunta base usada en todos los escenarios es:

> *¿Cuáles son las condiciones o situaciones en las que una persona participante puede ser
> descalificada o perder su participación en la promoción?*

Los resultados de cada ejecución (respuesta + fragmentos recuperados) se guardan como JSON en
`app/resultados_nestle/`, junto con un análisis comparativo automático.

---

## 🎯 Objetivo

- Construir un pipeline RAG funcional sobre un documento legal real de consumo masivo.
- Entender cómo el **tamaño de fragmento** y el **solapamiento** afectan el contexto recuperado.
- Analizar cómo el número de fragmentos recuperados (`k`) cambia la completitud de la respuesta.
- Comparar el efecto de la **temperatura** (0.1 vs 0.8) sobre el estilo, la fidelidad al
  documento y el riesgo de alucinaciones.

---

## ✨ Características destacadas

- **RAG end-to-end:** carga de PDF → split → embeddings → vector store → retriever → cadena.
- **Cuatro escenarios parametrizados** ejecutados de forma automática y reproducible.
- **Separación de almacenamiento:** carpeta de persistencia y colección Chroma propias por
  escenario (`chroma_nestle_ofertas_basicos/<escenario>`), sin mezclar con prácticas anteriores.
- **Prompt especializado** en términos y condiciones de promociones de consumo masivo / Nestlé.
- **Persistencia de resultados** en JSON + generación automática de un informe comparativo
  (`analisis_taller.md`).
- **Resiliencia:** reintentos con *backoff* ante errores de *rate limit* (HTTP 429).
- **Localización automática del PDF** en las rutas habituales del proyecto.

---

## 🛠️ Tecnologías utilizadas

| Categoría | Herramienta |
|-----------|-------------|
| Lenguaje | Python 3.12 |
| Framework RAG | LangChain (`langchain`, `langchain-core`, `langchain-classic`) |
| LLM | OpenAI `gpt-4o-mini` (vía `langchain-openai`) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Base vectorial | Chroma (`langchain-chroma`, `chromadb`) |
| Split de texto | `RecursiveCharacterTextSplitter` (`langchain-text-splitters`) |
| Lectura de PDF | `pypdf` |
| Configuración | `python-dotenv` |
| Contenedores | Docker / Docker Compose |

---

## 📂 Estructura del proyecto

```
python-langchain-dev/
├── basesycondicionesnestle.pdf        # Documento fuente de la promoción Nestlé
├── chroma_nestle_ofertas_basicos/     # Persistencia Chroma (una subcarpeta por escenario)
├── requirements.txt                   # Dependencias del proyecto
├── docker-compose.yml / Dockerfile    # Entorno containerizado
├── .env                               # OPENAI_API_KEY (no versionado)
└── app/
    ├── rag_todoclaro.py               # Script principal del taller RAG Nestlé
    └── resultados_nestle/             # Salidas de cada ejecución
        ├── escenario_base.json
        ├── escenario_1.json
        ├── escenario_2.json
        ├── escenario_3.json
        └── analisis_taller.md         # Informe comparativo generado automáticamente
```

> **Nota:** el script principal conserva el nombre `rag_todoclaro.py` pero su contenido ya
> está adaptado al caso Nestlé (ruta del PDF, `persist_directory`, `collection_name`, prompt
> y pregunta).

### Configuración de los escenarios

| Escenario | `chunk_size` | `chunk_overlap` | `k` | `temperature` | Fragmentos indexados |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Base | 1000 | 200 | 4 | 0.1 | 35 |
| Escenario 1 | 350 | 20 | 2 | 0.1 | 90 |
| Escenario 2 | 1000 | 200 | 6 | 0.1 | 35 |
| Escenario 3 | 1000 | 200 | 6 | 0.8 | 35 |

### Cómo ejecutar

```bash
# 1. Configurar la clave en .env
echo "OPENAI_API_KEY=sk-..." > .env

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el taller completo (los 4 escenarios)
python app/rag_todoclaro.py
```

> ⚠️ Requiere créditos activos de la API de OpenAI.

---

## 📝 Respuestas de la actividad

*(Basadas en las ejecuciones reales guardadas en `app/resultados_nestle/`.)*

### Pregunta 1 — Escenario base (`chunk_size=1000`, `overlap=200`, `k=4`, `temp=0.1`)

**Beneficios de los fragmentos largos con traslape considerable:**

Con `chunk_size=1000` el PDF se dividió en solo **35 fragmentos**, y cada uno alcanzó a
contener bloques completos del texto legal. Esto fue decisivo para la pregunta sobre
descalificación: un mismo fragmento agrupó varias causas relacionadas (los «hackers», los
«caza promociones», la presunción de fraude, los intentos de alterar la operación de la
promoción). El traslape de 200 caracteres evitó que esas listas de condiciones se cortaran de
forma abrupta entre fragmentos vecinos, conservando la continuidad de las frases. Como
resultado, la respuesta del escenario base fue **completa y coherente**, integrando causas
provenientes de las páginas 4 y 5 sin perder el hilo legal.

**Limitaciones observadas:**

Como el documento solo produce 35 fragmentos, recuperar `k=4` implica traer una porción
amplia del total; algunos fragmentos recuperados aportaban información solo parcialmente
relacionada con la descalificación (por ejemplo, condiciones sobre la entrega de premios o
gastos no reembolsables), lo que introduce «ruido» en el contexto. Además, fragmentos grandes
consumen más *tokens* del prompt, y el límite de 4 párrafos obliga a condensar, por lo que
algunos matices presentes en el contexto pueden quedar resumidos u omitidos.

### Pregunta 2 — Escenario 1 (`chunk_size=350`, `overlap=20`, `k=2`)

Al reducir el tamaño del fragmento, el PDF pasó de **35 a 90 fragmentos**: piezas mucho más
numerosas y granulares. El traslape mínimo (20) redujo la redundancia entre fragmentos
vecinos, pero aumentó el riesgo de **cortar listas o condiciones** que ocupan varias líneas,
fragmentando ideas que en el escenario base aparecían juntas.

El mayor impacto vino de combinar fragmentos pequeños con `k=2`: el modelo recibió mucho menos
contexto. La respuesta siguió identificando causas válidas de descalificación —el uso de
sistemas automáticos/semiautomáticos para obtener ventaja, los «hackers» y el no poder ganar
en más de una tienda—, pero **quedó más incompleta** que la del escenario base: no mencionó la
presunción de fraude, los «caza promociones» ni los intentos de alterar la operación de la
promoción, simplemente porque esos fragmentos no fueron recuperados. En conclusión, los chunks
pequeños mejoran la *precisión local* de cada fragmento (más enfocado), pero con `k` bajo
sacrifican *cobertura*, reduciendo la exactitud global de una respuesta que requiere reunir
varias causales dispersas en el documento.

### Pregunta 3 — Temperatura 0.1 (escenario 2) vs 0.8 (escenario 3)

Ambos escenarios usaron exactamente la misma recuperación (`chunk_size=1000`, `overlap=200`,
`k=6`), de modo que la única variable fue la temperatura.

- **Estilo de redacción:** con `temp=0.1` (escenario 2) la respuesta es directa, enumerativa y
  con poca variación léxica, muy pegada a la formulación del documento. Con `temp=0.8`
  (escenario 3) la redacción es algo más explicativa, con mayor parafraseo y conectores, aunque
  en esta ejecución ambas respuestas resultaron de extensión y contenido similares.

- **Fidelidad al documento:** la versión de `temp=0.1` se apega de forma más literal a las
  causas listadas en los fragmentos (incluye detalles específicos como la *demostradora
  capacitada en sitio*, el premio no reclamado o el fallecimiento del participante). La de
  `temp=0.8` mantiene las causas correctas pero tiende a **generalizar y reorganizar**, con un
  poco más de riesgo de incluir inferencias no explícitas en el texto fuente.

- **Cuándo usar cada una:** para consultas legales, de *compliance* o respuestas auditables
  —como las de este taller— conviene `temp=0.1`, por su mayor fidelidad y menor riesgo de
  alucinaciones. La `temp=0.8` es preferible para borradores explicativos o material divulgativo
  dirigido a usuarios finales, siempre con revisión humana posterior.

---

## 👥 Autores

- **Valentina Mancilla** 
