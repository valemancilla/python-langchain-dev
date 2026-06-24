from langchain_core.prompts import PromptTemplate
import streamlit as st
import os
from langchain_openai import ChatOpenAI
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Configuración inicial
st.set_page_config(page_title="Chatbot Básico", page_icon="🤖")
st.title("🤖 Chatbot Básico con LangChain")
st.markdown("Este es un *chatbot de ejemplo* construido con LangChain + Streamlit. ¡Escribe tu mensaje abajo para comenzar!")

with st.sidebar:
    st.header("Configuración")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)
    model_name = st.selectbox("Modelo", ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini", "gemini-2.5-flash"])

    # Cargar API key según el proveedor del modelo seleccionado
    is_gemini_model = model_name.startswith("gemini")
    key_name = "GOOGLE_API_KEY" if is_gemini_model else "OPENAI_API_KEY"
    provider_name = "Google" if is_gemini_model else "OpenAI"
    api_key = os.getenv(key_name) or st.secrets.get(key_name)
    if not api_key:
        api_key = st.text_input(f"{provider_name} API Key", type="password")
        st.caption(f"También puedes configurarla en {key_name} o en .streamlit/secrets.toml")

# Inicializar el historial de mensajes en session_state
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Crear el template de prompt con comportamiento específico
prompt_template = PromptTemplate(
    input_variables=["mensaje", "historial"],
    template="""Eres un asistente útil y amigable llamado ChatBot Pro. 

Historial de conversación:
{historial}

Responde de manera clara y concisa a la siguiente pregunta: {mensaje}"""
)

# Crear cadena usando LCEL (LangChain Expression Language)
cadena = None
if api_key:
    if is_gemini_model:
        if ChatGoogleGenerativeAI is None:
            st.error("Falta la dependencia para Gemini: instala `langchain-google-genai`.")
        else:
            chat_model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)
            cadena = prompt_template | chat_model
    else:
        chat_model = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)
        cadena = prompt_template | chat_model

# Renderizar historial existente
for msg in st.session_state.mensajes:
    if isinstance(msg, SystemMessage):
        continue  # no mostrar mensajes del sistema al usuario
    
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

if st.button("🗑️ Nueva conversación"):
    st.session_state.mensajes = []
    st.rerun()

# Input de usuario
pregunta = st.chat_input("Escribe tu mensaje:")

if pregunta:
    if not cadena:
        missing_provider = "Google" if model_name.startswith("gemini") else "OpenAI"
        missing_key_name = "GOOGLE_API_KEY" if model_name.startswith("gemini") else "OPENAI_API_KEY"
        st.error(f"Falta la API Key de {missing_provider}. Configura {missing_key_name} para continuar.")
        st.stop()

    # Mostrar y almacenar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(pregunta)
    
    # Generar y mostrar respuesta del asistente
    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # Streaming de la respuesta
            for chunk in cadena.stream({"mensaje": pregunta, "historial": st.session_state.mensajes}):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
        st.session_state.mensajes.append(HumanMessage(content=pregunta))
        st.session_state.mensajes.append(AIMessage(content=full_response))
        
    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")
        error_provider = "Google" if model_name.startswith("gemini") else "OpenAI"
        error_key_name = "GOOGLE_API_KEY" if model_name.startswith("gemini") else "OPENAI_API_KEY"
        st.info(f"Verifica que tu API Key de {error_provider} ({error_key_name}) esté configurada correctamente.")
