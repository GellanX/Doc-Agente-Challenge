import streamlit as st
import tempfile
import os
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Doc Agent Q&A", page_icon="🤖", layout="wide")
st.title("🤖 Agente Interactivo de Documentos (RAG & Pandas)")

# Barra lateral para ingresar la API Key de Gemini
st.sidebar.header("🔑 Configuración")
api_key = st.sidebar.text_input("Google Gemini API Key", type="password")

if not api_key:
    st.info("Ingresa tu **Gemini API Key** en la barra lateral para activar el agente.", icon="👉")
    st.stop()

# LLM con max_retries=1 para no agotar la cuota de peticiones
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", 
    google_api_key=api_key, 
    temperature=0.2, 
    max_retries=1
)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# --- 1. INICIALIZACIÓN DEL HISTORIAL DE CHAT EN SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Botón en la barra lateral para limpiar la conversación
if st.sidebar.button("🧹 Limpiar conversación"):
    st.session_state.messages = []
    st.rerun()

# Carga de archivos
uploaded_file = st.file_uploader("Sube un documento para analizar (PDF o CSV)", type=["pdf", "csv"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    # Resetear el historial si el usuario cambia de archivo
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = []

    # --- RENDERIZAR MENSAJES ANTERIORES EN LA PANTALLA ---
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # --- CASO A: PDF (RAG con FAISS + Historial) ---
    if file_type == "pdf":
        st.sidebar.success("📄 Documento PDF cargado")
        
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if "vectorstore" not in st.session_state or st.session_state.get("file_hash") != file_hash:
            with st.spinner("Indexando PDF en base de datos vectorial..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_bytes)
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                splits = text_splitter.split_documents(docs)
                
                st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
                st.session_state.file_hash = file_hash
                os.remove(tmp_path)

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Prompt adaptado para recibir el historial (chat_history)
        system_prompt = (
            "Eres un asistente preciso especializado en responder preguntas sobre el documento adjunto.\n"
            "Usa los siguientes fragmentos de contexto recuperados para responder de forma concisa.\n"
            "Si la respuesta no se encuentra en el documento, indícalo de manera clara.\n\n"
            "Contexto:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        user_query = st.chat_input("Escribe tu pregunta sobre el documento...")

        if user_query:
            # 1. Mostrar mensaje del usuario
            st.chat_message("user").write(user_query)
            
            # 2. Convertir el historial de session_state al formato que entiende LangChain
            langchain_history = []
            for m in st.session_state.messages:
                if m["role"] == "user":
                    langchain_history.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    langchain_history.append(AIMessage(content=m["content"]))

            try:
                with st.spinner("Procesando tu consulta..."):
                    response = rag_chain.invoke({
                        "input": user_query,
                        "chat_history": langchain_history
                    })
                    answer = response.get("answer", "No se obtuvo respuesta.")
                    
                    # 3. Mostrar respuesta del asistente
                    st.chat_message("assistant").write(answer)
                    
                    # 4. Guardar ambos mensajes en st.session_state
                    st.session_state.messages.append({"role": "user", "content": user_query})
                    st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.warning("⏳ Alcanzaste el límite de peticiones de Gemini (15 RPM). Espera unos 15 segundos.")
                else:
                    st.error(f"Ocurrió un error inesperado: {e}")

    # --- CASO B: CSV (Agente Pandas + Historial) ---
    elif file_type == "csv":
        st.sidebar.success("📊 Archivo CSV cargado")
        df = pd.read_csv(uploaded_file)

        with st.expander("👁️ Vista previa de la tabla", expanded=False):
            st.dataframe(df.head(5), use_container_width=True)

        pandas_agent = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type="tool-calling",
            verbose=True,
            allow_dangerous_code=True,
            max_iterations=7,
            handle_parsing_errors=True
        )

        user_query = st.chat_input("¿Qué deseas saber o calcular sobre el CSV?")
        if user_query:
            st.chat_message("user").write(user_query)

            # Para el agente de Pandas construimos el prompt concatenando el contexto reciente
            # (El agente opera mejor si le formateamos la conversación explícitamente en el prompt)
            formatted_history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-4:]])
            query_with_context = f"Historial previo:\n{formatted_history}\n\nPregunta actual: {user_query}" if formatted_history else user_query

            with st.spinner("Analizando datos..."):
                try:
                    response = pandas_agent.invoke({"input": query_with_context})
                    answer = response.get("output", "Sin respuesta.")
                    
                    st.chat_message("assistant").write(answer)
                    
                    # Guardar en session_state
                    st.session_state.messages.append({"role": "user", "content": user_query})
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        st.warning("⏳ Alcanzaste el límite de peticiones de Gemini. Espera unos segundos e intenta nuevamente.")
                    else:
                        st.error(f"Error al procesar la consulta: {e}")