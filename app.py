import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from io import BytesIO
import os

load_dotenv()  # This loads the .env file at the application start

class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}

class DocumentVectorStore:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

def init_resources():
    if 'db' in st.session_state:
        schema_info = get_table_info(st.session_state.db)
        db_context = {
            'schema': schema_info,
            'db_connection': st.session_state.db
        }
    else:
        db_context = None

    if 'pdf_text' in st.session_state:
        document_vector_store = get_vectorstore_from_text(st.session_state.pdf_text)
        doc_context = {'vector_store': document_vector_store}
    else:
        doc_context = None

    return db_context, doc_context

def search_document(vector_store, query):
    query_embedding = OpenAIEmbeddings().embed_query(query)
    scores = cosine_similarity([query_embedding], vector_store.embeddings)[0]
    top_n = 5
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return contextual_search(vector_store, query, top_indices)

def contextual_search(vector_store, query, top_indices):
    query_keywords = set(query.lower().split())
    best_score = -1
    best_match_index = None

    for idx in top_indices:
        text = vector_store.documents[idx].page_content.lower()
        words = text.split()
        score = 0

        for i, word in enumerate(words):
            if word in query_keywords:
                score += 1
                proximity_score = len(words) - i
                score += proximity_score / len(words)

        if score > best_score:
            best_score = score
            best_match_index = idx

    return vector_store.documents[best_match_index].page_content if best_match_index is not None else "No relevant match found"

def handle_query(user_query, db_context, doc_context, chat_history):
    if "database" in user_query.lower() and db_context:
        response = get_response(user_query, db_context['db_connection'], chat_history)
    elif "document" in user_query.lower() and doc_context:
        # Use the search_document function to find relevant content in the vector store
        response = search_document(doc_context['vector_store'], user_query)
    else:
        response = "Please specify 'database' or 'document' in your query or ensure the correct setup."
    return response

def add_welcome_message(chat_history):
    welcome_message = "Hello! I'm here to help you. You can ask me questions about your database or document, and I'll do my best to provide accurate answers."
    if not chat_history:
        chat_history.append(AIMessage(content=welcome_message))

def get_text_from_pdf(uploaded_file):
    pdf_file = BytesIO(uploaded_file.getvalue())
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def get_vectorstore_from_text(text):
    document_chunks = custom_split_documents(text)
    embeddings_model = OpenAIEmbeddings()
    embeddings = [embeddings_model.embed_query(doc.page_content) for doc in document_chunks]
    return DocumentVectorStore(document_chunks, np.array(embeddings))

def custom_split_documents(text):
    words = text.split()
    max_char_length = 500
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_char_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [Document(chunk) for chunk in chunks]

def init_database(user, password, host, port, database):
    try:
        connection = mysql.connector.connect(
            host=host, user=user, password=password, database=database, port=port
        )
        return connection
    except mysql.connector.Error as err:
        st.error(f"Database connection failed: {err}")
        return None

def get_sql_chain(db):
    template = """
    You are a virtual assistant skilled in SQL. Based on the database schema and user's query, generate a SQL statement that provides the correct data.

    Database Schema: {schema}

    Recent Queries: {chat_history}

    Current Question: {question}

    Please write the SQL query from database connected that answers the above question:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),        # our OpenAI API key
        model="gpt-4-turbo",          # OpenAI model, e.g., "text-davinci-003", "gpt-4" etc.
        temperature=0.6,        # Optional: control the randomness of the output
        max_tokens=200         # Optional: maximum length of the response
    )
    return (
        RunnablePassthrough.assign(schema=lambda _: get_table_info(db))
        | prompt
        | llm
        | StrOutputParser()
    )

def get_table_info(connection):
    cursor = connection.cursor()
    cursor.execute("""
    SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = %s;""", (connection.database,))
    schema_info = cursor.fetchall()
    cursor.close()
    return "\n".join([f"Table: {row[0]}, Column: {row[1]}, Type: {row[2]}" for row in schema_info])

def get_response(user_query, db, chat_history):
    sql_chain = get_sql_chain(db)
    try:
        result = sql_chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
        response = '\n'.join(map(str, result)) if isinstance(result, list) else str(result)
        return response if response.strip() else "No SQL response generated."
    except Exception as e:
        return f"Failed to process the database query: {str(e)}"

st.set_page_config(page_title="Q&A Application", page_icon="🤖")
st.markdown("# 🤖 Q&A Application")

with st.sidebar:
    st.subheader("Database Connection")
    host = st.text_input("Host", value="localhost")
    port = st.text_input("Port", value="3306")
    user = st.text_input("User", value="user1")
    password = st.text_input("Password", type="password", value="foo")
    database = st.text_input("Database", value="data1202")
    if st.button("Connect"):
        port = int(port) if port.isdigit() else None
        if port is not None:
            db = init_database(user, password, host, port, database)
            if db:
                st.session_state.db = db
                st.success("Connected to database!")
            else:
                st.session_state.pop('db', None)
        else:
            st.error("Invalid port number.")

    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        pdf_text = get_text_from_pdf(uploaded_file)
        st.session_state.pdf_text = pdf_text
        st.session_state.doc_context = get_vectorstore_from_text(pdf_text)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

add_welcome_message(st.session_state.chat_history)

user_query = st.chat_input("Type your message here...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    db_context, doc_context = init_resources()
    response = handle_query(user_query, db_context, doc_context, st.session_state.chat_history)
    st.session_state.chat_history.append(AIMessage(content=response))

for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)