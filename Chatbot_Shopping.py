import streamlit as st
import shelve
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.title("ShopWise AI: Personalized Shopping Experience Powered by AI")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Initialize the Ollama model
if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = OllamaLLM(model="mistral")
    
# Initialize embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load PDF and set up vector store
if "retriever" not in st.session_state:
    # Load and split the PDF
    pdf_loader = PyPDFLoader("../AI-CHATBOT-SHOPPING-RECOMENDATION/Data/spring-summer-catalogue-2020.pdf")
    docs = pdf_loader.load_and_split()

    # Initialize embeddings and retriever
    vectorstore = Chroma.from_documents(
        documents=docs,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory="./chroma_db"
    )
    st.session_state["retriever"] = vectorstore.as_retriever()
    
# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

    # Add the introduction message if the history is empty
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hi, I am ShopWise AI, what can I help you to recommend for the product?"
        })

# Sidebar for deleting chat history
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        
# Define the RAG prompt template
after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)    

# Function to generate response using RAG
def query_rag(messages, retriever, model):
    # Extract the latest user query
    user_query = messages[-1]["content"]

    # # Retrieve context from retriever
    # docs = retriever.get_relevant_documents(user_query)
    # context = "\n\n".join([doc.page_content for doc in docs])

    # Construct the RAG chain
    # after_rag_chain = (
    # {
    #     "context": lambda question: "\n".join(
    #         [doc.page_content for doc in retriever.invoke(question)]
    #     ),  # Extract context from retriever
    #     "question": RunnablePassthrough(),
    # }
    # | after_rag_prompt
    # | model
    # | StrOutputParser()
    # )   
    
    #-------------------------------------------------
    
    # Retrieve context from retriever
    docs = retriever.invoke(user_query)  # Updated to use invoke
    context = "\n\n".join([doc.page_content for doc in docs])

    # Construct the input dictionary for the chain
    input_data = {"context": context, "question": user_query}
    
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda

    # Construct the RAG chain properly with RunnableLambda to handle context
    after_rag_chain = (
        {
            "context": RunnableLambda(lambda _: context),  # Wrap context in a RunnableLambda
            "question": RunnablePassthrough()  # Use passthrough for the question
        }
        | after_rag_prompt
        | model
        | StrOutputParser()
    )

    # Generate the response
    return after_rag_chain.invoke(user_query)

# Main chat interface
if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # Process user query through RAG
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""

        try:
            full_response = query_rag(
                st.session_state.messages,
                st.session_state["retriever"],
                st.session_state["ollama_model"]
            )
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"An error occurred: {str(e)}"
            message_placeholder.markdown(full_response)

    # Append the assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)


