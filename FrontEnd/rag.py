import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import datetime



#Initial setup
Time = datetime.datetime.now() # Output: Current date and time in YYYY-MM-DD HH:MM:SS.SSSSSS format
load_dotenv() #Loading Environment

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
print("Connection established")

def pdf_to_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        Reader = PdfReader(pdf)
        for page in Reader.pages:
            text += page.extract_text()
    print("Text extracted")
    return text

def text_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000 , chunk_overlap=3000)
    chunks = text_splitter.split_text(text)
    print("Broken Into Chunks")
    return chunks
def vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectors_store = FAISS.from_texts(text_chunks, embeddings)
    vectors_store.save_local("faiss_index")
    print("Vector Embeddings saved")

def conversational_chain():
    prompt_template = """Answer as correct as possible with complete detail with the provided context.\n\n
    Context:\n{context}\n
    Question:\n{question}?\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model = "gemini-1.5-pro-latest",temperature = 0.9)
    prompt = PromptTemplate(template = prompt_template,input_variables = ["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    pdf_docs = [doc.page_content for doc in docs]

    # Outputting the source documents
    filename = "log.txt"
    with open(filename,"w") as file:
        file.write(f"Question:{user_question} Time:{Time}")
        for i, pdf_doc in enumerate(pdf_docs, start=1):
            file.write(f"Document {i}:")
            file.write(pdf_doc)
    st.success("Source Documents Verified")

    chain = conversational_chain()
    response = chain({"input_documents":docs,"question":user_question},return_only_outputs=True)
    print(response)
    st.write("Reply :",response["output_text"])

def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("PDF AI")
    user_question = st.text_input("Ask a Question")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Choose PDF Files",accept_multiple_files=True,type="pdf")
        if st.button("Submit and Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_to_text(pdf_docs)
                text_chunks = text_to_chunks(raw_text)
                vectorstore(text_chunks)
                st.success("Vectorization Complete")
if _name_ == "_main_":
    main()


#Query"list all the materials( materials name , net amount,Order qty, UoM, Price per unit ,Net value ),Total amount, Ship-to address,Invoice address,Date:,Delivery date:,Terms of delivery and terms payment, with serial number for every item. give this in terms of generating invoice ."