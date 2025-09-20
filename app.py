from flask import Flask, render_template, request
from src.helper import download_hugging_face_embaddings
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from src.prompt import *

app = Flask(__name__)
load_dotenv()

# API Keys
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Embeddings + Pinecone
embeddings = download_hugging_face_embaddings()
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,
    max_output_tokens=500,
    google_api_key=GEMINI_API_KEY,
)

# Prompt + Retrieval Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
reg_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]   # ✅ FIXED
    print("User:", msg)
    response = reg_chain.invoke({"input": msg})
    print("Bot:", response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
