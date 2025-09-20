from src.helper import load_pdf_file, text_split, download_hugging_face_embaddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()   

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# load and process the data
extracted_data = load_pdf_file(data="Data/")
text_chunks= text_split(extracted_data) 
embeddings = download_hugging_face_embaddings()

# create and store the embeddings in pinecone vector db 
pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
index_name = "medicalbot"

pc.create_index(name=index_name,
                dimension=384, 
                metric="cosine", 
                spec=ServerlessSpec(
                    cloud=  "aws",
                    region= "us-east-1",
                    )
                )


# embed eachchunk and upsert the embendings into your pincone index
docsearch = PineconeVectorStore.from_documents(
    documents =text_chunks, 
    index_name=index_name,
    embedding=embeddings, 
    )