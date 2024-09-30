###creat 
from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

###idk kaffe
import pinecone 
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key="",  # find at app.pinecone.io
    environment="us-east-1-aws"  # next to api key in console
)
index_name = "langchain-chatbot"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
