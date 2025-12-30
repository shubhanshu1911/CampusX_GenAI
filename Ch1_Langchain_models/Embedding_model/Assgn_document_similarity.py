# document_similarity_google.py
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()  # expects GEMINI_API_KEY or GOOGLE_API_KEY in your .env

client = genai.Client()

# 1) Embed your document collection (cricket bios)
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# 2) Embed query
query = "tell me about Bumrah"

# 3) Fetch embeddings from Gemini
all_texts = [query] + documents

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=all_texts,
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)

# 4) Convert to numpy arrays
embeddings = [np.array(e.values) for e in response.embeddings]
query_emb, doc_embs = embeddings[0], embeddings[1:]

# 5) Compute similarities
scores = cosine_similarity([query_emb], doc_embs)[0]
idx = int(np.argmax(scores))
best_doc = documents[idx]
best_score = scores[idx]

# 6) Output result
print(f"Query: {query}")
print(f"Most similar document: {best_doc}")
print(f"Similarity score: {best_score:.4f}")

########################################################################################

# document_similarity_openAI embedding model

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about bumrah'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)



