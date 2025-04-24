from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class InputText(BaseModel):
    text1: str
    text2: str

@app.post("/")
async def get_similarity_score(input_text: InputText):
    embeddings = model.encode([input_text.text1, input_text.text2], convert_to_tensor=True)
    score = float(util.pytorch_cos_sim(embeddings[0], embeddings[1])[0])
    return {"similarity score": round(score, 4)}
