# file: retrievers.py
from supabase import Client
from typing import List, Dict
from base import BaseRetriever
from sentence_transformers import SentenceTransformer, models
from pyvi.ViTokenizer import tokenize
from utils.db_services import get_supabase
import os
from dotenv import load_dotenv

load_dotenv()
# model = SentenceTransformer("Duchaha/BGE_M3_Finetune")
# cache_folder = os.getenv("CACHE_FOLDER")    
# model = SentenceTransformer(cache_folder)

# Load model đã fine-tune với CLS pooling
word_embedding_model = models.Transformer('Duchaha/BGE_M3_Finetune')

# Dùng CLS token để tạo embeddings cho câu
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode="cls")

# Kết hợp lại thành mô hình SentenceTransformer
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


class SupabaseRetriever(BaseRetriever):
    def __init__(self):
        self.supabase : Client = get_supabase()

    async def retrieve(self, query: str, match_count=3, match_threshold=0.4) -> List[Dict]:
        # sentence = tokenize(query)
        sentence = query
        embedding = model.encode(sentence).tolist()
        response = self.supabase.rpc(
            'match_documents_v2',
            {
                'query_embedding': embedding, 
                'match_count': match_count,
                'match_threshold': match_threshold
            }
        ).execute()
        return response.data if hasattr(response, 'data') else []
