import os
import json
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

# 往上两级（rag/ -> 项目根目录）
_RAG_DIR = os.path.dirname(os.path.abspath(__file__))       
_PROJECT_ROOT = os.path.dirname(_RAG_DIR) 

class VectorMemory:
    def __init__(
            self,
            dim=768,
            index_file="",
            triple_file="",
            emb_model_name="shibing624/text2vec-base-chinese"
    ):
        self.dim = dim
        self.index_file = index_file or os.path.join(_PROJECT_ROOT, 'memory', 'vector.faiss')
        self.triple_file = triple_file or os.path.join(_PROJECT_ROOT, 'memory', 'triples.jsonl')
        self.emb_model = SentenceTransformer(emb_model_name)
        self.index = faiss.IndexFlatL2(self.dim)
        self.triples: List[Dict] = []
        self._init_storage_files()
    
    def _init_storage_files(self):
        
        # Ensure the dir exists
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)

        if os.path.exists(self.triple_file):
            with open(self.triple_file, 'r', encoding='utf-8') as f:
                self.triples = [json.loads(line) for line in f if line.strip()]
        else:
            self.triples = []

        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
    
    def triple_to_text(self, triple_obj: Dict) -> str:

        return(
            f"record: {triple_obj.get('record', '')}\n"
            f"exam_result: {triple_obj.get('exam_result', '')}\n"
            f"diagnostic: {triple_obj.get('diagnostic', '')}"
        )
    
    def encode(self, triple: Dict) -> np.ndarray:

        text = self.triple_to_text(triple_obj=triple)
        emb = self.emb_model.encode([text])
        if emb.shape == (1, self.dim):
            return emb.astype('float32')
        else:
            raise ValueError('Embedding shape error!')
    
    def add_triple(self, triple: Dict):

        emb = self.encode(triple)
        self.index.add(emb)
        self.triples.append(triple)

        with open(self.triple_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(triple, ensure_ascii=False)+"\n")

    def save_index(self):

        faiss.write_index(self.index, self.index_file)

    def search(self, query: str, top_k=5) -> List[Dict]:

        # Step1: encode，确保输出是(1, dim)的二维float32
        q_emb = self.emb_model.encode([query])
        q_emb = np.array(q_emb, dtype='float32')

        # Step2: 强制保证二维，faiss不接受一维输入
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)

        # Step3: 确保内存连续，faiss底层C++要求contiguous array
        q_emb = np.ascontiguousarray(q_emb)

        # Step4: 正确传参，D=distances(n,k)，I=labels(n,k)
        D, I = self.index.search(q_emb, top_k)

        print(f"I[0] is {I[0]}")

        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.triples):
                results.append(self.triples[idx])
        return results

    def batch_import(self, triple_list: List[Dict]):

        texts = [self.triple_to_text(t) for t in triple_list]
        embs = self.emb_model.encode(texts).astype('float32')
        self.index.add(embs)
        self.triples.extend(triple_list)
        with open(self.triple_file, 'a', encoding='utf-8') as f:
            for triple in triple_list:
                f.write(json.dumps(triple, ensure_ascii=False)+'\n')

    def get_triple(self, idx: int) -> Dict:

        if 0 <= idx < len(self.triples):
            return self.triples[idx]
        raise IndexError(f'No triple:{idx}')
    
    
        


