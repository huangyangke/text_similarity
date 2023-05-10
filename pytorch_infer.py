from langchain.embeddings import HuggingFaceEmbeddings
import torch
import sentence_transformers
import numpy as np
from onnxsim import simplify
import onnx

# 可选模型列表
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

class EmbeddingModel():
    def __init__(self, 
                model_name = "GanymedeNil/text2vec-large-chinese", 
                cache_fold='checkpoints', # 下载的权重文件保存路径
                device ='cuda'):
        # 初始化embedding模型
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                cache_folder=cache_fold,
                                model_kwargs={'device': device})
    
    # 得到query文本向量
    def embed_query(self, text:str):
        query_result = self.embeddings.embed_query(text)#1024
        return query_result
    
    # 得到text chunks之后的chunks向量
    def embed_documents(self, documents:list):
        doc_result = self.embeddings.embed_documents(documents)#num_chunks,1024
        return doc_result
    
# v1 (n, feat_dim) 未归一化
# v2 (m, feat_dim) 未归一化
# return (n, m)
def get_cos_similar_matrix(v1, v2, l2_normalize = False):
    res = np.dot(v1, np.array(v2).T)  # 向量点乘
    if l2_normalize:
        denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
        res = res / denom
    res[np.isneginf(res)] = 0
    return res # -1~1
    #return 0.5 + 0.5 * res 0~1    
if __name__ == '__main__':
    # embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict['text2vec'],
    #                                cache_folder='checkpoints',
    #                                model_kwargs={'device': "cuda" })
    # text = "This is a test document."
    # query_result = embeddings.embed_query(text)#1024
    # doc_result = embeddings.embed_documents([text,text])#num_chunks,1024
    model = EmbeddingModel(cache_fold='checkpoints')
    text1 = "北京的天气怎么样"
    text2 = "中国首都的天气怎样"
    
    text3 = "如何更换花呗绑定银行卡"
    text4 = "花呗更改绑定银行卡"
    embedding1 = model.embed_query(text1)
    embedding2 = model.embed_query(text2)
    sim = get_cos_similar_matrix(np.array(embedding1)[np.newaxis], np.array(embedding2)[np.newaxis], True)
    print(sim)

    embedding1 = model.embed_documents([text3,text1])
    embedding2 = model.embed_documents([text2,text4])
    sim = get_cos_similar_matrix(np.array(embedding1), np.array(embedding2), True)
    print(sim)
