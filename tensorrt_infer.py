from tensorrt_utils import TensorRTModel
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from torch import Tensor, device
from transformers import AutoTokenizer
import torch
import time
import numpy as np

class EmbeddingModel():
    def __init__(self, pytorch_model_path, tensorrt_model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pytorch_model_path, do_lower_case=False)
        self.bert = TensorRTModel(tensorrt_model_path)
        self.device = 'cuda:0'
        self.batchsize = 32
        
    def batch_to_device(self, batch):
        """
        send a pytorch batch to a device (CPU/GPU)
        """
        for key in batch:
            if isinstance(batch[key], Tensor):
                batch[key] = batch[key].to(self.device)
        return batch    
    
    def tokenize(self, texts):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        to_tokenize = [str(text).strip() for text in texts]
        output.update(self.tokenizer(to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=512))
        return output   
    
    def embed_query(self, text: str, normalize_embeddings = False) -> List[float]:
        text = text.replace("\n", " ")
        embedding = self.encode(text, normalize_embeddings)
        return embedding
    
    def embed_documents(self, texts: List[str], normalize_embeddings = False) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.encode(texts, normalize_embeddings)
        return embeddings
              
    def encode(self, sentences: Union[str, List[str]], normalize_embeddings = False):
        input_was_string = False
        # 判断输入的是否是单个句子 如果是转换成列表
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True
            
        all_embeddings = []
        # 将句子按照长度从大到小进行排序 
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]        

        for start_index in range(0, len(sentences), self.batchsize):
            sentences_batch = sentences_sorted[start_index:start_index+self.batchsize]
            features = self.tokenize(sentences_batch)
            features = self.batch_to_device(features)    
            embeddings = self.bert(features)['sentence_embeddings']
            # 特征归一化
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)            
            embeddings = embeddings.tolist()
            all_embeddings.extend(embeddings)
        
        # 还原原来的顺序    
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        
        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings
    
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
    tensorrt_model_path = '/mnt/cluster/huangyangke/llm/text2vector/checkpoints/trt_model/text2vec_large_chinese.fp32.trt'
    pytorch_model_path = 'checkpoints/GanymedeNil_text2vec-large-chinese'
    model = EmbeddingModel(pytorch_model_path, tensorrt_model_path)
    text1 = "北京的天气怎么样"
    text2 = "中国首都的天气怎样"
    
    text3 = "如何更换花呗绑定银行卡"
    text4 = "花呗更改绑定银行卡"
    embedding1 = model.embed_query(text1, True)
    embedding2 = model.embed_query(text2, True)
    sim = get_cos_similar_matrix(np.array(embedding1)[np.newaxis], np.array(embedding2)[np.newaxis])
    print(sim)
    
    embedding1 = model.embed_documents([text3,text1], True)
    embedding2 = model.embed_documents([text2,text4], True)
    sim = get_cos_similar_matrix(np.array(embedding1), np.array(embedding2))
    print(sim)
    
    ################## 速度测试
    # tokenizer = AutoTokenizer.from_pretrained(tf_from_s_path, do_lower_case=False)
    # trt_model = TensorRTModel(tensorrt_text_model)

    # # 构建输入
    # st = ['北京的天气怎么样']
    # inputs = tokenizer(
    #     st,
    #     padding=True, #填充到最长序列 单个序列不会填充
    #     truncation='longest_first', 
    #     return_tensors="pt", 
    #     max_length=512
    # )

    # for key in inputs:
    #     inputs[key] = inputs[key].cuda()
    # print(inputs) 
    # infer_time = 1
    # start = time.time()
    # for i in range(infer_time):
    #     trt_output = trt_model(inputs=inputs)
    # spend_time = time.time() - start
    # print('spend time:', spend_time)
    # print(trt_output['sentence_embeddings'].shape)
    # # print(trt_output['sentence_embeddings'].tolist())
