import os
import json
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
from torch import nn
import time

class EmbeddingModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path)
        self.bert = AutoModel.from_pretrained(model_path, from_tf=False, config=config)

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 注意bert模型输入顺序
        last_hidden_state, pooler_output = self.bert(input_ids, attention_mask, token_type_ids, return_dict=False)
        sentence_embeddings = self.mean_pooling(last_hidden_state, attention_mask)
        return sentence_embeddings
        
if __name__ == '__main__':
    ####### 加载训练好的模型和tokenizer
    tf_from_s_path='checkpoints/GanymedeNil_text2vec-large-chinese'
    output_dir='checkpoints/onnx_model'
    export_model_path = os.path.join(output_dir, 'text2vec_large_chinese.onnx')

    # config = AutoConfig.from_pretrained(tf_from_s_path)
    # model = AutoModel.from_pretrained(tf_from_s_path, from_tf=False, config=config)
    tokenizer = AutoTokenizer.from_pretrained(tf_from_s_path, do_lower_case=False)
    model = EmbeddingModel(tf_from_s_path)
    #print(model)
    
    ####### 构建输入
    model.eval()
    model.to('cpu')
    # 构建输入
    st = ['我真的很喜欢你呀呀呀呀']
    inputs = tokenizer(
        st,
        padding=True, #填充到最长序列 单个序列不会填充
        truncation='longest_first', 
        return_tensors="pt", 
        max_length=512
    )
    # # print(inputs)
    # # print(inputs['input_ids'].shape) #input_ids token_type_ids attention_mask
    # # print(inputs['token_type_ids'].shape)
    # # print(inputs['attention_mask'].shape)
    # 推理次数
    infer_time = 1
    start = time.time()
    with torch.no_grad():
        for i in range(infer_time):
            output = model(**inputs)
            print(output)
    spend_time = time.time() - start
    print('spend time:', spend_time)
    # ####### 导出onnx
    # with torch.no_grad():
    #     symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
    #     torch.onnx.export(model,                                            # model being run
    #                       args=tuple(inputs.values()),                      # model input (or a tuple for multiple inputs)
    #                       f=export_model_path,                              # where to save the model (can be a file or file-like object)
    #                       opset_version=13,                                 # the ONNX version to export the model to
    #                       do_constant_folding=True,                         # whether to execute constant folding for optimization
    #                       input_names=['input_ids',                         # the model's input names
    #                                    'attention_mask',
    #                                    'token_type_ids'],
    #                       output_names=['sentence_embeddings'],                    # the model's output names
    #                       dynamic_axes={'input_ids': symbolic_names,        # variable length axes
    #                                     'attention_mask' : symbolic_names,
    #                                     'token_type_ids' : symbolic_names,
    #                                     'sentence_embeddings' : symbolic_names},)
    #     print("Model exported at ", export_model_path)    

    # ####### onnxruntime cpu推理
    # import onnxruntime
    # import numpy
    # sess_options = onnxruntime.SessionOptions()
    # sess_options.optimized_model_filepath = os.path.join(output_dir, 'text2vec_large_chinese_cpu.onnx')
    # session = onnxruntime.InferenceSession(export_model_path, sess_options, providers=['CPUExecutionProvider'])

    # ort_inputs = {k:v.cpu().numpy() for k, v in inputs.items()}
    # print(ort_inputs)
    # for _input in session.get_inputs():
    #     print(_input.name)
    # ort_outputs = session.run(output_names = ['sentence_embeddings'], 
    #                           input_feed = ort_inputs)
    # print(ort_outputs[0].tolist())
