import os
import tensorrt as trt
from tensorrt.tensorrt import Logger, Runtime
from tensorrt_utils import TensorRTShape, build_engine

if __name__ == '__main__':
    # onnx模型路径
    input_text_onnx_path = '/mnt/cluster/huangyangke/llm/text2vector/checkpoints/onnx_model/text2vec_large_chinese.onnx'
    # 保存trt模型路径
    save_tensorrt_path = '/mnt/cluster/huangyangke/llm/text2vector/checkpoints/trt_model/text2vec_large_chinese'
    # 动态维度设置
    flag_fp16 = False
    min_batch_size = 1
    max_batch_size = 128
    min_seq_len = 1
    max_seq_len = 512
    
    text_input_shape = [TensorRTShape((min_batch_size, min_seq_len),
                                    (min_batch_size, min_seq_len),
                                    (max_batch_size, max_seq_len), 'input_ids'),
                        TensorRTShape((min_batch_size, min_seq_len),
                                    (min_batch_size, min_seq_len),
                                    (max_batch_size, max_seq_len), 'attention_mask'),
                        TensorRTShape((min_batch_size, min_seq_len),
                                    (min_batch_size, min_seq_len),
                                    (max_batch_size, max_seq_len), 'token_type_ids')]  
    # trt初始化操作  
    trt_logger: Logger = trt.Logger(trt.Logger.INFO)
    runtime: Runtime = trt.Runtime(trt_logger)
    trt.init_libnvinfer_plugins(trt_logger, '')

    # ONNX -> TensorRT
    engine = build_engine(
        runtime=runtime,
        onnx_file_path=input_text_onnx_path,
        logger=trt_logger,
        input_shapes=text_input_shape,
        workspace_size=10000 * 1024 * 1024,
        fp16=flag_fp16,
        int8=False,
    )
    
    if not flag_fp16:
        text_fp32_trt_path = f"{save_tensorrt_path}.fp32.trt"
        print(f"Saving the text FP32 TensorRT model at {text_fp32_trt_path} ...")
        with open(text_fp32_trt_path, 'wb') as f:
            f.write(bytearray(engine.serialize()))
    else:
        text_fp16_trt_path = f"{save_tensorrt_path}.fp16.trt"
        print(f"Saving the text FP16 TensorRT model at {text_fp16_trt_path} ...")
        with open(text_fp16_trt_path, 'wb') as f:
            f.write(bytearray(engine.serialize()))
    print("Finished ONNX to TensorRT conversion...")
