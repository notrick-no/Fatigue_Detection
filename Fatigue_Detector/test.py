import os
import paddle
import numpy as np

print(paddle.__version__)  # 确认版本号


def load_model(params_file, model_file, use_gpu=False, use_mkl=False, mkl_thread_num=4):
    # 自动适配新旧版本
    if hasattr(paddle.inference, 'Config'):
        # 新版 API (>=2.5.0)
        config = paddle.inference.Config(model_file, params_file)
        predictor_class = paddle.inference.create_predictor
    else:
        # 旧版 API (<2.5.0)
        config = paddle.fluid.core.AnalysisConfig(model_file, params_file)
        predictor_class = paddle.fluid.core.create_paddle_predictor
    
    # 通用配置
    if use_gpu:
        config.enable_use_gpu(100, 0)
    else:
        config.disable_gpu()
    
    if use_mkl and not use_gpu:
        config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(mkl_thread_num)
    
    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_ir_optim(True)
    
    return predictor_class(config)

# 加载模型
predictor = load_model(
    os.path.join('models/paddle', '__params__'),
    os.path.join('models/paddle', '__model__'),
    use_gpu=False
)

# 准备输入数据
input_data = np.random.rand(1, 3, 260, 260).astype('float32')

# 兼容新旧版本的输入处理
input_names = predictor.get_input_names()
if hasattr(predictor, 'get_input_handle'):
    # 最新版处理方式 (Paddle >= 2.6.0)
    input_handle = predictor.get_input_handle(input_names[0])
    input_handle.copy_from_cpu(input_data)
elif hasattr(predictor, 'get_inputs'):
    # 中间版本处理方式 (Paddle 2.5.x)
    input_handle = predictor.get_inputs()[0]
    input_handle.copy_from_cpu(input_data)
else:
    # 旧版处理方式
    input_tensor = predictor.get_input_tensor(input_names[0])
    input_tensor.copy_from_cpu(input_data)

# 执行预测
predictor.run()

# 兼容新旧版本的输出处理
output_names = predictor.get_output_names()
if hasattr(predictor, 'get_output_handle'):
    # 最新版处理方式 (Paddle >= 2.6.0)
    output_handle = predictor.get_output_handle(output_names[0])
    result = output_handle.copy_to_cpu()
elif hasattr(predictor, 'get_outputs'):
    # 中间版本处理方式 (Paddle 2.5.x)
    output_handle = predictor.get_outputs()[0]
    result = output_handle.copy_to_cpu()
else:
    # 旧版处理方式
    output_tensor = predictor.get_output_tensor(output_names[0])
    result = output_tensor.copy_to_cpu()

print("Output shape:", result.shape)
