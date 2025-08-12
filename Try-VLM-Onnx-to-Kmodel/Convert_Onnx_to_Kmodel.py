import os
import numpy as np
from PIL import Image
import onnxsim
import onnx
import nncase

def parse_model_input_output(model_file):
    onnx_model = onnx.load(model_file)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    input_names = list(set(input_all) - set(input_initializer))
    input_tensors = [
        node for node in onnx_model.graph.input if node.name in input_names]

    inputs = []
    for e in input_tensors:
        onnx_type = e.type.tensor_type
        input_dict = {}
        # 用 helper.tensor_dtype_to_np_dtype 替代 mapping.TENSOR_TYPE_TO_NP_TYPE
        try:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(onnx_type.elem_type)
        except AttributeError:
            dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_type.elem_type]
        input_dict['name'] = e.name
        input_dict['dtype'] = dtype
        input_dict['shape'] = [dim.dim_value if dim.dim_value > 0 else 1 for dim in onnx_type.shape.dim]
        inputs.append(input_dict)
    return inputs

def onnx_simplify(model_file, dump_dir, input_shapes):
    onnx_model = onnx.shape_inference.infer_shapes(onnx.load(model_file))
    # 用 overwrite_input_shapes 替换 input_shapes
    onnx_model, check = onnxsim.simplify(onnx_model, overwrite_input_shapes=input_shapes)
    assert check, "Simplified ONNX model could not be validated"
    simplified_file = os.path.join(dump_dir, 'simplified.onnx')
    onnx.save_model(onnx_model, simplified_file)
    return simplified_file

def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        return f.read()

def generate_image_data(shape, batch, calib_dir):
    img_paths = [os.path.join(calib_dir, p) for p in os.listdir(calib_dir) if p.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in {calib_dir}")
    data = []
    for i in range(batch):
        img_data = Image.open(img_paths[i % len(img_paths)]).convert('RGB')
        img_data = img_data.resize((shape[3], shape[2]), Image.BILINEAR)
        img_data = np.asarray(img_data, dtype=np.uint8)
        img_data = np.transpose(img_data, (2, 0, 1))  # HWC->CHW
        data.append([img_data[np.newaxis, ...]])
    return data

def generate_random_data(shape, batch):
    data = []
    for i in range(batch):
        data.append([np.random.randint(0, 255, shape, dtype=np.uint8)])
    return data

def is_image_input(input_shape):
    # 判断输入 shape 是否为图片（NCHW 或 NHWC，且C=3或1，H,W>1）
    if len(input_shape) == 4 and (input_shape[1] == 3 or input_shape[1] == 1) and input_shape[2] > 1 and input_shape[3] > 1:
        return True
    return False

def convert_onnx_to_kmodel(onnx_path, kmodel_path, calib_dir):
    dump_dir = os.path.join('./tmp', os.path.splitext(os.path.basename(onnx_path))[0])
    os.makedirs(dump_dir, exist_ok=True)
    print(f"\n=== Converting {onnx_path} ===")

    # 1. 解析输入
    inputs = parse_model_input_output(onnx_path)
    input_shapes = {inp['name']: inp['shape'] for inp in inputs}
    input_shape = list(input_shapes.values())[0]
    input_name = list(input_shapes.keys())[0]

    # 2. onnx simplify（用 overwrite_input_shapes）
    try:
        model_file = onnx_simplify(onnx_path, dump_dir, {input_name: input_shape})
    except Exception as e:
        print(f"onnx simplify failed: {e}")
        return

    # 3. compile_options
    compile_options = nncase.CompileOptions()
    compile_options.target = "k230"
    compile_options.preprocess = is_image_input(input_shape)
    compile_options.input_shape = input_shape
    compile_options.input_type = 'uint8'
    compile_options.input_range = [0, 255]
    compile_options.input_layout = 'NCHW'
    compile_options.output_layout = 'NCHW'
    compile_options.dump_ir = False
    compile_options.dump_asm = False
    compile_options.dump_dir = dump_dir

    # ----------- 这里加动态shape ShapeBucket配置 -----------
    # 开启ShapeBucket
    compile_options.shape_bucket = True
    # 动态shape字段，假如你的输入是 [N, C, H, W]，想让N(或某个轴)支持动态
    # 这里举例，让第0维（batch）支持1~8，分2段
    # 具体参数要查nncase官方文档或 help(nncase.ShapeBucketOptions)
    shape_bucket_options = nncase.ShapeBucketOptions()
    shape_bucket_options.axis = 0  # 动态的维度索引，例如batch
    shape_bucket_options.min_shape = 1
    shape_bucket_options.max_shape = 8
    shape_bucket_options.segment_number = 2  # 分成2段
    compile_options.shape_bucket_options = shape_bucket_options
    # ------------------------------------------------------
    
    # 4. compiler
    compiler = nncase.Compiler(compile_options)
    model_content = read_model_file(model_file)
    import_options = nncase.ImportOptions()
    try:
        compiler.import_onnx(model_content, import_options)
    except Exception as e:
        print(f"import onnx failed: {e}")
        return

    # 5. ptq_options
    ptq_options = nncase.PTQTensorOptions()
    ptq_options.samples_count = 8
    try:
        if is_image_input(input_shape):
            ptq_options.set_tensor_data(generate_image_data(input_shape, ptq_options.samples_count, calib_dir))
        else:
            ptq_options.set_tensor_data(generate_random_data(input_shape, ptq_options.samples_count))
    except Exception as e:
        print(f"PTQ generate data failed: {e}")
        return
    try:
        compiler.use_ptq(ptq_options)
    except Exception as e:
        print(f"use_ptq failed: {e}")
        return

    # 6. compile and save
    try:
        compiler.compile()
        kmodel = compiler.gencode_tobytes()
        with open(kmodel_path, 'wb') as f:
            f.write(kmodel)
        print(f"Convert {onnx_path} -> {kmodel_path} finished.")
    except Exception as e:
        print(f"Compile failed: {e}")

if __name__ == '__main__':
    os.makedirs('./kmodel', exist_ok=True)
    # 三个模型
    model_list = [
        './SmolVLM-256M-Instruct/onnx/vision_encoder_int8.onnx',
        './SmolVLM-256M-Instruct/onnx/embed_tokens_int8.onnx',
        './SmolVLM-256M-Instruct/onnx/decoder_model_merged_int8.onnx'
    ]
    for onnx_path in model_list:
        kmodel_name = os.path.splitext(os.path.basename(onnx_path))[0] + '.kmodel'
        kmodel_path = os.path.join('./kmodel', kmodel_name)
        convert_onnx_to_kmodel(onnx_path, kmodel_path, './calib_images')