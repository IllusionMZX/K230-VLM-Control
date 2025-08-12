import onnx
import onnxsim

input_path = "./SmolVLM-256M-Instruct/onnx/decoder_model_merged_int8.onnx"
output_path = "./static_onnx/decoder_model_merged_int8_static.onnx"

# 构造所有输入的静态shape
static_shapes = {
    "inputs_embeds": [1, 1, 576],
    "attention_mask": [1, 1],
    "position_ids": [1, 1],
}

# 30层的past_key/val，每个都是 [1, 3, 16, 64]
for i in range(30):
    static_shapes[f"past_key_values.{i}.key"] = [1, 3, 16, 64]
    static_shapes[f"past_key_values.{i}.value"] = [1, 3, 16, 64]

print("将使用如下静态shape配置：")
for k, v in static_shapes.items():
    print(k, v)

model = onnx.load(input_path)
model_simp, check = onnxsim.simplify(
    model,
    overwrite_input_shapes=static_shapes
)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_path)
print("已静态化完成，输出文件：", output_path)