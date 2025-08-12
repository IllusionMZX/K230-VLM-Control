import torch
import socket
import struct
import os
from datetime import datetime
import gradio as gr
import glob
import time
from threading import Thread
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq
import base64
import requests
import json
import re
from PIL import Image, ImageDraw, ImageFont

# 豆包API KEY配置（使用环境变量）
DOU_BAO_API_KEY = os.environ.get("ARK_API_KEY", "YOUR_DOUBAO_API_KEY")

SERVER_HOST = '192.168.43.144'
SERVER_PORT = 9090
SAVE_DIR = './recv_frames'
os.makedirs(SAVE_DIR, exist_ok=True)

# 全局保存所有已连接客户端socket
client_sockets = []

# 用于存储当前prompt_box的值
latest_prompt_value = "Describe the image concisely."

# 启动socket接收线程
def recv_frames():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((SERVER_HOST, SERVER_PORT))
    s.listen(1)
    print(f'Server listening on {SERVER_HOST}:{SERVER_PORT}')
    global latest_prompt_value
    while True:
        conn, addr = s.accept()
        print(f'Client connected: {addr}')
        client_sockets.append(conn)
        try:
            while True:
                raw_len = b''
                while len(raw_len) < 4:
                    chunk = conn.recv(4 - len(raw_len))
                    if not chunk:
                        break
                    raw_len += chunk
                if len(raw_len) < 4:
                    break
                img_len = int.from_bytes(raw_len, 'big')
                img_bytes = b''
                while len(img_bytes) < img_len:
                    chunk = conn.recv(img_len - len(img_bytes))
                    if not chunk:
                        break
                    img_bytes += chunk
                if len(img_bytes) < img_len:
                    break
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                fname = os.path.join(SAVE_DIR, f'frame_{ts}.jpg')
                with open(fname, 'wb') as f:
                    f.write(img_bytes)
                print(f'Saved frame: {fname} ({img_len} bytes)')
        except Exception as e:
            print(f'Error: {e}')
        finally:
            try:
                client_sockets.remove(conn)
            except Exception:
                pass
            conn.close()
            print(f'Client disconnected: {addr}')



import threading
Thread(target=recv_frames, daemon=True).start()




# ====== CUDA/设备调试信息 ======
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch.cuda.current_device():", torch.cuda.current_device())
    print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))

# 加载Qwen2.5-VL-3B-Instruct（4bit量化，官方推荐BitsAndBytesConfig写法）
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DOUBAO_MODEL_NAME = "doubao-1-5-vision-pro-32k-250115"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_NAME,
    device_map="cuda",
    quantization_config=bnb_config
)
qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_NAME,use_fast=True)
print("Qwen2.5-VL-3B-Instruct model device:", next(qwen_model.parameters()).device)


def get_latest_image():
    files = sorted(glob.glob(os.path.join(SAVE_DIR, '*.jpg')), reverse=True)
    return files[0] if files else None

def vlm_infer(prompt, interval_ms):
    latest_img = get_latest_image()
    if not latest_img:
        return "No image received from K230 yet."
    mode = getattr(vlm_infer, 'selected_mode', 'Local')
    if mode == 'API':
        # 只支持API模型（目前仅Doubao）
        with open(latest_img, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode()
        url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DOU_BAO_API_KEY}"
        }
        data = {
            "model": DOUBAO_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            if 'choices' in result and result['choices']:
                print("[Doubao API] 调用成功，返回内容：", result['choices'][0]['message']['content'])
                return result['choices'][0]['message']['content']
            else:
                print("[Doubao API] 返回异常：", result)
                return str(result)
        except Exception as e:
            print(f"[Doubao API] 调用失败: {e}")
            return f"Doubao API error: {e}"
    else:
        # 只保留Qwen2.5-VL-3B-Instruct本地推理
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": latest_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = qwen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(qwen_model.device)
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)  # 增加max_new_tokens用于JSON输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        time.sleep(interval_ms / 1000)
        return output_text[0]

def draw_bounding_boxes(image_path, detection_result):
    """
    在图片上绘制目标检测框，自动提取markdown中的JSON部分
    """
    try:
        # 自动提取markdown格式中的JSON数组
        json_str = None
        if isinstance(detection_result, str):
            match = re.search(r"```json\s*([\s\S]+?)```", detection_result)
            if match:
                json_str = match.group(1).strip()
            else:
                json_start = detection_result.find('[')
                json_end = detection_result.rfind(']')
                if json_start != -1 and json_end != -1:
                    json_str = detection_result[json_start:json_end+1]
                else:
                    json_str = detection_result.strip()
        else:
            json_str = detection_result

        detections = json.loads(json_str)
        if not isinstance(detections, list):
            return None
        
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow', 'magenta']
        
        for i, detection in enumerate(detections):
            if 'bbox_2d' in detection and 'label' in detection:
                bbox = detection['bbox_2d']
                label = detection['label']
                sub_label = detection.get('sub_label', '')
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
                color = colors[i % len(colors)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                label_text = f"{label}({sub_label})" if sub_label else label
                draw.text((x1, max(0, y1-20)), label_text, fill=color)
        
        # 修改保存路径
        output_dir = os.path.join(os.getcwd(), "Detection_output")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path).replace('.jpg', '_with_boxes.jpg')
        output_path = os.path.join(output_dir, base_name)
        img.save(output_path)
        return output_path
        
    except Exception as e:
        print(f"[Detection] 绘制检测框失败: {e}")
        return None

with gr.Blocks() as demo:
    gr.Markdown("# K230 VLM Interaction App")
    with gr.Row():
        img_box = gr.Image(label="Latest Frame", interactive=False)
        detection_img_box = gr.Image(label="Detection Result", interactive=False, visible=False)
    
    mode_selector = gr.Dropdown(
        choices=["Local", "API"],
        value="Local",
        label="推理模式 (Local/API)"
    )
    control_mode_checkbox = gr.Checkbox(label="控制模式 (用于控制K230舵机/LED)", value=False)
    detection_mode_checkbox = gr.Checkbox(label="目标识别框模式", value=False)
    
    model_selector = gr.Dropdown(
        choices=[QWEN_MODEL_NAME],
        value=QWEN_MODEL_NAME,
        label="模型选择"
    )
    system_prompt_box = gr.Textbox(
        label="系统Prompt (仅控制模式下生效)", 
        value="你是一个根据手势识别的控制器，当识别到握拳手势时控制舵机旋转90度。", 
        interactive=True, 
        visible=True
    )
    prompt_box = gr.Textbox(label="用户Prompt:", value="Describe the image concisely.")
    response_box = gr.Textbox(label="Response:")
    interval_slider = gr.Slider(label="Interval between 2 requests:", minimum=0, maximum=5000, step=100, value=1000)
    start_btn = gr.Button("Start")
    gr.Markdown("**当前模式：** 普通模式/控制模式/目标识别框模式可切换。")

    def update_img():
        latest_img = get_latest_image()
        return latest_img, None

    # 新增：run_vlm增加control_mode参数
    def extract_control_rule(system_prompt):
        # 尝试提取"当识别到X时控制舵机旋转Y度"或"当识别到X时控制LED灯闪烁"
        # 返回[(trigger, action)]
        rules = []
        # 舵机规则
        for m in re.finditer(r'当识别到(.+?)时控制舵机旋转([\-\d]+)度', system_prompt):
            trigger = m.group(1).strip()
            angle = m.group(2).strip()
            rules.append((trigger, f'servo:{angle}'))
        # LED规则
        for m in re.finditer(r'当识别到(.+?)时控制LED灯闪烁', system_prompt):
            trigger = m.group(1).strip()
            rules.append((trigger, 'led:blink'))
        return rules

    def run_vlm(user_prompt, interval_ms, control_mode, system_prompt, detection_mode):
        global latest_prompt_value
        
        # 目标识别框模式
        if detection_mode:
            detection_prompt = 'Detect all objects in the image and return their locations in JSON format. The format should be like [{"bbox_2d": [x1, y1, x2, y2], "label": "object_name", "sub_label": "additional_info"}]. Only return the JSON array without any additional text. Markdown format with ````json at the beginning.'
            prompt = detection_prompt
        # 控制模式下拼接系统prompt
        elif control_mode:
            prompt = f"[系统指令]{system_prompt}\n[用户指令]{user_prompt}"
        else:
            prompt = user_prompt
            
        latest_prompt_value = prompt
        
        # 读取当前选择的模式和模型
        mode = getattr(run_vlm, 'selected_mode', 'Local')
        vlm_infer.selected_mode = mode
        if mode == 'Local':
            vlm_infer.selected_model = getattr(run_vlm, 'selected_model', QWEN_MODEL_NAME)
        else:
            vlm_infer.selected_model = getattr(run_vlm, 'selected_model', DOUBAO_MODEL_NAME)
        
        # 推理
        result = vlm_infer(prompt, interval_ms)
        
        # 目标识别框模式：绘制检测框
        if detection_mode:
            latest_img = get_latest_image()
            if latest_img:
                detection_img_path = draw_bounding_boxes(latest_img, result)
                if detection_img_path:
                    return result, gr.update(value=detection_img_path, visible=True)
                else:
                    return result, gr.update(visible=True)
            return result, gr.update(visible=True)
        
        # 控制模式下自动规则提取和模糊匹配
        elif control_mode:
            rules = extract_control_rule(system_prompt)
            for trigger, action in rules:
                # 模糊匹配（忽略大小写，包含即可）
                if trigger.lower() in result.lower():
                    ctrl_cmd = action.encode("ascii")
                    for conn in client_sockets[:]:
                        try:
                            conn.sendall(len(ctrl_cmd).to_bytes(4, 'big'))
                            conn.sendall(ctrl_cmd)
                            print(f"[Control] Sent {action} to client (trigger: {trigger})")
                        except Exception as e:
                            print(f"[Control] Error sending control cmd: {e}")
                            try:
                                client_sockets.remove(conn)
                            except Exception:
                                pass
            return result, gr.update(visible=False)
        
        # 普通模式
        return result, gr.update(visible=False)

    def on_model_change(model_name):
        run_vlm.selected_model = model_name
        vlm_infer.selected_model = model_name
        return

    def on_mode_change(mode):
        run_vlm.selected_mode = mode
        vlm_infer.selected_mode = mode
        if mode == 'Local':
            return gr.update(choices=[QWEN_MODEL_NAME], value=QWEN_MODEL_NAME, interactive=True)
        else:
            return gr.update(choices=[DOUBAO_MODEL_NAME], value=DOUBAO_MODEL_NAME, interactive=True)

    # 控制模式下系统prompt可用，否则禁用
    def update_system_prompt_visibility(control_mode):
        return gr.update(interactive=control_mode)

    # 目标识别框模式切换时的界面更新
    def update_detection_mode_visibility(detection_mode):
        return gr.update(visible=detection_mode)

    mode_selector.change(fn=on_mode_change, inputs=mode_selector, outputs=model_selector)
    model_selector.change(fn=on_model_change, inputs=model_selector, outputs=None)
    start_btn.click(fn=update_img, outputs=[img_box, detection_img_box])
    
    control_mode_checkbox.change(fn=update_system_prompt_visibility, inputs=control_mode_checkbox, outputs=system_prompt_box)
    detection_mode_checkbox.change(fn=update_detection_mode_visibility, inputs=detection_mode_checkbox, outputs=detection_img_box)
    
    start_btn.click(
        fn=run_vlm, 
        inputs=[prompt_box, interval_slider, control_mode_checkbox, system_prompt_box, detection_mode_checkbox], 
        outputs=[response_box, detection_img_box], 
        api_name="run_vlm"
    )

demo.launch(share=True)
