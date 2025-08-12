# K230 Multimodal Visual Intelligent Interaction and Hardware Control System

[English](#en) | [中文](#cn)

![系统框架图](E:\Code_Project\K230\Omni_VLM\Image\系统框架图.png)


#  <span id="en">1\. Project Overview</span>

This project builds a Vision Language Model (VLM) interactive system based on the Kendryte K230 development board and a cloud server. The K230 is responsible for real-time image acquisition and sending them to the server via a Socket. The server-side deploys a multimodal large model, which receives and analyzes the images, and interacts with the user through a Gradio web interface.

The system not only understands image content and generates text descriptions but also supports advanced functions like **object detection** and **remote control**. For example, the server can send control commands to the K230 based on image analysis results (such as recognizing a specific hand gesture) to drive a servo or light up an LED, achieving closed-loop control of smart hardware.

## 2\. System Architecture

This system consists of two parts:

- **Client (K230)**:

  - Runs the `K230-客户端程序-含硬件控制.py` script.
  - Uses the onboard camera (sensor) to capture images.
  - Connects to the server via Wi-Fi and uses the Socket protocol to send image data.
  - Listens for control commands sent from the server and uses the `machine` library to control hardware like servos and LEDs.

- **Server (PC/Cloud Server)**:

  - Runs the `服务端程序-含VLM-PC本地部署-API调用.py` script.
  - Starts a Socket service to receive image data from the K230.
  - Deploys two inference modes:
    - **Local Inference**: Uses the Qwen/Qwen2.5-VL-3B-Instruct model for inference on a local GPU.
    - **API Inference**: Calls the API of the Volcano Engine "Doubao" large model (doubao-1-5-vision-pro-32k-250115) for inference.
  - Provides a Gradio Web UI for users to view real-time images, input prompts, and get model responses.
  - Sends control commands to the K230 based on the user's operating mode in the UI.

## 3\. Main Features

- **Real-time Image Transmission**: The K230 transmits the camera feed to the server in real-time.
- **Multimodal Interactive Interface**: Users can converse with the VLM through the Gradio interface.
- **Dual Inference Engines**: Supports flexible switching between a local GPU model and a cloud-based API model.
- **Multiple Working Modes**:
  - **Normal Mode**: Describes the image or answers questions about it.
  - **Object Detection Box Mode**: Detects all objects in the image, returns their positions and labels in JSON format, and draws detection boxes on the interface.
  - **Control Mode**: Automatically triggers hardware control on the K230 based on the system prompt set by the user and the image content.
- **Hardware Remote Control**: Remotely controls the servo and LED lights connected to the K230 based on the VLM's analysis results.

## 4\. Environment Configuration

### 4.1 Server Side

- **Hardware Requirements**:

  - A PC or cloud server with an NVIDIA GPU (required for local inference mode).
  - Ensure that NVIDIA drivers and CUDA Toolkit are installed correctly.

- **Software Requirements**:

  - Python 3.8+

  - Use `pip` to install the required dependencies:

    ```bash
    pip install torch torchvision torchaudio
    pip install gradio transformers bitsandbytes accelerate sentencepiece pillow requests
    ```

    **Note**: The `torch` version needs to match the CUDA version.

- **API Key Configuration**:

  - If you need to use the API inference mode (Doubao large model), please set the `ARK_API_KEY` environment variable.

  - **Linux/macOS**:

    ```bash
    export ARK_API_KEY="YOUR_DOUBAO_API_KEY"
    ```

  - **Windows**:

    ```powershell
    $env:ARK_API_KEY="YOUR_DOUBAO_API_KEY"
    ```

    Alternatively, you can directly modify the following line in the `服务端程序-含VLM-PC本地部署-API调用.py` file:

    ```python
    # Replace "YOUR_DOUBAO_API_KEY" with your actual key
    DOU_BAO_API_KEY = os.environ.get("ARK_API_KEY", "YOUR_DOUBAO_API_KEY")
    ```

### 4.2 Client Side (K230)

- **Hardware Requirements**:
  - Kendryte K230 series development board.
  - Peripherals like a servo, LED, etc. (for testing control mode).
- **Software Requirements**:
  - MicroPython firmware for the K230.
  - Ensure the firmware includes necessary libraries such as `media`, `network`, and `machine`.

## 5\. How to Run

### Step 1: Modify Configuration

- **Server Configuration**:

  1. Open the `服务端程序-含VLM-PC本地部署-API调用.py` file.

  2. Modify `SERVER_HOST` to your server's **local area network IP address**.

     ```python
     SERVER_HOST = '192.168.43.144' # <-- Change to your server's IP
     ```

- **Client Configuration**:

  1. Open the `K230-客户端程序-含硬件控制.py` file.

  2. Modify `SERVER_IP` to your server's IP address, ensuring it matches the server configuration.

     ```python
     SERVER_IP = '192.168.43.144' # <-- Change to your server's IP
     ```

  3. Modify the Wi-Fi SSID and password in `wlan.connect()`.

     ```python
     # Change to your Wi-Fi name and password
     wlan.connect('mate', '291481578')
     ```

### Step 2: Start the Server

1. On your server, open a terminal and navigate to the project directory.

2. Run the server script:

   ```bash
   python 服务端程序-含VLM-PC本地部署-API调用.py
   ```

3. After the script starts, it will load the model and launch the Gradio service. The terminal will output information similar to the following, which includes a public access link (if `share=True`):

   ```
   Server listening on 192.168.43.144:9090
   Qwen2.5-VL-3B-Instruct model device: cuda:0
   Running on local URL:  http://127.0.0.1:7860
   Running on public URL: https://xxxxxxxxxx.gradio.live
   ```

### Step 3: Start the Client

1.  Upload the modified `K230-客户端程序-含硬件控制.py` to the SD card of the K230 development board.
2.  Run the script on the K230.
3.  Once the K230 successfully connects to Wi-Fi, it will automatically connect to the server.

## 6\. Usage Instructions

1.  **Send Image**: After the K230 successfully connects to the server, **press the physical button (GPIO21) on the development board**. The camera will take a picture and send it to the server.
2.  **Access Web UI**: Open the Gradio link (local or public) generated when the server started in your computer or mobile browser.
3.  **View Image**: After pressing the button on the K230, the captured image will be displayed in the "Latest Frame" area of the Gradio interface.
4.  **Interact with the Model**:
    - **Inference Mode**: In the "Inference Mode (Local/API)" dropdown menu, select **Local** (to use the local Qwen model) or **API** (to use the Doubao model).
    - **Normal Mode**: In the "User Prompt" input box, enter your question (e.g., "Describe this image in detail"), then click the "Start" button. The model's response will be displayed in the "Response" area.
    - **Object Detection Box Mode**:
      - Check the "Object Detection Box Mode" checkbox.
      - Click the "Start" button.
      - The model will return information about the detected objects (in JSON format) and display the image with bounding boxes in the "Detection Result" area.
    - **Control Mode**:
      - Check the "Control Mode" checkbox.
      - Define the control rules in the "System Prompt", for example: control the servo to rotate 90 degrees when a fist gesture is recognized.
      - Enter a trigger command in the "User Prompt", for example: analyze the gesture in the picture.
      - Click the "Start" button. If the model recognizes a condition in the image that satisfies the system prompt (e.g., "fist"), the server will automatically send a control command (`servo:90`) to the K230. Upon receiving the command, the K230 will rotate the servo.

## 7\. File Description

- `服务端程序-含VLM-PC本地部署-API调用.py`: **Core server script**. Contains the Socket service, model loading, Gradio interface logic, and inference processing.
- `K230-客户端程序-含硬件控制.py`: **Core client script**. Runs on the K230 and is responsible for image acquisition, sending, and receiving control commands.


---



##  <span id="cn">1. 项目概述</span>

本项目构建了一个基于勘智 K230 开发板与云端服务器的视觉语言模型（VLM）交互系统。 K230 端负责实时采集图像并通过 Socket 将其发送到服务器；服务器端部署了多模态大模型，负责接收并分析图像，并通过一个 Gradio Web 界面与用户进行交互。

该系统不仅能理解图像内容并生成文字描述，还支持 **目标检测** 、 **远程控制** 等高级功能，例如：服务器可以根据图像分析结果（如识别到特定手势）向 K230 发送控制指令，以驱动舵机或点亮 LED，实现智能硬件的闭环控制。

## 2. 系统架构

本系统由两部分组成：

- **客户端 (K230)**:
  - 运行 `K230-客户端程序-含硬件控制.py` 脚本。
  - 使用板载摄像头 (sensor) 采集图像。
  - 通过 Wi-Fi 连接到服务器，并使用 Socket 协议发送图像数据。
  - 监听服务器下发的控制指令，并调用 `machine` 库控制舵机 (Servo) 和 LED 等硬件。
  
- **服务器端 (PC/Cloud Server)**:
  - 运行 `服务端程序-含VLM-PC本地部署-API调用.py` 脚本。
  - 启动一个 Socket 服务，用于接收来自 K230 的图像数据。
  - 部署了两种推理模式：
    - **本地推理** : 使用 Qwen/Qwen2.5-VL-3B-Instruct 模型在本地 GPU 上进行推理。
    - **API 推理** : 调用火山方舟“豆包”大模型的 API (doubao-1-5-vision-pro-32k-250115) 进行推理。
  - 提供一个 Gradio Web UI，方便用户查看实时图像、输入 Prompt 并获取模型响应。
  - 根据用户在 UI 上的操作模式，向 K230 发送控制指令。

## 3. 主要功能

- **实时图像传输**: K230 将摄像头画面实时传输到服务器。
- **多模态交互界面**: 用户可通过 Gradio 界面与 VLM 进行对话。
- **双推理引擎**: 支持在本地 GPU 模型和云端 API 模型之间灵活切换。
- **多种工作模式**:
  - **普通模式**: 对图像进行描述或问答。
  - **目标识别框模式**: 检测图像中的所有物体，并以 JSON 格式返回其位置和标签，同时在界面上绘制检测框。
  - **控制模式**: 根据用户设定的系统指令（System Prompt）和图像内容，自动触发对 K230 硬件的控制。
- **硬件远程控制**: 可通过 VLM 的分析结果，远程控制 K230 连接的舵机和 LED 灯。

## 4. 环境配置

### 4.1 服务器端

- **硬件要求**:
  - 一台拥有 NVIDIA GPU 的 PC 或云服务器（本地推理模式需要）。
  - 确保已正确安装 NVIDIA 驱动和 CUDA Toolkit。
- **软件要求**:
  
  - Python 3.9+
  - 使用 `pip` 安装所需依赖库：
    ```bash
    pip install torch torchvision torchaudio
    pip install gradio transformers bitsandbytes accelerate sentencepiece pillow requests
    ```
    **注意**: `torch` 版本需要与 CUDA 版本匹配。
- **API Key 配置**:
  - 如果需要使用 API 推理模式（豆包大模型），请设置环境变量 `ARK_API_KEY`。
  - **Linux/macOS**:
    ```bash
    export ARK_API_KEY="YOUR_DOUBAO_API_KEY"
    ```
  - **Windows**:
    ```powershell
    $env:ARK_API_KEY="YOUR_DOUBAO_API_KEY"
    ```
    或者，您可以直接在 `服务端程序-含VLM-PC本地部署-API调用.py` 文件中修改以下代码行：
    ```python
    # 将 "YOUR_DOUBAO_API_KEY" 替换为您的真实密钥
    DOU_BAO_API_KEY = os.environ.get("ARK_API_KEY", "YOUR_DOUBAO_API_KEY")
    ```

### 4.2 客户端 (K230)

- **硬件要求**:
  - 勘智 K230 系列开发板。
  - 舵机、LED 等外设（用于测试控制模式）。
- **软件要求**:
  - K230 的 MicroPython 固件。
  - 确保固件包含 `media`, `network`, `machine` 等必要的库。

## 5. 运行方式

### 步骤 1: 修改配置

- **服务器配置**:
  1.  打开 `K230-客户端程序-含硬件控制.py` 文件。
  2.  修改 `SERVER_HOST` 为您服务器的 **局域网 IP 地址**。
      ```python
      SERVER_HOST = '192.168.43.144' # <-- 修改为您的服务器 IP
      ```
- **客户端配置**:
  
  1.  打开 `服务端程序-含VLM-PC本地部署-API调用.py` 文件。
  2.  修改 `SERVER_IP` 为您服务器的 IP 地址，确保与服务器配置一致。
      ```python
      SERVER_IP = '192.168.43.144' # <-- 修改为您的服务器 IP
      ```
  3.  修改 `wlan.connect()` 中的 Wi-Fi SSID 和密码。
      ```python
      # 修改为您的 Wi-Fi 名称和密码
      wlan.connect('mate', '291481578')
      ```

### 步骤 2: 启动服务器

1.  在您的服务器上，打开终端，进入项目目录。
2.  运行服务器脚本：
    ```bash
    python 服务端程序-含VLM-PC本地部署-API调用.py
    ```
3.  脚本启动后，会加载模型并启动 Gradio 服务。 终端会输出类似以下的信息，其中包含一个公网访问链接（如果 `share=True`）：
    ```
    Server listening on 192.168.43.144:9090
    Qwen2.5-VL-3B-Instruct model device: cuda:0
    Running on local URL:  http://127.0.0.1:7860
    Running on public URL: https://xxxxxxxxxx.gradio.live
    ```

### 步骤 3: 启动客户端

1.  将修改好的 `K230-客户端程序-含硬件控制.py` 上传到 K230 开发板的 SD 卡中。
2.  在 K230 上运行该脚本。
3.  K230 连接 Wi-Fi 成功后，会自动连接到服务器。

## 6. 使用说明

1.  **发送图像**: 在 K230 成功连接到服务器后，**按下开发板上的物理按键 (GPIO21)**，摄像头会拍摄一张照片并发送到服务器。
2.  **访问 Web UI**: 在电脑或手机浏览器中打开服务器启动时生成的 Gradio 链接 (本地或公网链接)。
3.  **查看图像**: 按下 K230 按键后，拍摄的图像会显示在 Gradio 界面的 "Latest Frame" 区域。
4.  **与模型交互**:
    - **推理模式**: 在 "推理模式 (Local/API)" 下拉菜单中选择 **Local** (使用本地 Qwen 模型) 或 **API** (使用豆包模型)。
    - **普通模式**: 在 "用户 Prompt" 输入框中输入您的问题（如 "详细描述这张图片"），然后点击 "Start" 按钮。模型的回答会显示在 "Response" 区域。
    - **目标识别框模式**:
      - 勾选 "目标识别框模式" 复选框。
      - 点击 "Start" 按钮。
      - 模型会返回检测到的物体信息（JSON 格式），并在 "Detection Result" 区域显示带有边界框的图像。
    - **控制模式**:
      - 勾选 "控制模式" 复选框。
      - 在 "系统 Prompt" 中定义控制规则，例如：当识别到握拳手势时控制舵机旋转 90 度。
      - 在 "用户 Prompt" 中输入触发指令，例如：分析图中的手势。
      - 点击 "Start" 按钮。 如果模型在图像中识别到了满足系统指令的条件（如“握拳”），服务器会自动向 K230 发送控制指令 (`servo:90`)，K230 接收到指令后会转动舵机。

## 7. 文件说明

- `服务端程序-含VLM-PC本地部署-API调用.py`: **核心服务器脚本**。包含 Socket 服务、模型加载、Gradio 界面逻辑和推理处理。
- `K230-客户端程序-含硬件控制.py`: **核心客户端脚本**。运行在 K230 上，负责图像采集、发送和接收控制指令。

