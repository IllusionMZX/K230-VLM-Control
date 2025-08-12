'''
K230端: 用sensor.snapshot()获取当前帧,通过socket发送至VLM服务端,接收VLM服务器端控制命令,控制舵机/LED。
'''
import time, gc
from media.sensor import *
from media.display import *
from media.media import *
import network, socket
from machine import Pin, FPIOA

# 屏幕分辨率
lcd_width = 800
lcd_height = 480

# WIFI连接
WIFI_LED = Pin(52, Pin.OUT)
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
if not wlan.isconnected():
    print('connecting to network...')
    for i in range(3):
        wlan.connect('mate', '291481578')
        if wlan.isconnected():
            break
if wlan.isconnected():
    print('connect success')
    WIFI_LED.value(1)
    while wlan.ifconfig()[0] == '0.0.0.0':
        pass
    print('network information:', wlan.ifconfig())
else:
    for i in range(3):
        WIFI_LED.value(1)
        time.sleep_ms(300)
        WIFI_LED.value(0)
        time.sleep_ms(300)
    wlan.active(False)
    raise Exception('WIFI connect failed')

# 摄像头初始化
sensor = Sensor()
sensor.reset()
sensor.set_framesize(width=800, height=480) #设置帧大小为LCD分辨率(800x480)，默认通道0
sensor.set_pixformat(Sensor.RGB565)


Display.init(Display.ST7701, width=lcd_width, height=lcd_height, to_ide=True)
MediaManager.init()

sensor.run()

# Socket连接
SERVER_IP = '192.168.43.144'
SERVER_PORT = 9090
s = socket.socket()
addr = socket.getaddrinfo(SERVER_IP, SERVER_PORT)[0][-1]
s.connect(addr)
s.settimeout(0)


# 按键和LED初始化
fpioa = FPIOA()
fpioa.set_function(52, FPIOA.GPIO52)
fpioa.set_function(21, FPIOA.GPIO21)
WIFI_LED = Pin(52, Pin.OUT)  # 复用
KEY = Pin(21, Pin.IN, Pin.PULL_UP)

# 舵机初始化（参考servo.py）
fpioa.set_function(42, FPIOA.PWM0)
from machine import PWM
S1 = PWM(0, 50, 0, enable=True)

def Servo(servo, angle):
    servo.duty((angle+90)/180*10+2.5)

clock = time.clock()

last_send_time = time.time()
recv_last_time = time.time()

# 新增：用于保存服务端返回的消息
server_text = ""
# 新增：用于控制Capture显示帧数
capture_show = 0
angle = 0

import _thread
def recv_control_thread():
    global server_text
    global angle   # 加这一行
    while True:
        try:
            raw_len = s.recv(4)
            if raw_len and len(raw_len) == 4:
                msg_len = int.from_bytes(raw_len, 'big')
                msg = s.recv(msg_len)
                if msg:
                    try:
                        cmd = msg.decode('ascii')
                        # 解析控制指令
                        if cmd.startswith('servo:'):
                            angle = int(cmd.split(':')[1])
                            Servo(S1, angle)
                            time.sleep(1)
                            print(f"[Control] 舵机转到{angle}度")
                            server_text = f"Servo to {angle}"
                        elif cmd.startswith('led:'):
                            # 示例：led:blink
                            if 'blink' in cmd:
                                for _ in range(3):
                                    WIFI_LED.value(1)
                                    time.sleep_ms(200)
                                    WIFI_LED.value(0)
                                    time.sleep_ms(200)
                                server_text = "LED blinked"
                        else:
                            server_text = cmd
                    except Exception as e:
                        print(f"[Control] 指令解析异常: {e}")
                        server_text = msg.decode('ascii')
        except:
            pass

_thread.start_new_thread(recv_control_thread, ())

while True:
    clock.tick()
    img = sensor.snapshot()  # 获取当前帧
    now = time.time()

    # 按键控制：按下时保存并发送图片
    if KEY.value() == 0:  # 按键被按下
        time.sleep_ms(10)  # 消抖
        if KEY.value() == 0:
            WIFI_LED.value(1)  # 按下时点亮LED
            img.save('/sdcard/frame.jpg')  # 保存为JPEG
            with open('/sdcard/frame.jpg', 'rb') as f:
                img_bytes = f.read()
            img_len = len(img_bytes)
            s.send(img_len.to_bytes(4, 'big'))
            s.send(img_bytes)
            print(f'Sent frame by key, size: {img_len} bytes')
            capture_show = 10  # 连续显示10帧
            while not KEY.value():  # 等待按键松开
                pass
            WIFI_LED.value(0)  # 松开时熄灭LED

    # 屏幕显示
    img.draw_string_advanced(0, 0, 30, 'FPS: '+str("%.3f"%(clock.fps())), color = (255, 255, 255))
    if capture_show > 0:
        img.draw_string_advanced(0, 80, 30, "Capture", color=(255,0,0))
        capture_show -= 1
    Display.show_image(img)
