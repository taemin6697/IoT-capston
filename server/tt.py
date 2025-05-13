import gradio as gr
# import time # 필요하다면 함수 내에서 사용

def show_webcam_feed(image_from_webcam):
    # 이 함수는 live=True 와 streaming=True 설정으로 인해
    # 웹캠에서 새로운 프레임이 들어올 때마다 호출되려고 시도합니다.
    # print(f"Received frame with shape: {image_from_webcam.shape if image_from_webcam is not None else 'None'}")
    return image_from_webcam

webcam_input_component = gr.Image(
    label="웹캠 입력",
    sources=["webcam"], # 이전 논의대로 이 부분은 사용하시는 Gradio 버전에 따라 확인 필요
    streaming=True,
    type="numpy",
    height=64,
    width=64
)

iface = gr.Interface(
    fn=show_webcam_feed,
    inputs=webcam_input_component,
    outputs=gr.Image(label="Webcam Output"),
    title="Webcam Stream Test",
    live=True  # live=True 만 유지
    # every=0.1  <-- 이 줄을 제거하거나 주석 처리
)

iface.launch(server_name="0.0.0.0", share=True)