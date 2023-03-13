import os
from webbrowser import get
import av
import sys
sys.path.append(".")
import cv2
import time
import json
import torch
import warnings
import datetime
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from face_recognition.facenet import *
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

warnings.filterwarnings('ignore')

def detect_faces(img_path):
    if type(img_path) == str:
        image = Image.open(img_path)
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, 1)
    else:
        image = img_path
        if type(image) != np.ndarray:
            image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, 1)

    face = mtcnn(image)
    return face

def add_embedding(face_cropped, user_code):
    new_name = str(user_code)
    new_emb = resnet(face_cropped.to(device).unsqueeze(0)).detach()
    data = [embedding_list.append(new_emb), name_list.append(new_name)]
    torch.save(data, 'deployment/assets/embedding.pt')

def main(img, get_ax=False):
    if get_ax:
        name, faces = recognize(img, get_axes=get_ax)
        if name == 'Unknown' and faces is None:
            return {"name": name}, faces
        return json.load(open(f"deployment/assets/info/{name}.json", 'r')), faces[0]
    
    else:
        name = recognize(img)
        if name == 'Unknown':
            return {"name": name}
        return json.load(open(f"deployment/assets/info/{name}.json", 'r'))

def app():
    """Приложение для распознавания лиц"""

    st.title("Приложение для распознавания лиц")
    st.text("Создавать с помощью Streamlit и глубокого обучения")

    activities = ["Обо мне", "Загрузить данные", "Распознавания", "Распознавание веб-камеры в реальном времени"]
    choice = st.sidebar.selectbox("Выберите действие", activities)

    if choice == 'Обо мне':
        st.subheader("Приложение для аутентификации по лицу и голосу")
        st.markdown(
            "Создан с помощью Streamlit [Truong Son](https://github.com/vuongtruongson99). Веб-приложение позволяет добавить пользователя в базу данных и проверить его позже.")
        st.subheader("Студент:")
        members = ''' 
            **Выонг Чыонг Шон** - ИКБО-05-19\n'''
        st.markdown(members)
    
    elif choice == 'Загрузить данные':
        st.subheader("Добавьте свое лицо в базу данных")
        user_name = st.text_input("Ваше имя:")
        user_dob = st.date_input("Дата рождения:", min_value=datetime.date(1940, 1, 1))
        user_code = st.text_input("Личный код:")
        image = st.camera_input("Сделать фото")

        if image is not None:
            image = Image.open(image).convert('RGB')
            face = detect_faces(image)

            # Save new image to database
            user_folder = os.path.join('deployment/assets/database', user_code)
            if not os.path.isdir(user_folder):
                os.mkdir(user_folder)
            new_upload_path = os.path.join(user_folder, time.strftime("%Y%m%d%H%M%S.jpg"))
            image.save(new_upload_path)

            # Count number of user in database
            user_counts = len(next(os.walk('deployment/assets/database'))[1])
            print(f"На данный момент есть {user_counts} пользователей!!!")

            # Save user's information to json file
            user_info_path = os.path.join(user_folder.replace(f"database\{user_code}", "info"), f"{user_code}.json")
            user_info = {
                "name": user_name,
                "user_code": user_code,
                "user_dob": user_dob.strftime('%m/%d/%Y'),
                "img_num": len([name for name in os.listdir(f'deployment/assets/database/{user_code}') if name.endswith('jpg') or name.endswith('JPG')])
            }
            with open(user_info_path, 'w+') as f:
                json.dump(user_info, f, indent=2)
            
            # Add face cropped to embedding file
            if face is not None:
                add_embedding(face, user_code)
                st.success(f'Загружено: встраивание успешно сохранено!')

    elif choice == 'Распознавания':
        image = st.camera_input("Сделать фото")
        if image is not None:
            image = Image.open(image)

            enhace_type = st.sidebar.radio(
                "Аугментация", ["Оригинал", "Оттенки серого", "Контраст", "Яркость", "Размытие"]
            )

            if enhace_type == "Оттенки серого":
                new_img = np.array(image.convert('RGB'))
                img = cv2.cvtColor(new_img, 1)
                result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                st.image(result)

            elif enhace_type == "Контраст":
                c_rate = st.sidebar.slider("Контраст", 0.5, 3.5)
                enhacer = ImageEnhance.Contrast(image)
                result = enhacer.enhance(c_rate)
                st.image(result)

            elif enhace_type == "Яркость":
                c_rate = st.sidebar.slider("Яркость", 0.5, 3.5)
                enhacer = ImageEnhance.Brightness(image)
                result = enhacer.enhance(c_rate)
                st.image(result)

            elif enhace_type == 'Размытие':
                new_img = np.array(image.convert('RGB'))
                blur_rate = st.sidebar.slider('Размытие', 0.5, 3.5)
                img = cv2.cvtColor(new_img, 1)
                result = cv2.GaussianBlur(img, (11, 11), blur_rate)
                st.image(result)
            
            else:
                result = image

        if st.button("Процесс"):
            with st.spinner(text="🐱‍🏍 Распознавание..."):
                data = main(result)
                st.write(data)
                st.balloons()
                # os.remove("recognition.py")

    elif choice == "Распознавание веб-камеры в реальном времени":
        st.warning("Внимание: Чтобы использовать этот режим, вам необходимо предоставить доступ к веб-камере.")
        message = "🐱‍👓 Подождите секунду, кое-что сделать..."

        with st.spinner(message):
            class VideoProcessor:
                def recv(self, frame):
                    frame = frame.to_ndarray(format="bgr24")
                    frame = cv2.cvtColor(frame, 1)
                    
                    info, faces = main(frame, get_ax=True)
                    
                    if faces is None:
                        return av.VideoFrame.from_ndarray(frame, format="bgr24")
                
                    cv2.rectangle(frame, (int(faces[0]), int(faces[1])), (int(faces[2]), int(faces[3])), (0, 255, 0), 3)
                    frame = cv2.putText(frame, info['name'], (int(faces[0]), int(faces[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                    return frame
            
            webrtc_streamer(
                key="SOWN",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False}
            )

if __name__ == '__main__':
    app()