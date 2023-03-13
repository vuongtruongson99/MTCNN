# Приложение для распознавания лиц
## Requirements:
1) Настроить виртуальной среды с помощью Conda
```
cd <MTCNN_dir>
conda create -n face_recognition python=3.9
conda activate face_recognition
```
2) Установить необходимые библиотеки
```
pip install -r requirements.txt
```
---

## Запустить приложение:
```shell
streamlit run deployment/app.py
```
