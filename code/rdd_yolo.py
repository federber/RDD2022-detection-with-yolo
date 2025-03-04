import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import argparse
import json
import torch
import yaml
from ultralytics import YOLO

from utils import *


#project_config_path = "/home/jupyter/datasphere/project/project_config.json"
#source_dataset_path = "/home/jupyter/datasphere/project/data/RDD22_Japan"
#yolo_config_path = "/home/jupyter/datasphere/project/yolo_config.yaml"

source_dataset_path = "/app/dataset"
project_config_path = "/app/project_config.json"
yolo_config_path = "/app/yolo_config.yaml"

config = load_project_config(project_config_path)

# Папка, куда будет скопирован датасет
dataset_root = "datasets"
dataset_path = os.path.join(dataset_root, "processed_dataset")

# Копируем датасет в рабочую папку
if not os.path.exists(dataset_path):
    shutil.copytree(source_dataset_path, dataset_path)
    print(f"Датасет скопирован в {dataset_path}")
else:
    print(f"Датасет уже скопирован, используем существующую копию.")



data_dir = os.path.join(dataset_path, "train")  # Начальная папка с изображениями и аннотациями
val_dir = os.path.join(dataset_path, "val")  # Папка для валидации

# Проверяем и переименовываем папку annotations -> labels в train
annotations_path = os.path.join(data_dir, "annotations")
xmls_path = os.path.join(annotations_path, "xmls")
labels_path = os.path.join(data_dir, "labels")


if os.path.exists(annotations_path) and os.path.exists(xmls_path):
    os.rename(annotations_path, labels_path)  # Переименовываем annotations -> labels
    xmls_path = os.path.join(labels_path, "xmls")
    print("Папка 'annotations' переименована в 'labels'.")

    # Перемещаем все XML-файлы из labels/xmls в labels
    for xml_file in os.listdir(xmls_path):
        xml_file_path = os.path.join(xmls_path, xml_file)
        if os.path.isfile(xml_file_path) and xml_file.endswith(".xml"):
            shutil.move(xml_file_path, labels_path)

    # Удаляем пустую папку xmls
    os.rmdir(xmls_path)
    print("Файлы из 'labels/xmls' перемещены, папка 'xmls' удалена.")


original_classes_set = set()
for xml_file in os.listdir(labels_path):
    if xml_file.endswith(".xml"):
        convert_voc_to_yolo(os.path.join(labels_path, xml_file), labels_path, config["class_mapping"], original_classes_set)
        os.remove(os.path.join(labels_path, xml_file))
detected_orig_classes = sorted(list(original_classes_set))
original_classes = detected_orig_classes if config["original_classes"] == "auto" else config["original_classes"]
print(f"Определенные оригинальные классы: {sorted(list(original_classes_set))}")
print(f"Используемые оригинальные классы: {original_classes}")



# Проверяем, существует ли уже папка val
if os.path.exists(val_dir):
    print("Папка 'val' уже существует. Разделение данных не требуется.")
else:
    # Создаем папки для валидации
    os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)

    # Получаем список всех изображений (предполагаем формат .jpg)
    image_files = [f for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith(".jpg")]

    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

    def move_files(file_list, src_folder, dest_folder):
        for file in file_list:
            shutil.move(os.path.join(src_folder, "images", file), os.path.join(dest_folder, "images", file))

            label_file = file.replace(".jpg", ".txt")
            if os.path.exists(os.path.join(src_folder, "labels", label_file)):
                shutil.move(os.path.join(src_folder, "labels", label_file), os.path.join(dest_folder, "labels", label_file))

    move_files(val_files, data_dir, val_dir)
    print("Разделение на train и val завершено!")


device = check_cuda() if config["device"] == "auto" else config["device"]
if device == "cuda" and not torch.cuda.is_available():
    print("CUDA недоступна! Переключаемся на CPU.")
    device = "cpu"

print(f"Используется устройство: {device}")

# Обновляем yolo_config.yaml
update_yolo_config(yolo_config_path, config["class_mapping"], original_classes, dataset_path)


#Меняем формат имен классов на тот, который поддерживает YOLO
tr_labels_path = os.path.join(dataset_path, "train/labels")
val_labels_path = os.path.join(dataset_path, "val/labels")

with open(yolo_config_path, "r") as file:
    yaml_data = yaml.safe_load(file)
class_list = yaml_data.get("names", [])


convert_labels(tr_labels_path, class_list)
convert_labels(val_labels_path, class_list)
update_yaml_config(yolo_config_path)



import torch
from ultralytics import YOLO

print(f"Using device: {device}")

# Загружаем модель
model = YOLO("yolo11m.yaml").to(device)

# Запуск обучения
model.train(data=yolo_config_path, epochs=120, imgsz=600, iou = 0.7, device=device)