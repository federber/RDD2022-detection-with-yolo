
def check_cuda():
    """Проверка доступности CUDA"""
    if not torch.cuda.is_available():
        print("CUDA недоступна, работаем на CPU.")
        return "cpu"

    try:
        torch.zeros(1).cuda()
        print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    except Exception as e:
        print(f"Ошибка использования CUDA: {e}. Переключаемся на CPU.")
        return "cpu"
    

def load_project_config(config_file):
    """Загружает конфигурацию проекта из JSON-файла."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Файл {config_file} не найден!")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"Загружена конфигурация проекта из {config_file}")
    return config


def convert_voc_to_yolo(xml_file, output_dir, class_mapping, class_set):
    """Конвертирует XML (VOC) в YOLO и собирает оригинальные классы."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    txt_filename = os.path.splitext(os.path.basename(xml_file))[0] + ".txt"
    txt_filepath = os.path.join(output_dir, txt_filename)

    with open(txt_filepath, "w") as txt_file:
        for obj in root.findall("object"):
            original_class = obj.find("name").text
            class_set.add(original_class)  # Добавляем в множество всех классов

            new_class = None
            if class_mapping and class_mapping != "orig":
                for mapped_class, original_classes in class_mapping.items():
                    if original_class in original_classes:
                        new_class = mapped_class
                        break

            if new_class is None:
                new_class = original_class


            class_id = list(class_mapping.keys()).index(new_class) if class_mapping != "orig" else original_class

            bbox = obj.find("bndbox")
            xmin, ymin, xmax, ymax = map(int, [bbox.find("xmin").text, bbox.find("ymin").text,
                                               bbox.find("xmax").text, bbox.find("ymax").text])
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            


def update_yolo_config(config_path, class_mapping, original_classes, dataset_path):
    """Обновляет yolo_config.yaml: классы, количество классов, пути train/val."""
    if not os.path.exists(config_path):
        print(f"Файл {config_path} не найден! Пропускаем обновление.")
        return
    dataset_path = os.path.join(os.getcwd(), dataset_path)
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["path"] = dataset_path
    config["train"] = train_path
    config["val"] = val_path
    config["names"] = list(original_classes if class_mapping == "orig" else class_mapping.keys())
    config["nc"] = len(config["names"])

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Обновлен {config_path}:")
    print(f"   - Классы ({config['nc']}): {config['names']}")
    print(f"   - Train path: {config['train']}")
    print(f"   - Val path: {config['val']}")



def convert_labels(labels_dir, class_list):
    """Заменяет имена классов на их индексы в YOLO label-файлах."""
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_dir, filename)
            
            # Читаем содержимое label-файла
            with open(file_path, "r") as file:
                lines = file.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                class_name = parts[0]  # Первое значение - название класса
                
                # Заменяем имя класса на его индекс
                if class_name in class_list:
                    class_id = class_list.index(class_name)
                    new_line = f"{class_id} " + " ".join(parts[1:])  # Обновляем строку
                    new_lines.append(new_line)
                else:
                    print(f"Класс {class_name} не найден в списке!")

            # Перезаписываем файл с обновленными классами
            with open(file_path, "w") as file:
                file.write("\n".join(new_lines) + "\n")

    print("Все labels успешно обновлены!")


def update_yaml_config(config_path):
    """Обновляет yolo_config.yaml, заменяя имена классов на их индексы."""
    if not os.path.exists(config_path):
        print(f"Файл {config_path} не найден!")
        return

    # Загружаем YAML-файл
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    if "names" in config and isinstance(config["names"], list):
        # Обновляем список классов, заменяя их на индексы
        num_classes = len(config["names"])
        config["names"] = list(range(num_classes))

        # Перезаписываем YAML
        with open(config_path, "w") as file:
            yaml.dump(config, file, default_flow_style=False)

        print("Файл yolo_config.yaml успешно обновлен!")
    else:
        print("Ошибка: 'names' не найден в yolo_config.yaml!")

