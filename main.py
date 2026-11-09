"""
Red Eye (redeye) - Детекция человека в красной одежде с использованием YOLO и OpenCV.
"""

import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Детекция людей в красной одежде при помощи камеры и YOLO"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Индекс веб-камеры (по умолчанию: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Модель YOLO (например: yolov8n.pt, yolo11n.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Минимальная уверенность YOLO (по умолчанию: 0.45)",
    )
    parser.add_argument(
        "--red-threshold",
        type=float,
        default=0.12,
        help="Порог доли красного цвета в области одежды (по умолчанию: 0.12 или 12%%)",
    )
    parser.add_argument(
        "--show-only-red",
        action="store_true",
        help="Отображать рамки только для людей в красном",
    )
    parser.add_argument(
        "--debug-torso",
        action="store_true",
        help="Отображать прямоугольник анализируемой области одежды (торса)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Путь для сохранения видео (опционально, например output.mp4)",
    )
    return parser.parse_args()


def calculate_red_ratio(torso_bgr):
    """
    Вычисляет долю красных пикселей в переданной области изображения.
    Красный цвет в HSV разбит на два диапазона (на краях шкалы Hue: 0-10 и 160-180).
    """
    if torso_bgr is None or torso_bgr.size == 0:
        return 0.0

    # Перевод в цветовое пространство HSV
    hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)

    # Диапазон 1 (нижний край спектра)
    lower_red1 = np.array([0, 80, 50])
    upper_red1 = np.array([10, 255, 255])

    # Диапазон 2 (верхний край спектра)
    lower_red2 = np.array([160, 80, 50])
    upper_red2 = np.array([180, 255, 255])

    # Создание бинарных масок
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Фильтрация шумов (морфологическое закрытие)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Подсчет доли красных пикселей
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = torso_bgr.shape[0] * torso_bgr.shape[1]

    if total_pixels == 0:
        return 0.0

    return red_pixels / float(total_pixels)


def draw_detection(
    frame,
    box,
    is_red,
    red_ratio,
    confidence,
    show_torso_box=False,
    torso_coords=None,
):
    """
    Отрисовывает рамку человека и информационные метки.
    """
    x1, y1, x2, y2 = box

    if is_red:
        # Ярко-красный для человека в красной одежде
        color = (0, 0, 255)
        text_label = f"RED CLOTHES: {red_ratio * 100:.1f}% ({confidence:.2f})"
        border_thickness = 3
    else:
        # Зеленый / серый для обычного человека
        color = (180, 180, 180)
        text_label = f"Person: {red_ratio * 100:.1f}% red ({confidence:.2f})"
        border_thickness = 2

    # Рамка вокруг человека
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, border_thickness)

    # Подложка под текст для лучшей читаемости
    (text_w, text_h), baseline = cv2.getTextSize(
        text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    label_y = max(y1 - 10, text_h + 10)
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - 4),
        (x1 + text_w + 6, label_y + baseline),
        color,
        -1,
    )

    # Текст надписи
    text_color = (255, 255, 255) if is_red else (20, 20, 20)
    cv2.putText(
        frame,
        text_label,
        (x1 + 3, label_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color,
        2,
        cv2.LINE_AA,
    )

    # Опционально: подсветка области торса
    if show_torso_box and torso_coords is not None:
        tx1, ty1, tx2, ty2 = torso_coords
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 255, 0), 1)


def main():
    args = parse_arguments()

    print("=" * 60)
    print("Запуск системы Red Eye: Детекция человека в красной одежде")
    print(f"Загрузка модели YOLO: {args.model}...")
    print("=" * 60)

    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Ошибка загрузки модели {args.model}: {e}")
        return

    # Открытие видеопотока с камеры
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть камеру с индексом {args.camera}.")
        print("Попробуйте указать другой индекс: python main.py --camera 1")
        return

    # Задаем желаемое разрешение камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Инициализация записи видео (если запрошено)
    video_writer = None
    if args.save:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.save, fourcc, 20.0, (frame_width, frame_height)
        )
        print(f"Запись видео включена: {args.save}")

    show_only_red = args.show_only_red
    show_torso = args.debug_torso
    red_threshold = args.red_threshold

    print("\nУправление:")
    print("  [Q] или [ESC] - Выход")
    print("  [R] - Переключить фильтр (только люди в красном / все)")
    print("  [T] - Показать/скрыть область анализа торса")
    print("  [+] / [-] - Увеличить/уменьшить порог красного")
    print("\nКамера запущена. Ожидание кадров...\n")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Предупреждение: Не удалось получить кадр с камеры.")
            break

        # Расчет FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
        prev_time = curr_time

        frame_h, frame_w = frame.shape[:2]

        # Запуск инференса YOLO
        # classes=[0] означает детекцию только класса "person"
        results = model.predict(
            source=frame,
            classes=[0],
            conf=args.conf,
            verbose=False,
        )

        red_persons_count = 0
        total_persons_count = 0

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box_data in boxes:
                total_persons_count += 1
                coords = box_data.xyxy[0].cpu().numpy().astype(int)
                conf = float(box_data.conf[0].cpu().numpy())

                x1, y1, x2, y2 = coords
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_w, x2), min(frame_h, y2)

                box_h = y2 - y1
                box_w = x2 - x1

                if box_h <= 10 or box_w <= 10:
                    continue

                # Выделяем область торса (одежда):
                # берем от 15% до 70% высоты человека и от 12% до 88% ширины
                torso_y1 = max(0, int(y1 + 0.15 * box_h))
                torso_y2 = min(frame_h, int(y1 + 0.70 * box_h))
                torso_x1 = max(0, int(x1 + 0.12 * box_w))
                torso_x2 = min(frame_w, int(x2 - 0.12 * box_w))

                torso_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
                red_ratio = calculate_red_ratio(torso_crop)
                is_red = red_ratio >= red_threshold

                if is_red:
                    red_persons_count += 1

                # Отрисовка
                if is_red or not show_only_red:
                    draw_detection(
                        frame,
                        (x1, y1, x2, y2),
                        is_red=is_red,
                        red_ratio=red_ratio,
                        confidence=conf,
                        show_torso_box=show_torso,
                        torso_coords=(torso_x1, torso_y1, torso_x2, torso_y2),
                    )

        # Вывод информационной панели на экран
        hud_bg_color = (0, 0, 180) if red_persons_count > 0 else (40, 40, 40)
        cv2.rectangle(frame, (10, 10), (380, 105), hud_bg_color, -1)
        cv2.rectangle(frame, (10, 10), (380, 105), (255, 255, 255), 1)

        cv2.putText(
            frame,
            f"RED EYE DETECTOR | FPS: {fps:.1f}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Red Persons: {red_persons_count} / {total_persons_count}",
            (20, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255) if red_persons_count > 0 else (200, 200, 200),
            2,
        )
        filter_text = "Only Red" if show_only_red else "All Persons"
        cv2.putText(
            frame,
            f"Threshold: {red_threshold * 100:.0f}% | Mode: {filter_text}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            frame,
            "[Q] Exit  [R] Mode  [T] Torso  [+/-] Sens",
            (20, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
        )

        # Если сохраняем видео
        if video_writer is not None:
            video_writer.write(frame)

        # Отображение окна
        cv2.imshow("Red Eye - Red Clothes Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
            break
        elif key in (ord("r"), ord("R")):
            show_only_red = not show_only_red
            print(f"Режим изменен: show_only_red={show_only_red}")
        elif key in (ord("t"), ord("T")):
            show_torso = not show_torso
            print(f"Отображение торса: show_torso={show_torso}")
        elif key in (ord("+"), ord("=")):
            red_threshold = min(1.0, red_threshold + 0.02)
            print(f"Порог красного цвета: {red_threshold * 100:.0f}%")
        elif key in (ord("-"), ord("_")):
            red_threshold = max(0.02, red_threshold - 0.02)
            print(f"Порог красного цвета: {red_threshold * 100:.0f}%")

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Завершение работы программы.")


if __name__ == "__main__":
    main()
