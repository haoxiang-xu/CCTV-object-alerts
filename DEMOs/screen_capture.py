import os
import cv2
import torch
import platform
import numpy as np
import tkinter as tk
from PIL import ImageGrab, Image, ImageDraw, ImageFont
from PIL.Image import Resampling
from ultralytics import YOLO
from ultralytics.engine.results import Results

def load_fonts():
    font_path_mac = "../FONTs/STHeiti Light.ttc"
    font_path_win = "./FONTs/Quicksand-VariableFont_wght.ttf"
    
    if platform.system() == "Darwin":  # macOS
        try:
            font = ImageFont.truetype(font_path_mac, 16)
        except IOError as e:
            print(f"Failed to load macOS font: {e}")
    elif platform.system() == "Windows":  # Windows
        try:
            font = ImageFont.truetype(os.path.abspath(font_path_win), 16)
        except IOError as e:
            print(f"Failed to load Windows font: {e}")
            print(os.path.abspath(font_path_win))
    else:
        print("Unsupported OS. Loading default font.")
        font = ImageFont.load_default()
    return font
        
def screen_capture(frames_per_second = 16):
    # Set up the display window name
    cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)
    
    while True:
        # Capture the screen
        screen = ImageGrab.grab()
        # Convert the image to an array
        screen_np = np.array(screen)
        # Convert the color space from BGR (OpenCV default) to RGB
        screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)

        # Display the captured screen
        cv2.imshow('Screen Capture', screen_np)

        # Wait for 100 milliseconds (1 second)
        if cv2.waitKey(1000//frames_per_second) & 0xFF == ord('q'):
            break

        # Check if the window is closed
        if cv2.getWindowProperty('Screen Capture', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()
def screen_capture_and_detect(frames_per_second = 16, resize_dimension = (640, 480), object_confidence_threshold = 0.08, remove_large_objects = True, person_detection_only = False):
    # Initialize the YOLO model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("../MODELs/yolov8n").to(device)

    # Set up the display window name
    cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)

    font = load_fonts()
    bg_color_ranges = {"#FF0000_#181818": [0,1], "#FF9900_#181818": [1,14], "#341A36_#FFFFFF": [14,24], "#00C036_#181818": [24,80]}
    color_labels = []
    skip_classes = [62, 63, 72]
    for i in range(0,80):
        for color in bg_color_ranges:
            if i >= bg_color_ranges[color][0] and i < bg_color_ranges[color][1]:
                color_labels.append(color)
                break
    
    recording = False
    video_writer = None
    video_file_path = "output.avi"
    
    while True:
        # Capture the screen
        screen = ImageGrab.grab()
        screen_resized = screen.resize(resize_dimension, Resampling.LANCZOS)
        # Convert the image to an array
        if device == 'cuda':
            try:
                img_tensor = torch.from_numpy(np.array(screen_resized)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
                results = model(img_tensor, verbose=False)
            except:
                continue
        else:
            try:
                results = model(screen_resized, verbose=False)
            except:
                continue
        result = results[0] # Get the first result
        boxes = result.boxes.xyxy # Get the bounding boxes
        cls = result.boxes.cls.tolist() # Get the class IDs
        conf = result.boxes.conf.tolist() # Get the confidence values
        
        names = result.names

        draw = ImageDraw.Draw(screen_resized)
        for index in range(len(boxes)):
            if round(conf[index], 2) < object_confidence_threshold:
                continue
            if cls[index] in skip_classes:
                continue
            if person_detection_only and cls[index] != 0:
                continue
            
            box_data = boxes[index].tolist()  # Convert tensor to list
            filling_color = color_labels[int(cls[index])].split("_")

            # Check if the length of the box_data matches expected number of elements for just coordinates
            if len(box_data) == 4:
                x1, y1, x2, y2 = box_data  # Unpack the coordinates
                if remove_large_objects and abs(x2-x1) * abs(y2-y1) > 1/3 * (resize_dimension[0] * resize_dimension[1]) or abs(x2-x1) > 4/6 * resize_dimension[0]:
                    continue

                cls_label = names[cls[index]]  # Get the class name using class ID or default to "Unknown"
                conf_label = int(round(conf[index], 2)*100)  # Get the confidence and convert to percentage
                label = f"{cls_label} {conf_label}%"  # Create label with class name and confidence

                draw.rectangle([x1, y1, x2, y2], outline=filling_color[0], width=3)  # Draw the rectangle
                text_bg = [x1, max(y1 - 16,0), x1 + (len(cls_label)+5) * 9, max(16,y1)] # Create background rectangle for text
                draw.rectangle(text_bg, fill=filling_color[0])
                draw.text((x1+2, max(y1 - 16,0)), label, fill=filling_color[1], font=font)  # Draw the label    
        
        # Draw Recording Indicator
        if recording:
            draw.rectangle([0, 0, resize_dimension[0], 4], fill="red", width=3)
        
        screen_np = np.array(screen_resized)
        # Convert the color space from BGR (OpenCV default) to RGB
        screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)
        
        if recording:
            if video_writer is None:  # Start new video writer
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(video_file_path, fourcc, frames_per_second, resize_dimension)
            video_writer.write(screen_np)  # Write frame to video file

        # Display the captured screen
        cv2.imshow('Screen Capture', screen_np)

        # Wait for 1000 milliseconds (1 second)
        key = cv2.waitKey(1000//frames_per_second) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if recording:
                # Stop recording and release the video writer
                recording = False
                video_writer.release()
                video_writer = None
                print("Recording stopped and saved.")
            else:
                # Start recording
                recording = True
                print("Recording started.")

        # Check if the window is closed
        if cv2.getWindowProperty('Screen Capture', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    default_frames_per_second = 16
    default_resize_dimension = (768, 432)
    default_object_confidence_threshold = 0.32
    
    screen_capture_and_detect(16, (1920, 1080), person_detection_only=False)
    
