import os
import cv2
import torch
import requests
import platform
import numpy as np
import tkinter as tk
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import ImageGrab, Image, ImageDraw, ImageFont
from PIL.Image import Resampling
from ultralytics import YOLO
from ultralytics.engine.results import Results
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

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
def screen_capture_and_detect_yolo_v8(model_variants = "NANO",
                                      frames_per_second = 16, 
                                      resize_dimension = (640, 480), 
                                      object_confidence_threshold = 0.08, 
                                      remove_large_objects = True, 
                                      person_detection_only = False, 
                                      verbose = False):
    #FUNCTIONS -------------------------------------------------------------------------------------------------------------------FUNCTIONS
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
    def adjust_size_to_model(input_size, stride=32):
        # Adjust size so it's divisible by the model's stride
        new_width = (input_size[0] // stride) * stride
        new_height = (input_size[1] // stride) * stride
        return (new_width, new_height)
    def filter_rectangles(boxes, classes, confidences):
        # Convert to a PyTorch tensor if not already one
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes)
        if not isinstance(classes, torch.Tensor):
            classes = torch.tensor(classes)
        if not isinstance(confidences, torch.Tensor):
            confidences = torch.tensor(confidences)

        # Initialize a list to mark rectangles for removal
        to_remove = [False] * len(boxes)
        
        # Loop over all pairs of boxes to check for containment and same class
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if classes[i] == classes[j]:  # Check if they belong to the same class
                    # Check if box i is inside box j
                    if (boxes[i, 0] >= boxes[j, 0] and boxes[i, 1] >= boxes[j, 1] and
                        boxes[i, 2] <= boxes[j, 2] and boxes[i, 3] <= boxes[j, 3]):
                        # Choose to remove the one with lower confidence
                        if confidences[i] > confidences[j]:
                            to_remove[j] = True
                        else:
                            to_remove[i] = True
                    # Check if box j is inside box i
                    elif (boxes[j, 0] >= boxes[i, 0] and boxes[j, 1] >= boxes[i, 1] and
                        boxes[j, 2] <= boxes[i, 2] and boxes[j, 3] <= boxes[i, 3]):
                        # Choose to remove the one with lower confidence
                        if confidences[j] > confidences[i]:
                            to_remove[i] = True
                        else:
                            to_remove[j] = True

        # Filter out the boxes and classes marked for removal
        filtered_boxes = boxes[torch.tensor(to_remove) == False]
        filtered_classes = classes[torch.tensor(to_remove) == False]
        filtered_confidences = confidences[torch.tensor(to_remove) == False]

        return filtered_boxes, filtered_classes.tolist(), filtered_confidences.tolist(), len(boxes) - len(filtered_boxes)
    #FUNCTIONS ----------------------------------------------------------------------------------------------------------------------------   
     
    #CONSTs -------------------------------------------------------------------------------------------------------------------------CONSTs
    SKIPPING_CLASSES = [4, 6, 62, 63, 72]
    ADJUSTED_DIMENSION = adjust_size_to_model(resize_dimension)
    LABEL_BACKGROUND_COLORS = {"#FF0000_#181818": [0,1], "#FF9900_#181818": [1,14], "#341A36_#FFFFFF": [14,24], "#00C036_#181818": [24,80]}
    FONT = load_fonts()
    LABEL_BACKGROUND_COLOR = []
    for i in range(0,80):
        for color in LABEL_BACKGROUND_COLORS:
            if i >= LABEL_BACKGROUND_COLORS[color][0] and i < LABEL_BACKGROUND_COLORS[color][1]:
                LABEL_BACKGROUND_COLOR.append(color)
                break
    #CONSTs -------------------------------------------------------------------------------------------------------------------------------
    
    #OUTER STATIC VARIABLES -----------------------------------------------------------------------------------------OUTER STATIC VARIABLES
    TOTAL_OBJECTS_REMOVED = 0
    TOTAL_OBJECTS_REMOVED_BY_OVERLAPPING = 0
    TOTAL_OBJECTS_REMOVED_BY_CONFIDENCE = 0
    TOTAL_LARGE_OBJECTS_REMOVED = 0
    #OUTER STATIC VARIABLES ---------------------------------------------------------------------------------------------------------------
    
    #RECORDING VARIABLES ===============================================================================================RECORDING VARIABLES
    is_recording = False
    video_writer = None
    video_file_path = "output.avi"
    #RECORDING VARIABLES ==================================================================================================================

    #LOAD YOLO MODEL -------------------------------------------------------------------------------------------------------LOAD YOLO MODEL
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[SYSTEM IS NOW RUNNING ON {device}]")
    if model_variants == "NANO":
        model = YOLO("../MODELs/yolov8n").to(device)
    elif model_variants == "SMALL":
        model = YOLO("../MODELs/yolov8s").to(device)
    elif model_variants == "MEDIUM":
        model = YOLO("../MODELs/yolov8m").to(device)
    elif model_variants == "LARGE":
        model = YOLO("../MODELs/yolov8l").to(device)
    elif model_variants == "EXTRA-LARGE":
        model = YOLO("../MODELs/yolov8x").to(device)
    else:
        model = YOLO("../MODELs/yolov8n").to(device)
    #LOAD YOLO MODEL ----------------------------------------------------------------------------------------------------------------------

    cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)
    
    while True:
        #INTER STATIC VARIABLES ------------------------------------------------------------------------------------------------INTER STATIC VARIABLES
        OBJECT_REMOVED_FOR_THIS_FRAME = 0
        #INTER STATIC VARIABLES ----------------------------------------------------------------------------------------------------------------------
        
        screen = ImageGrab.grab()
        screen_resized = screen.resize(ADJUSTED_DIMENSION, Resampling.LANCZOS)
        img_tensor = torch.from_numpy(np.array(screen_resized)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        results = model(img_tensor, verbose=False)
        
        # Extract Results -------------------------------------------------------------------------------------------------------------------
        result = results[0]
        boxes = result.boxes.xyxy
        cls = result.boxes.cls.tolist()
        conf = result.boxes.conf.tolist()
        names = result.names
        boxes, cls, conf, obejct_removed_by_overlapping = filter_rectangles(boxes, cls, conf)
        # Extract Results -------------------------------------------------------------------------------------------------------------------

        draw = ImageDraw.Draw(screen_resized)
        
        for index in range(len(boxes)):
            # INTER LOOP FILTERING CONDITIONS ===============================================================================================
            if round(conf[index], 2) < object_confidence_threshold:
                OBJECT_REMOVED_FOR_THIS_FRAME += 1
                continue
            if cls[index] in SKIPPING_CLASSES:
                OBJECT_REMOVED_FOR_THIS_FRAME += 1
                TOTAL_OBJECTS_REMOVED_BY_CONFIDENCE += 1
                continue
            if person_detection_only and cls[index] != 0:
                OBJECT_REMOVED_FOR_THIS_FRAME += 1
                continue
            # INTER LOOP FILTERING CONDITIONS ===============================================================================================
            
            box_data = boxes[index].tolist()
            filling_color = LABEL_BACKGROUND_COLOR[int(cls[index])].split("_")

            if len(box_data) == 4:
                x1, y1, x2, y2 = box_data  # Unpack the coordinates
                if remove_large_objects and abs(x2-x1) * abs(y2-y1) > 1/3 * (resize_dimension[0] * resize_dimension[1]) or abs(x2-x1) > 4/6 * resize_dimension[0]:
                    OBJECT_REMOVED_FOR_THIS_FRAME += 1
                    TOTAL_LARGE_OBJECTS_REMOVED += 1
                    continue

                cls_label = names[cls[index]]  # Get the class name using class ID or default to "Unknown"
                conf_label = int(round(conf[index], 2)*100)  # Get the confidence and convert to percentage
                label = f"{cls_label} {conf_label}%"  # Create label with class name and confidence

                draw.rectangle([x1, y1, x2, y2], outline=filling_color[0], width=3)  # Draw the rectangle
                text_bg = [x1, max(y1 - 16,0), x1 + (len(cls_label)+5) * 9, max(16,y1)] # Create background rectangle for text
                draw.rectangle(text_bg, fill=filling_color[0])
                draw.text((x1+2, max(y1 - 16,0)), label, fill=filling_color[1], font=FONT)  # Draw the label    
        
        if is_recording:
            draw.rectangle([0, 0, resize_dimension[0], 4], fill="red", width=3)
        screen_np = np.array(screen_resized)
        screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)
        if is_recording:
            if video_writer is None:  # Start new video writer
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(video_file_path, fourcc, frames_per_second, resize_dimension)
            video_writer.write(screen_np)  # Write frame to video file

        cv2.imshow('Screen Capture', screen_np)
        
        if verbose:
            # UPDATE STATISTICS ============================================================================================================
            OBJECT_REMOVED_FOR_THIS_FRAME += obejct_removed_by_overlapping
            TOTAL_OBJECTS_REMOVED_BY_OVERLAPPING += obejct_removed_by_overlapping
            # UPDATE STATISTICS ============================================================================================================
            
            if OBJECT_REMOVED_FOR_THIS_FRAME > 0:
                TOTAL_OBJECTS_REMOVED += OBJECT_REMOVED_FOR_THIS_FRAME
                print(f"TOTAL OBJECT REMOVED: {TOTAL_OBJECTS_REMOVED}" + 
                      f"\n\tOBJECT REMOVED FOR THIS FRAME: {OBJECT_REMOVED_FOR_THIS_FRAME}" + 
                      f"\n\tOBJECT REMOVED BY OVERLAPPING: {TOTAL_OBJECTS_REMOVED_BY_OVERLAPPING}" + 
                      f"\n\tOBJECT REMOVED BY CONFIDENCE: {TOTAL_OBJECTS_REMOVED_BY_CONFIDENCE}\n" +
                      f"\tLARGE OBJECTS REMOVED: {TOTAL_LARGE_OBJECTS_REMOVED}\n")

        key = cv2.waitKey(1000//frames_per_second) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if is_recording:
                is_recording = False
                video_writer.release()
                video_writer = None
                print("Recording stopped and saved.")
            else:
                is_recording = True
                print("Recording started.")

        if cv2.getWindowProperty('Screen Capture', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
def screen_capture_and_detect_segformer_b5(frames_per_second = 16, resize_dimension = (640, 480)):
    def load_model_and_extractor(model_dir, model_id):
        # Check if the directory exists and is not empty
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SYSTEM IS NOW RUNNING ON {device}]")
        required_files = ['config.json', 'pytorch_model.bin']
        files_present = os.listdir(model_dir) if os.path.exists(model_dir) else []

        # Check if all required files are in the directory
        if all(file in files_present for file in required_files):
            print("Loading model and feature extractor from local directory.")
            model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device)
            feature_extractor = SegformerImageProcessor.from_pretrained(model_dir)
        else:
            print("Local directory is empty or missing files. Downloading model and feature extractor.")
            model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
            feature_extractor = SegformerImageProcessor.from_pretrained(model_id)

        return model, feature_extractor
    def decode_segmentation_mask(mask, labels):
        # Define a color for each label (in RGB)
        label_colors = np.array([
            [0, 0, 0],       # Background - Black
            [255, 192, 203], # Skin - Pink
            [0, 0, 255],     # Hair - Blue
            [255, 0, 0],     # Upper-body clothing - Red
            [0, 255, 0],     # Lower-body clothing - Green
            [0, 255, 255],   # Shoes - Cyan
            # Add additional colors for any other categories
            [255, 255, 0],   # Additional Category 1 - Yellow
            [255, 165, 0],   # Additional Category 2 - Orange
            [128, 0, 128],   # Additional Category 3 - Purple
            [255, 20, 147],  # Additional Category 4 - Deep Pink
            [75, 0, 130],    # Additional Category 5 - Indigo
            [0, 128, 128],    # Additional Category 6 - Teal
            [255, 255, 0],   # Additional Category 1 - Yellow
            [255, 165, 0],   # Additional Category 2 - Orange
            [128, 0, 128],   # Additional Category 3 - Purple
            [255, 20, 147],  # Additional Category 4 - Deep Pink
            [75, 0, 130],    # Additional Category 5 - Indigo
            [0, 128, 128]    # Additional Category 6 - Teal
        ])
        # Ensure label_colors covers all the labels present in the mask
        color_mask = label_colors[mask]
        return Image.fromarray(color_mask.astype(np.uint8))
    def apply_mask_on_image(original_image, mask, alpha=0.5):
        # Convert PIL image to array
        image_array = np.array(original_image)
        # Resize mask to match image size
        mask_resized = mask.resize(original_image.size, resample=Image.BILINEAR)
        mask_array = np.array(mask_resized)
        # Blend original image and color mask
        blended_image = (1 - alpha) * image_array + alpha * mask_array
        blended_image = blended_image.astype(np.uint8)
        return Image.fromarray(blended_image)

    # Initialize the YOLO model
    model_directory = '../Models/segformer-b5-finetuned-human-parsing'
    model_id = 'matei-dorian/segformer-b5-finetuned-human-parsing'
    model, processor = load_model_and_extractor(model_directory, model_id)

    # Set up the display window name
    cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)

    while True:
        screen = ImageGrab.grab()
        screen_resized = screen.resize(resize_dimension, Resampling.LANCZOS)
        inputs = processor(images=screen_resized, return_tensors='pt')
        inputs = inputs.to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
        segmentation_mask = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()

        draw = ImageDraw.Draw(screen_resized)
        
        original_image = screen  # Load your original image
        labels = ["Background", "Hat", "Hair", "Sunglasses", " Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"]
        decoded_mask = decode_segmentation_mask(segmentation_mask, labels)
        blended_image = apply_mask_on_image(original_image, decoded_mask, alpha=0.5)

        screen_np = np.array(blended_image)
        # Convert the color space from BGR (OpenCV default) to RGB
        screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)
        
        # Display the captured screen
        cv2.imshow('Screen Capture', screen_np)

        # Wait for 1000 milliseconds (1 second)
        key = cv2.waitKey(1000//frames_per_second) & 0xFF
        if key == ord('q'):
            break

        # Check if the window is closed
        if cv2.getWindowProperty('Screen Capture', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    default_frames_per_second = 64
    default_resize_dimension = (768, 432)
    default_object_confidence_threshold = 0.32
    
    screen_capture_and_detect_yolo_v8(model_variants = "LARGE",
                                      frames_per_second = default_frames_per_second,
                                      resize_dimension = (1920, 1080), 
                                      person_detection_only=False, 
                                      remove_large_objects=True, 
                                      verbose=True)
    #screen_capture_and_detect_segformer_b5(default_frames_per_second, (1920, 1080))
    
