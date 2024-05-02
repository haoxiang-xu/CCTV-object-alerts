import os
import cv2
import time
import torch
import requests
import datetime
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
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from torchvision import transforms

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
def screen_detection_segmentation(yolo_v8_size = "NANO", # [OPTIONS] "NANO", "SMALL", "MEDIUM", "LARGE", "EXTRA-LARGE"
                                  max_frames_per_second = 16, 
                                  input_frame_dimension = (640, 480), 
                                  # Filtering Conditions ------------------------------------------------------------------------------------- Filtering Conditions
                                  object_confidence_threshold = 0.08,
                                  remove_oversize_objects = True, 
                                  person_detection_only = False,
                                  # Additional Functionalties --------------------------------------------------------------------------- Additional Functionalties
                                  segmentation_on_person_option = "NONE", # [OPTIONS] "NONE", "SEGFORMER-B5", "SEGFORMER-B5-MAX-SIZE/X", "SEGFORMER-B5-MAX-CONF/X"
                                  segmentation_on_officers_option = "NONE", # [OPTIONS] "NONE", "MARK", "SEGMENT", "MARK_WITH_CONF_N/X", "SEGMENT_WITH_CONF_N/X"
                                  collecting_person_patches = "NONE", # [OPTIONS] "NONE", "SAVE_EVERY_N_SECONDS/X", "SAVE_OFFICERS_ONLY_EVERY_N_SECONDS/X", "SAVE_CIVILIANS_ONLY_EVERY_N_SECONDS/X"
                                  # Verbose --------------------------------------------------------------------------------------------------------------- Verbose
                                  verbose = False):
    #FUNCTIONS -------------------------------------------------------------------------------------------------------------------FUNCTIONS
    def loading_label_fonts():
        font_path_mac = "./FONTs/STHeiti Light.ttc"
        font_path_win = "./FONTs/Quicksand-VariableFont_wght.ttf"

        if platform.system() == "Darwin":  # macOS
            try:
                font = ImageFont.truetype(font_path_mac, 16)
            except IOError as e:
                print(f"[ERROR] -------- [Failed to load macOS font from {font_path_mac} due to {e}]")
        elif platform.system() == "Windows":  # Windows
            try:
                font = ImageFont.truetype(os.path.abspath(font_path_win), 16)
            except IOError as e:
                print(f"[ERROR] -------- [Failed to load Windows font from {os.path.abspath(font_path_win)} due to {e}]")
        else:
            print(f"[WARNING] ------ [Unsupported OS. Loading default font.]")
            font = ImageFont.load_default()
        return font       
    def loading_yolo_v8_models(yolo_v8_size, device):
        if yolo_v8_size == "NANO":
            model = YOLO("./MODELs/yolov8n").to(device)
        elif yolo_v8_size == "SMALL":
            model = YOLO("./MODELs/yolov8s").to(device)
        elif yolo_v8_size == "MEDIUM":
            model = YOLO("./MODELs/yolov8m").to(device)
        elif yolo_v8_size == "LARGE":
            model = YOLO("./MODELs/yolov8l").to(device)
        elif yolo_v8_size == "EXTRA-LARGE":
            model = YOLO("./MODELs/yolov8x").to(device)
        else:
            model = YOLO("./MODELs/yolov8n").to(device)
        print(f"[PROCESS] ------ [Yolo v8 {yolo_v8_size} is loaded into {device}]")
        return model
    def loading_segformer_b5_models(device):
        model_dir = "./Models/segformer-b5-finetuned-human-parsing"
        print(f"[PROCESS] ------ [Segformer b5 is loaded into {device}]")
        model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device)
        image_processor = SegformerImageProcessor.from_pretrained(model_dir)
        return model, image_processor
    def loading_binary_classification_resnet_model(model_size = 'resnet18', fine_tune = False):
        os.environ['TORCH_HOME'] = './MODELs'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_size == 'resnet18':
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
        elif model_size == 'resnet50':
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
        elif model_size == 'resnet101':
            weights = ResNet101_Weights.DEFAULT
            model = resnet101(weights=weights)
        else:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
        
        if fine_tune:
            for param in model.parameters():
                param.requires_grad = False
        
        num_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, 1),
            torch.nn.Sigmoid()
        )
        
        model.load_state_dict(torch.load('./MODELs/officer_classificaiton_resnet18_weights.pth'))
        print(f"[PROCESS] ------ [Classification {model_size} is loaded into {device}]")
        return model.to(device)
    def convert_input_dimension_to_GPU_dimension(input_size, stride=32):
        # Adjust size so it's divisible by the model's stride
        new_width = (input_size[0] // stride) * stride
        new_height = (input_size[1] // stride) * stride
        return (new_width, new_height)
    def filter_overlapping_detectations(boxes, classes, confidences):
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
    def filter_segmentation_await_box_index(segmentation_on_person_option, boxes, classes, confidences):
        if segmentation_on_person_option == "NONE":
            return None
        elif segmentation_on_person_option == "SEGFORMER-B5":
            return [index for index, cls in enumerate(classes) if cls == 0]
        elif "SEGFORMER-B5-MAX-SIZE" in segmentation_on_person_option:
            x = int(segmentation_on_person_option.split("/")[1])
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            cls = np.array(classes)
            indices_where_cls_is_zero = np.where(cls == 0)[0]
            filtered_boxes = boxes[indices_where_cls_is_zero]
            try:
                areas = (filtered_boxes[:, 2] - filtered_boxes[:, 0]) * (filtered_boxes[:, 3] - filtered_boxes[:, 1])
            except IndexError:
                print(f"[ERROR] -------- [Failed to Sort Boxes by Area due to IndexError]")
                return []
            sorted_indices_by_area = np.argsort(-areas)
            top_x_indices = indices_where_cls_is_zero[sorted_indices_by_area][:x]
            return top_x_indices
        elif "SEGFORMER-B5-MAX-CONF" in segmentation_on_person_option:
            x = int(segmentation_on_person_option.split("/")[1])
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            cls = np.array(classes)
            conf = np.array(confidences)
            indices_where_cls_is_zero = np.where(cls == 0)[0]
            filtered_conf = conf[indices_where_cls_is_zero]
            try:
                sorted_indices = np.argsort(-filtered_conf)
                original_indices_sorted_by_conf = indices_where_cls_is_zero[sorted_indices]
            except IndexError:
                print(f"[ERROR] -------- [Failed to Sort Boxes by Conf due to IndexError]")
                return []
            return original_indices_sorted_by_conf[:x]
        else:
            return None
    def decode_segmentation_mask(mask, labels):
        label_colors = np.array([
            [0, 0, 0],         # Background ----- Black
            [255, 255, 255],   # Hat --------------------------------------------------- White
            [133, 27, 27],     # Hair ----------- Brown#1
            [255, 0, 0],       # Sunglasses
            [255, 61, 0],      # Upper-clothes ---------------------- Red#2
            [42, 21, 171],     # Skirt ----------- Blue#2
            [255, 255, 255],   # Pants ------------------------------------------------- White
            [98, 28, 15],      # Dress ---------- Brown#2
            [128, 0, 128],     # Belt
            [154, 23, 156],    # Left-shoe ------------------------ Purple    
            [154, 23, 156],    # Right-shoe ----------------------- Purple
            [236, 150, 135],   # Face ------------------ Orange
            [236, 150, 135],   # Left-leg -------------- Orange
            [236, 150, 135],   # Right-leg ------------- Orange
            [236, 150, 135],   # Left-arm -------------- Orange
            [236, 150, 135],   # Right-arm ------------- Orange
            [75, 0, 130],      # Bag
            [0, 128, 128]      # Scarf
        ])
        # Ensure label_colors covers all the labels present in the mask
        color_mask = label_colors[mask]
        return Image.fromarray(color_mask.astype(np.uint8))
    def apply_segmentation_mask_on_picture(original_image, mask, alpha=0.96):
        image_array = np.array(original_image)
        mask_resized = mask.resize(original_image.size, resample=Image.BILINEAR)
        mask_array = np.array(mask_resized)
        blended_image = (1 - alpha) * image_array + alpha * mask_array
        blended_image = blended_image.astype(np.uint8)
        return Image.fromarray(blended_image)
    def segmentation_picture_with_segformer_b5(person_patch, model, processor, SEGFORMER_B5_CLOTHING_LABELS):
        raw_model_inputs = processor(images=person_patch, return_tensors='pt')
        inputs = raw_model_inputs.to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        segmentation_mask = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()
        
        original_picture = raw_model_inputs
        decoded_mask = decode_segmentation_mask(segmentation_mask, SEGFORMER_B5_CLOTHING_LABELS)
        blended_picture = apply_segmentation_mask_on_picture(person_patch, decoded_mask, alpha=0.5)
        
        return blended_picture
    def extract_and_save_person_patches(collecting_person_patches, person_patch, conf, CURRENT_FRAME_TIME, PREVIOUS_FRAME_TIME, COLLECTING_PERSON_PATCHES_EVERY_N_SECONDS, officer_classification_prediction):
        if "SAVE_OFFICERS_ONLY_EVERY_N_SECONDS" in collecting_person_patches and (CURRENT_FRAME_TIME - PREVIOUS_FRAME_TIME > COLLECTING_PERSON_PATCHES_EVERY_N_SECONDS or CURRENT_FRAME_TIME == PREVIOUS_FRAME_TIME) and officer_classification_prediction == 1:
            folder_path = "./DATA"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"person_patch_{timestamp}_{round(conf,2)}.png"
            person_patch.save(os.path.join(folder_path, filename))
            return True
        if "SAVE_CIVILIANS_ONLY_EVERY_N_SECONDS" in collecting_person_patches and (CURRENT_FRAME_TIME - PREVIOUS_FRAME_TIME > COLLECTING_PERSON_PATCHES_EVERY_N_SECONDS or CURRENT_FRAME_TIME == PREVIOUS_FRAME_TIME) and officer_classification_prediction == 0:
            folder_path = "./DATA"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"person_patch_{timestamp}_{round(conf,2)}.png"
            person_patch.save(os.path.join(folder_path, filename))
            return True
        if "SAVE_EVERY_N_SECONDS" in collecting_person_patches and CURRENT_FRAME_TIME - PREVIOUS_FRAME_TIME > COLLECTING_PERSON_PATCHES_EVERY_N_SECONDS or CURRENT_FRAME_TIME == PREVIOUS_FRAME_TIME and officer_classification_prediction == -1:
            folder_path = "./DATA"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"person_patch_{timestamp}_{round(conf,2)}.png"
            person_patch.save(os.path.join(folder_path, filename))
            return True
        return False  
    def classify_officers_with_classification_resnet(model, person_patch, segmentation_on_officers_option):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if "WITH_CONF_N" in segmentation_on_officers_option:
            threshold = int(segmentation_on_officers_option.split("/")[1])/100
        
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image = person_patch.convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                output = model(image)
                if output.item() > threshold:
                    prediction = 1
                else:
                    prediction = 0
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image = person_patch.convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                output = model(image)
                prediction = output.round().item()
        return prediction
    #FUNCTIONS ----------------------------------------------------------------------------------------------------------------------------   
     
    #CONSTs -------------------------------------------------------------------------------------------------------------------------CONSTs
    YOLOV8_SKIPPING_CLASSES = [4, 6, 62, 63, 72]
    YOLOV8_LABEL_BACKGROUND_COLORS = {"#FF0000_#181818": [0,1], "#FF9900_#181818": [1,14], "#341A36_#FFFFFF": [14,24], "#00C036_#181818": [24,80]}
    ADJUSTED_DIMENSION = convert_input_dimension_to_GPU_dimension(input_frame_dimension)
    SEGFORMER_B5_CLOTHING_LABELS = ["Background", "Hat", "Hair", "Sunglasses", " Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"]
    FONT = loading_label_fonts()
    LABEL_BACKGROUND_COLOR = []
    COLLECTING_PERSON_PATCHES_EVERY_N_SECONDS = 0
    if collecting_person_patches != "NONE":
        COLLECTING_PERSON_PATCHES_EVERY_N_SECONDS = int(collecting_person_patches.split("/")[1])
    for i in range(0,80):
        for color in YOLOV8_LABEL_BACKGROUND_COLORS:
            if i >= YOLOV8_LABEL_BACKGROUND_COLORS[color][0] and i < YOLOV8_LABEL_BACKGROUND_COLORS[color][1]:
                LABEL_BACKGROUND_COLOR.append(color)
                break
    #CONSTs -------------------------------------------------------------------------------------------------------------------------------
    
    #OUTER STATIC VARIABLES -----------------------------------------------------------------------------------------OUTER STATIC VARIABLES
    TOTAL_OBJECTS_REMOVED = 0
    TOTAL_OBJECTS_REMOVED_BY_OVERLAPPING = 0
    TOTAL_OBJECTS_REMOVED_BY_CONFIDENCE = 0
    TOTAL_OVERSIZE_OBJECTS_REMOVED = 0
    PREVIOUS_FRAME_TIME = time.time()
    #OUTER STATIC VARIABLES ---------------------------------------------------------------------------------------------------------------
    
    #RECORDING VARIABLES ===============================================================================================RECORDING VARIABLES
    is_recording = False
    video_writer = None
    video_file_path = "output.avi"
    #RECORDING VARIABLES ==================================================================================================================

    #LOAD REQUIRED MODELS ---------------------------------------------------------------------------------------------LOAD REQUIRED MODELS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_v8_model = loading_yolo_v8_models(yolo_v8_size, device)
    if segmentation_on_person_option != "NONE":
        segformer_b5_model, segformer_b5_image_processor = loading_segformer_b5_models(device)
    if segmentation_on_officers_option != "NONE":
        classification_officers_resnet_model = loading_binary_classification_resnet_model()
    #LOAD REQUIRED MODELS -----------------------------------------------------------------------------------------------------------------

    cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)

    while True:
        #INTER STATIC VARIABLES ------------------------------------------------------------------------------------------------INTER STATIC VARIABLES
        OBJECT_REMOVED_FOR_THIS_FRAME = 0
        CURRENT_FRAME_TIME = time.time()
        #INTER STATIC VARIABLES ----------------------------------------------------------------------------------------------------------------------
        
        original_picture = ImageGrab.grab()
        reshape_picture = original_picture.resize(ADJUSTED_DIMENSION, Resampling.LANCZOS)
        drawing_picture = reshape_picture.copy()
        tensor_picture = torch.from_numpy(np.array(drawing_picture)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        raw_model_output = yolo_v8_model(tensor_picture, verbose=False)
        
        # EXTRACT RESULTS -------------------------------------------------------------------------------------------------------------------
        model_output = raw_model_output[0]
        boxes = model_output.boxes.xyxy
        cls = model_output.boxes.cls.tolist()
        conf = model_output.boxes.conf.tolist()
        names = model_output.names
        boxes, cls, conf, obejct_removed_by_overlapping = filter_overlapping_detectations(boxes, cls, conf)
        if segmentation_on_person_option != "NONE":
            segmentation_await_box_index =  filter_segmentation_await_box_index(segmentation_on_person_option, boxes, cls, conf)
        # EXTRACT RESULTS -------------------------------------------------------------------------------------------------------------------

        draw = ImageDraw.Draw(drawing_picture)
        
        # DRAWING =============================================================================================================DRAWING
        for index in range(len(boxes)):
            box_data = boxes[index].tolist()
            # INTER LOOP FILTERING CONDITIONS ===============================================================================================
            if len(box_data) != 4:
                continue
            else:
                x1, y1, x2, y2 = box_data
            if round(conf[index], 2) < object_confidence_threshold:
                OBJECT_REMOVED_FOR_THIS_FRAME += 1
                continue
            if cls[index] in YOLOV8_SKIPPING_CLASSES:
                OBJECT_REMOVED_FOR_THIS_FRAME += 1
                TOTAL_OBJECTS_REMOVED_BY_CONFIDENCE += 1
                continue
            if person_detection_only and cls[index] != 0:
                OBJECT_REMOVED_FOR_THIS_FRAME += 1
                continue
            if remove_oversize_objects and abs(x2-x1) * abs(y2-y1) > 1/3 * (input_frame_dimension[0] * input_frame_dimension[1]) or abs(x2-x1) > 4/6 * input_frame_dimension[0]:
                OBJECT_REMOVED_FOR_THIS_FRAME += 1
                TOTAL_OVERSIZE_OBJECTS_REMOVED += 1
                continue
            # INTER LOOP FILTERING CONDITIONS ===============================================================================================
        
            cls_label = names[cls[index]]
            conf_label = int(round(conf[index], 2)*100)
            label = f"{cls_label} {conf_label}%"
            filling_color = LABEL_BACKGROUND_COLOR[int(cls[index])].split("_")
            
            # EXTRACT PERSON PATCH ===================================================================================EXTRACT PERSON PATCH
            officer_classification_prediction = -1
            if segmentation_on_person_option != "NONE" and segmentation_await_box_index is not None and index in segmentation_await_box_index:
                person_patch = reshape_picture.crop((int(x1), int(y1), int(x2), int(y2)))
                blended_patch = segmentation_picture_with_segformer_b5(person_patch, segformer_b5_model, segformer_b5_image_processor, SEGFORMER_B5_CLOTHING_LABELS)
                drawing_picture.paste(blended_patch, (int(x1), int(y1)))
            if segmentation_on_officers_option != "NONE" and cls[index] == 0:
                person_patch = reshape_picture.crop((int(x1), int(y1), int(x2), int(y2)))
                officer_classification_prediction = classify_officers_with_classification_resnet(classification_officers_resnet_model, person_patch, segmentation_on_officers_option)
                if officer_classification_prediction == 1:
                    label = f"OFFICER"
                    filling_color = ["#FF0000", "#FFFFFF"]
                elif officer_classification_prediction == 0:
                    if "SEGMENT" in segmentation_on_officers_option:
                        continue
                    label = f"CIVILIAN"
                    filling_color = ["#00FF00", "#000000"]
                else:
                    label = f"{cls_label} {conf_label}%"
            if collecting_person_patches != "NONE":
                person_patch = reshape_picture.crop((int(x1), int(y1), int(x2), int(y2)))
                if extract_and_save_person_patches(collecting_person_patches, person_patch, conf[index], CURRENT_FRAME_TIME, PREVIOUS_FRAME_TIME, COLLECTING_PERSON_PATCHES_EVERY_N_SECONDS, officer_classification_prediction):
                    PREVIOUS_FRAME_TIME = CURRENT_FRAME_TIME
            # EXTRACT PERSON PATCH =======================================================================================================
            
            # DRAWING =============================================================================================================DRAWING
            draw.rectangle([x1, y1, x2, y2], outline=filling_color[0], width=3)
            text_bg = [x1, max(y1 - 16,0), x1 + (len(cls_label)+5) * 9, max(16,y1)]
            draw.rectangle(text_bg, fill=filling_color[0])
            draw.text((x1+2, max(y1 - 16,0)), label, fill=filling_color[1], font=FONT) 
            # DRAWING ==================================================================================================================== 
        # DRAWING ====================================================================================================================
        
        if is_recording:
            draw.rectangle([0, 0, input_frame_dimension[0], 4], fill="red", width=3)
        screen_np = np.array(drawing_picture)
        screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)
        if is_recording:
            if video_writer is None:  # Start new video writer
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(video_file_path, fourcc, max_frames_per_second, input_frame_dimension)
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
                      f"\tOVERSIZE OBJECTS REMOVED: {TOTAL_OVERSIZE_OBJECTS_REMOVED}\n")

        # WINDOW CONTROL ==============================================================================================WINDOW CONTROL
        key = cv2.waitKey(1000//max_frames_per_second) & 0xFF
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
        # WINDOW CONTROL ============================================================================================================

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
    
    screen_detection_segmentation(yolo_v8_size = "NANO",
                                      max_frames_per_second = default_frames_per_second,
                                      input_frame_dimension = (1920, 1080), 
                                      # Filtering Conditions ------------------------------------------------------------------------------------- Filtering Conditions
                                      person_detection_only = True, 
                                      object_confidence_threshold = 0.32,
                                      remove_oversize_objects=True, 
                                      # Additional Functionalties --------------------------------------------------------------------------- Additional Functionalties
                                      segmentation_on_person_option = "NONE",
                                      segmentation_on_officers_option = "SEGMENT",
                                      collecting_person_patches = "SAVE_OFFICERS_ONLY_EVERY_N_SECONDS/6",#"SAVE_CIVILIANS_ONLY_EVERY_N_SECONDS/1",
                                      # Verbose --------------------------------------------------------------------------------------------------------------- Verbose
                                      verbose=False)
    #screen_capture_and_detect_segformer_b5(default_frames_per_second, (1920, 1080))
    
