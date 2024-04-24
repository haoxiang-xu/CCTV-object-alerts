import numpy as np
import cv2
from PIL import ImageGrab, Image, ImageDraw, ImageFont
from PIL.Image import Resampling
from ultralytics import YOLO
from ultralytics.engine.results import Results
        
def screen_capture(fames_per_second = 16):
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
        if cv2.waitKey(1000//fames_per_second) & 0xFF == ord('q'):
            break

        # Check if the window is closed
        if cv2.getWindowProperty('Screen Capture', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()
def screen_capture_and_detect(fames_per_second = 16, resize_dimension = (640, 480)):
    # Initialize the YOLO model
    model = YOLO()  # Adjust the path as necessary

    # Set up the display window name
    cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)
    
    # Load the font for the labels
    font = ImageFont.load_default()
    bg_color_ranges = {"#FF0000_#181818": [0,1], "#FF9900_#181818": [1,14], "#341A36_#FFFFFF": [14,24], "#00C036_#181818": [24,80]}
    color_labels = []
    skip_classes = [62, 63, 72]
    for i in range(0,80):
        for color in bg_color_ranges:
            if i >= bg_color_ranges[color][0] and i < bg_color_ranges[color][1]:
                color_labels.append(color)
                break
    
    while True:
        # Capture the screen
        screen = ImageGrab.grab()
        screen_resized = screen.resize(resize_dimension, Resampling.LANCZOS)
        # Convert the image to an array
        results = model(screen_resized)
        result = results[0] # Get the first result
        boxes = result.boxes.xyxy # Get the bounding boxes
        cls = result.boxes.cls.tolist() # Get the class IDs
        conf = result.boxes.conf.tolist() # Get the confidence values
        
        names = result.names

        draw = ImageDraw.Draw(screen_resized)
        for index in range(len(boxes)):
            if round(conf[index], 2) < 0.32:
                continue
            if cls[index] in skip_classes:
                continue
            
            box_data = boxes[index].tolist()  # Convert tensor to list
            filling_color = color_labels[int(cls[index])].split("_")

            # Check if the length of the box_data matches expected number of elements for just coordinates
            if len(box_data) == 4:
                x1, y1, x2, y2 = box_data  # Unpack the coordinates

                cls_label = names[cls[index]]  # Get the class name using class ID or default to "Unknown"
                conf_label = int(round(conf[index], 2)*100)  # Get the confidence and convert to percentage
                label = f"{cls_label} {conf_label}%"  # Create label with class name and confidence

                draw.rectangle([x1, y1, x2, y2], outline=filling_color[0], width=3)  # Draw the rectangle
                text_bg = [x1, max(y1 - 16,0), x1 + (len(cls_label)+5) * 6, max(16,y1)] # Create background rectangle for text
                draw.rectangle(text_bg, fill=filling_color[0])
                draw.text((x1+2, max(y1 - 16,0)), label, fill=filling_color[1], font=font)  # Draw the label    
            else:
                print("Unexpected box data format:", box_data)  # Add an error message
        
        screen_np = np.array(screen_resized)
        # Convert the color space from BGR (OpenCV default) to RGB
        screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)

        # Display the captured screen
        cv2.imshow('Screen Capture', screen_np)

        # Wait for 1000 milliseconds (1 second)
        if cv2.waitKey(1000//fames_per_second) & 0xFF == ord('q'):
            break

        # Check if the window is closed
        if cv2.getWindowProperty('Screen Capture', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    default_fames_per_second = 16
    default_resize_dimension = (768, 432)
    screen_capture_and_detect(default_fames_per_second, default_resize_dimension)






