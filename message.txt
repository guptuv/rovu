import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO

# Hypothetical rover control module
#import rover_control  # This module needs to be implemented based on your rover's hardware

# Load the YOLOv8 model
yolo_model = YOLO('yolov8n.pt')

# Load the DeepLabV3 model
seg_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
seg_model.eval()

# Preprocessing function for DeepLabV3
preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Color palette for segmentation map
palette = torch.tensor([
    [0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70],
    [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30],
    [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180],
    [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
])

def segment_image(image):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    with torch.no_grad():
        output = seg_model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    segmented_img = palette[output_predictions.numpy()]
    return segmented_img

def decide_action(detections, frame_width):
    """
    Decide the rover's action based on the detected objects.
    """
    action_taken = False
    for result in detections:
        x1, y1, x2, y2, conf, cls = result
        label = yolo_model.names[int(cls)]
        
        # Calculate the center of the bounding box
        center_x = (x1 + x2) / 2
        width = x2 - x1

        """ if label == 'person':
            if conf > 0.5:
                if width > frame_width / 3:  # Person is very close
                    #rover_control.stop()
                    action_taken = True
                    break
                elif center_x < frame_width / 3:
                    #rover_control.turn_left()
                    action_taken = True
                    break
                elif center_x > frame_width * 2 / 3:
                    #rover_control.turn_right()
                    action_taken = True
                    break
                else:
                    #rover_control.stop()
                    action_taken = True
                    break

        elif label in ['car', 'truck', 'motorcycle']:
            if conf > 0.5:
                if width > frame_width / 3:  # Vehicle is very close
                    #rover_control.stop()
                    action_taken = True
                    break
                elif center_x < frame_width / 3:
                    #rover_control.turn_left()
                    action_taken = True
                    break
                elif center_x > frame_width * 2 / 3:
                    #rover_control.turn_right()
                    action_taken = True
                    break
                else:
                    #rover_control.stop()
                    action_taken = True
                    break

        elif label == 'stop sign':
            if conf > 0.5:
                #rover_control.stop()
                action_taken = True
                break

        elif label == 'traffic light':
            if conf > 0.5:
                # Implement logic based on the traffic light color (if detected)
                #rover_control.stop()  # Default action
                action_taken = True
                break

        elif label == 'pothole':
            if conf > 0.5:
                if center_x < frame_width / 3:
                    #rover_control.turn_left()
                elif center_x > frame_width * 2 / 3:
                    #rover_control.turn_right()
                else:
                    #rover_control.slow_down()
                action_taken = True
                break

        elif label == 'lane marking':
            if conf > 0.5:
                # Implement lane-following logic
                #rover_control.stay_in_lane()
                action_taken = True
                break

        elif label == 'road sign':
            if conf > 0.5:
                # Implement logic for different road signs (e.g., speed limit)
                #rover_control.follow_road_sign()
                action_taken = True
                break

        elif label == 'crosswalk':
            if conf > 0.5:
                #rover_control.yield_to_pedestrians()
                action_taken = True
                break

        elif label in ['dog', 'cat']:
            if conf > 0.5:
                #rover_control.stop()
                action_taken = True
                break

    if not action_taken:
        #rover_control.move_forward() """

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to PIL image
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform object detection
    results = yolo_model(pil_img)
    yolo_results = results[0].boxes.data.numpy()  # Extract boxes

    # Perform semantic segmentation
    segmented_img = segment_image(pil_img)

    # Convert segmented image to BGR for OpenCV display
    segmented_img_bgr = cv2.cvtColor(np.array(segmented_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)

    # Draw YOLO results on the frame
    for result in yolo_results:
        x1, y1, x2, y2, conf, cls = result
        label = f"{yolo_model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Combine the original frame and segmented image side by side
    combined_img = np.hstack((frame, segmented_img_bgr))

    # Display the combined image
    cv2.imshow('Object Detection and Semantic Segmentation', combined_img)

    # Decide and take action based on detections
    decide_action(yolo_results, frame.shape[1])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
