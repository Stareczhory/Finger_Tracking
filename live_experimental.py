import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import handle_image
import time
import math

VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

current_frame = None
coordinates = None
cords_num_array = [8, 12, 16, 20]
cords_angles = []

def calculate_angle(cords: HandLandmarkerResult):
    hand_landmarker_result = cords
    coordinate_values = []
    for landmarks in hand_landmarker_result.hand_landmarks:
        for idx, landmark in enumerate(landmarks):
            coordinate_values.append(landmark)
    if len(coordinate_values) != 0:
        num_counter = 0
        for _ in cords_num_array:
            # length from origin (landmark 0) to midpoint
            length_c = math.sqrt((coordinate_values[8+num_counter].z - coordinate_values[0].z)**2 + (coordinate_values[5+num_counter].y - coordinate_values[0].y)**2)
            # length from midpoint to endpoint
            length_b = math.sqrt((coordinate_values[5+num_counter].z - coordinate_values[8+num_counter].z)**2 + (coordinate_values[5+num_counter].y - coordinate_values[8+num_counter].y)**2)
            # length from endpoint to origin
            length_d = math.sqrt((coordinate_values[8+num_counter].z - coordinate_values[0].z)**2 + (coordinate_values[8+num_counter].y - coordinate_values[0].y)**2)
            angle_cb = math.acos((length_d**2 - length_b**2 - length_c**2)/-(2*length_c*length_b))
            cords_angles.append(round(math.degrees(angle_cb), 2))
            num_counter += 4
# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global current_frame, coordinates
    annotated_image = handle_image.draw_landmarks_on_image(output_image.numpy_view(), result)
    current_frame = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)
    coordinates = result


base_options = python.BaseOptions(model_asset_path=r'C:\Users\jakub\Downloads\hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1,
                                       running_mode=VisionRunningMode.LIVE_STREAM,
                                       result_callback=print_result)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # reset the 2-dimensional list
    # get current time that past in ms (last three digits)
    timestamp = int(time.time() * 1000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    detector.detect_async(mp_image, timestamp)
    if current_frame is not None:
        cv.imshow('Hand Tracking', current_frame)
        cords_angles = []
        calculate_angle(coordinates)
        print(cords_angles)

    else:
        cv.imshow('Hand Tracking', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()