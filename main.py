import cv2
import pickle
import cvzone
import numpy as np

# Video feed
cap = cv2.VideoCapture('./media/carPark.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Load pre-defined parking space positions
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    print("Error: CarParkPos file not found.")
    exit()

width, height = 107, 48

def checkParkingSpace(imgPro):
    spaceCounter = 0
    occupiedCounter = 0

    for pos in posList:
        x, y = pos

        # Cropping the image
        imgCrop = imgPro[y:y + height, x:x + width]

        # Count the non-zero pixels in the cropped image
        count = cv2.countNonZero(imgCrop)

        # Set rectangle color and thickness based on occupancy
        if count < 900:
            color = (0, 255, 0)  # Green for empty space
            spaceCounter += 1
        else:
            color = (0, 0, 255)  # Red for occupied space
            occupiedCounter += 1

        thickness = 5 if count < 900 else 2

        # Draw rectangle and put text on the frame
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 5), scale=1.5, thickness=2, offset=0, colorR=color)

    # Display the counts of empty and occupied spaces
    cvzone.putTextRect(img, f'Empty: {spaceCounter}/{len(posList)}', (100, 50), scale=3, thickness=5, offset=20, colorR=(0, 255, 0))
    cvzone.putTextRect(img, f'Occupied: {occupiedCounter}/{len(posList)}', (100, 100), scale=3, thickness=5, offset=20, colorR=(0, 0, 255))

while True:
    # Loop the video
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    checkParkingSpace(imgDilate)

    # Display the frames
    cv2.imshow("Image", img)
    
    # Wait for a key press to slow down the video
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()