import cv2
import numpy as np
import time
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
import mediapipe as mp

class InvisibleCloak:
    def __init__(self, lower_color, upper_color, num_bg_frames=30):
        self.lower_color = lower_color
        self.upper_color = upper_color
        self.num_bg_frames = num_bg_frames
        self.cap = cv2.VideoCapture(0)
        self.background = None
        self.frame_count = 0
        self.snapshot_dir = "snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)

        weights = MobileNet_V2_Weights.DEFAULT
        self.model = models.mobilenet_v2(weights=weights)
        self.model.eval()
        self.transform = weights.transforms()
        self.classes = weights.meta["categories"]

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.multi_tracker = cv2.MultiTracker_create()

    def create_background(self):
        print("Capturing background. Please move out of frame.")
        backgrounds = []
        for i in range(self.num_bg_frames):
            ret, frame = self.cap.read()
            if ret:
                backgrounds.append(frame)
            else:
                print(f"Warning: Could not read frame {i+1}/{self.num_bg_frames}")
            time.sleep(0.1)
        if backgrounds:
            self.background = np.median(backgrounds, axis=0).astype(np.uint8)
        else:
            raise ValueError("Could not capture any frames for background")

    def create_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
        return mask

    def apply_cloak_effect(self, frame, mask):
        mask_inv = cv2.bitwise_not(mask)
        fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        bg = cv2.bitwise_and(self.background, self.background, mask=mask)
        return cv2.add(fg, bg)

    def display_frame_info(self, frame):
        self.frame_count += 1
        height, width = frame.shape[:2]
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, time.strftime("%H:%M:%S"), (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame

    def save_snapshot(self, frame):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.snapshot_dir, f"snapshot_{timestamp}.png")
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved as {filename}")

    def toggle_background_recalibration(self):
        print("Recalibrating background. Please move out of frame.")
        try:
            self.create_background()
            print("Background recalibrated.")
        except ValueError as e:
            print(f"Error during recalibration: {e}")

    def analyze_frame_with_ai(self, frame):
        try:
            input_tensor = self.transform(frame).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
            return self.classes[predicted.item()]
        except Exception as e:
            print(f"AI analysis error: {e}")
            return None

    def detect_gestures(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                distances = [
                    np.linalg.norm(np.array([landmarks[i].x, landmarks[i].y]) - np.array([landmarks[j].x, landmarks[j].y]))
                    for i, j in [(4, 8), (8, 12), (12, 16), (16, 20)]
                ]
                if all(d < 0.05 for d in distances):
                    cv2.putText(frame, "Fist Gesture", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif distances[0] < 0.05:
                    cv2.putText(frame, "Pinch Gesture", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                elif distances[1] < 0.05:
                    cv2.putText(frame, "Peace Gesture", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Hand Detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return frame

    def initialize_object_tracking(self, frame):
        bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker = cv2.TrackerCSRT_create()
        self.multi_tracker.add(tracker, frame, bbox)
        cv2.destroyWindow("Frame")

    def update_object_tracking(self, frame):
        success, boxes = self.multi_tracker.update(frame)
        for i, newbox in enumerate(boxes):
            x, y, w, h = [int(v) for v in newbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(frame, f"Object {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        return frame

    def run(self):
        print("OpenCV version:", cv2.__version__)

        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        try:
            self.create_background()
        except ValueError as e:
            print(f"Error: {e}")
            self.cap.release()
            return

        print("Starting main loop. Press 'q' to quit, 's' to save a snapshot, 'r' to recalibrate background, 't' to track object.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame.")
                time.sleep(1)
                continue

            mask = self.create_mask(frame)
            result = self.apply_cloak_effect(frame, mask)
            result = self.display_frame_info(result)

            ai_prediction = self.analyze_frame_with_ai(frame)
            if ai_prediction:
                cv2.putText(result, f"AI: {ai_prediction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            result = self.detect_gestures(result)
            result = self.update_object_tracking(result)

            cv2.imshow('Invisible Cloak', result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_snapshot(result)
            elif key == ord('r'):
                self.toggle_background_recalibration()
            elif key == ord('t'):
                self.initialize_object_tracking(frame)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    cloak = InvisibleCloak(lower_blue, upper_blue)
    cloak.run()
