#!/bin/sh

# Start face emotion pipeline, then launch xiaozhi_client (matches original start order)
./ui_and_ai face_detection_320.kmodel 0.6 0.2 face_emotion.kmodel None 0 &
sleep 1
./xiaozhi_client "wakeup" &