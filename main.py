"""
Main project file
"""

import time
import cv2
from utils import place_greenscreen_image, get_probable_eyes, resize_image
from common import *


def get_frame(input_stream):
    """
    Get the current frame to analise.
    """
    if not input_stream.isOpened() or input_stream is None:
        print("unable to get stream, releasing resources...")
        input_stream.release()

        print("getting image...")
        frame = cv2.imread(TEST_IMAGE)
    else:
        _, frame = input_stream.read()
    return frame


def process(frame):
    """
    Gets a frame, does some CV magic, returns a processed frame.
    """
    # Load assets
    face_casc = cv2.CascadeClassifier(FACE_CASC_PATH)
    eyes_casc = cv2.CascadeClassifier(EYES_CASC_PATH)

    anime_eye = cv2.imread(EYE_PATH)
    # mustache = cv2.imread(MUSTACHE_PATH)

    faces = face_casc.detectMultiScale(frame, scaleFactor=1.1, minSize=(30, 30), minNeighbors=5,
                                       flags=cv2.CASCADE_SCALE_IMAGE)

    for (face_x, face_y, face_w, face_h) in faces:

        frame = cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), COLOR_RED)

        face_area = frame[face_y:face_y + face_h, face_x:face_x + face_w]

        potential_eyes = eyes_casc.detectMultiScale(face_area, scaleFactor=1.1, minSize=(5, 5),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)

        probable_eyes_list = get_probable_eyes(potential_eyes)

        for (eye_x, eye_y, eye_w, eye_h) in probable_eyes_list:
            eye_x += face_x
            eye_y += face_y

            # resized_mustache = resize_image(mustache, new_width=eye_w * 1.5)
            # resized_mustache_height = resized_mustache.shape[0]
            #
            # frame = place_greenscreen_image(frame, resized_mustache, eye_x, eye_y - (resized_mustache_height // 2))

            resized_anime_eye = resize_image(anime_eye, new_width=eye_w * 1.2)

            frame = place_greenscreen_image(frame, resized_anime_eye, eye_x, eye_y)

    return frame


def start_processing_loop():
    """
    Main processing loop.
    """
    # Try to initialize a video stream
    print("setting up video stream...")
    input_stream = cv2.VideoCapture(0)

    # Main loop
    while True:
        frame = get_frame(input_stream)

        processed_frame = process(frame)

        cv2.imshow('Video', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1. / FRAME_RATE)

    cv2.destroyAllWindows()


def main():
    start_processing_loop()


if __name__ == '__main__':
    main()
