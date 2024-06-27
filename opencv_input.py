import cv2
# import face_recognition
import pickle
import numpy as np
from keras.models import load_model
import dlib
import itertools
from scipy.spatial import distance

cnn_model = load_model("best-vgg-19-model.h5")

# knn_model = pickle.load(open("emotion_model_3", 'rb'))

facec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def get_landmarks(image):
    #:type image : cv2 object
    #:rtype landmarks : list of tuples where each tuple represents
    #                  the x and y co-ordinates of facial keypoints

    # Bounding Box co-ordinates around the face(Training data is 48*48(cropped faces))
    rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]

    # Read Image using OpenCV
    # image = cv2.imread(image_path)
    # Detect the Faces within the image
    landmarks = [(p.x, p.y) for p in predictor(image, rects[0]).parts()]
    return image, landmarks


def landmarks_edist(face_landmarks):
    e_dist = []
    # FILL ME IN!
    # Use this to get the distance between two points:
    #               distance.euclidean(face_landmarks[i],face_landmarks[j])
    for i, j in itertools.combinations(range(68), 2):
        e_dist.append(distance.euclidean(face_landmarks[i], face_landmarks[j]))

    return e_dist


# def preprocess_frame(image):
#     image = cv2.GaussianBlur(image, (5, 5), 0)
#
#     _, face_landmarks = get_landmarks(image)
#     return landmarks_edist(face_landmarks)  # Using our feature function!


cam = cv2.VideoCapture(0)

process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = cam.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    # face_locations = face_recognition.face_locations(rgb_small_frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)

    # Display the results
    for (x, y, w, h) in faces:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        '''
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        '''
        # Draw a box around the face
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)

        font = cv2.FONT_HERSHEY_DUPLEX
        print(gray.shape,x,y,w,h)
        f = gray[y:y+h,x:x+w]
        print(f.shape)
        frame2 = cv2.resize(f, (48, 48))
        frame2 = frame2.reshape(1, 48, 48, 1)
        frame2 = np.concatenate((frame2, frame2, frame2), axis=3)
        # frame2 = np.reshape(frame2,(34,67))
        # frame2.reshape((34, 67))
        prediction = cnn_model.predict(frame2)
        label_num = np.argmax(prediction, axis=1)

        if label_num == 0:
            label = "Angry"
        elif label_num == 1:
            label = "Happy"
        elif label_num == 2:
            label = "Sad"
        elif label_num == 3:
            label = "Surprise"
        elif label_num == 4:
            label = "Neutral"
        # cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.putText(frame, label, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    k = cv2.waitKey(1)

    # Hit 'ESC' on the keyboard to quit!
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break