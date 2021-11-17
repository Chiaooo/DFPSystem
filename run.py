import cv2.cv2 as cv2
import face_recognition
from tensorflow import keras

eye_model = keras.models.load_model('output/best_model_5.h5')


def eye_cropper(frame):
    facial_features_list = face_recognition.face_landmarks(frame)
    try:
        eye = facial_features_list[0]['left_eye']
    except:
        try:
            eye = facial_features_list[0]['right_eye']
        except:
            return

    x_max = max([coordinate[0] for coordinate in eye])
    x_min = min([coordinate[0] for coordinate in eye])
    y_max = max([coordinate[1] for coordinate in eye])
    y_min = min([coordinate[1] for coordinate in eye])

    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range > y_range:
        right = round(.5 * x_range) + x_max
        left = x_min - round(.5 * x_range)
        bottom = round((((right - left) - y_range)) / 2) + y_max
        top = y_min - round((((right - left) - y_range)) / 2)
    else:
        bottom = round(.5 * y_range) + y_max
        top = y_min - round(.5 * y_range)
        right = round((((bottom - top) - x_range)) / 2) + x_max
        left = x_min - round((((bottom - top) - x_range)) / 2)

    cropped = frame[top:(bottom + 1), left:(right + 1)]

    cropped = cv2.resize(cropped, (80, 80))
    image_for_prediction = cropped.reshape(-1, 80, 80, 3)

    return image_for_prediction


cap = cv2.VideoCapture(0)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(cap.get(cv2.CAP_PROP_FPS))

if not cap.isOpened():
    raise IOError('Cannot open webcam')

counter = 0

while True:
    ret, frame = cap.read()
    frame_count = 0
    if frame_count == 0:
        frame_count += 1
        pass
    else:
        count = 0
        continue

    image_for_prediction = eye_cropper(frame)
    try:
        image_for_prediction = image_for_prediction / 255.0
    except:
        continue

    prediction = eye_model.predict(image_for_prediction)

    if prediction < 0.5:
        counter = 0
        status = 'Open'

        cv2.rectangle(frame, (round(w / 2) - 110, 20), (round(w / 2) + 110, 80), (38, 38, 38), -1)

        cv2.putText(frame, status, (round(w / 2) - 80, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_4)
        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1 - 20, y1 + h1 - 20), (0, 0, 0), -1)
        cv2.putText(frame, 'Active', (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                    2)
    else:
        counter = counter + 1
        status = 'Closed'
        cv2.rectangle(frame, (round(w / 2) - 110, 20), (round(w / 2) + 110, 80), (38, 38, 38), -1)
        cv2.putText(frame, status, (round(w / 2) - 104, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_4)
        x1, y1, w1, h1 = 0, 0, 175, 75

        cv2.rectangle(frame, (x1, x1), (x1 + w1 - 20, y1 + h1 - 20), (0, 0, 0), -1)
        cv2.putText(frame, 'Active', (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                    2)

        # 若大于3，发出警告

        if counter > 2:
            x1, y1, w1, h1 = 400, 400, 400, 100
            cv2.rectangle(frame, (round(w / 2) - 160, round(h) - 200), (round(w / 2) + 160, round(h) - 120),
                          (0, 0, 255), -1)

            cv2.putText(frame, 'DRIVER SLEEPING', (round(w / 2) - 136, round(h) - 146), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2, cv2.LINE_4)

            cv2.imshow('Drowsiness Detection', frame)
            k = cv2.waitKey(1)
            counter = 1
            continue
    cv2.imshow('Drowsiness Detection', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
