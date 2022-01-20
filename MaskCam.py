import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model('Saved_Model/my_model')
class_names = ['no', 'yes']
face = cv2.CascadeClassifier("head_detector.xml")

webcam = cv2.VideoCapture(0)
while True:
    ret, frame = webcam.read()
    if frame.any()!=None:
        faces = face.detectMultiScale(
            frame, scaleFactor=1.1,
            minNeighbors=5,
            minSize=(150, 150)
        )
        for(x, y, w, h) in faces:
            print(w)
            photo=frame[y:y+h,x:x+w]
            #I think we put the line here
            batch_size = 128
            img_height = 180
            img_width = 180
            img = cv2.resize(photo,(180,180))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
            cv2.putText(frame, "{} {:.2f}".format(class_names[np.argmax(score)], 100 * np.max(score)), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Capturing", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #key = cv2.waitKey(1)
webcam.release()
cv2.destroyAllWindows()
