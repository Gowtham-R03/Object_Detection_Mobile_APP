import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time

########### PARAMETERS ##############

width = 640
height = 480
threshold = 0.2  # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 1
#####################################

#### CREATE CAMERA OBJECT
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

#### LOAD THE TRAINNED MODEL
model = load_model("my_new_model.h5")

#### PREPORCESSING FUlNCTION
def preProcessing(img):
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img = cv2.equalizeHist(img)
    img = img/255
    img = np.expand_dims(img, axis=0)
    return img

def model_predict(img):
    preds = model.predict(img)
    preds = np.argmax(preds, axis=1)

    ######################################### Firebase

    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase-SDK.json')
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    doc_ref = db.collection('Flower_info').document('Types')

    ##########################################
    if preds == 0:
        preds = "daisy"
        doc_ref.set({
            'name': preds,
            'level': 'normal'
        })
    elif preds == 1:
        preds = "dandelion"
        doc_ref.set({
            'name': preds,
            'level': 'normal'
        })
    elif preds == 2:
        preds = "roses"
        doc_ref.set({
            'name': preds,
            'level': 'normal'
        })
    elif preds == 3:
        preds = "sunflowers"
        doc_ref.set({
            'name': preds,
            'level': 'normal'
        })
    else:
        preds = "tulips"
        doc_ref.set({
            'name': preds,
            'level': 'normal'
        })


    return preds

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (224, 224))
    img = preProcessing(img)
    # cv2.imshow("Processsed Image", img)
    img = img.reshape(1, 224, 224, 3)
    #### PREDICT
    classIndex = str(model_predict(img))
    #print(classIndex)
    predictions = model.predict(img)
    #print(predictions)
    probVal = np.amax(predictions)
    # print(classIndex, probVal)


    if probVal > threshold:
        cv2.putText(imgOriginal, str(classIndex) + "   "+str(probVal),
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 1)

    cv2.imshow("Original Image", imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break