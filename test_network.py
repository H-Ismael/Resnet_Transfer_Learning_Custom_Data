# USAGE
# python test_network.py --model snt.model --image images/examples/amb.png
#(tenv) D:\******trashIR\Image-Classification-Transfer-Learning-master\Image-Classification-Transfer-Learning-master>
#python test_network.py --model M1 --image D:\*****\trashIR\Image-Classification-Transfer-Learning-master\Image-Classification-Transfer-Learning-master\original_dataset\test\IMG_98.jpg
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig = image.copy()

image = cv2.resize(image, (256, 256))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


print("[INFO] loading network...")
model = load_model(args["model"])

# estmation de proba
(ambulance, notambulance) = model.predict(image)[0]

# labeling nd estimating
label = "ambulance" if ambulance > notambulance else "Not ambulance"
proba = ambulance if ambulance > notambulance else notambulance
label = "{}: {:.2f}%".format(label, proba * 100)

# draw comment nd pred
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

cv2.imshow("Output", output)
cv2.waitKey(0)
print('Done')
