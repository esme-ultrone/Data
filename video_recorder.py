import cv2
import argparse
import os


"""
Permet d'enregistrer une vidéo directement sous forme de 
suite de frames.

Usage : 
    video_recorder.py nomDuDossierDeReception
"""

parser = argparse.ArgumentParser()
parser.add_argument("outputDir", help="dossier de destination")
args = parser.parse_args()

outputDir = args.outputDir
print("enregistrement des frames dans " + outputDir)

if not os.path.exists(outputDir):
    os.mkdir(outputDir)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)


frameCount = 0

while True:
    ret, frame = cap.read()

    frameCount+=1

    cv2.imshow('recording', frame)

    #timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    cv2.imwrite("%s/%d.jpg" % (outputDir, frameCount), frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

