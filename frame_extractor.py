import cv2
import os
import labeled_intervals
from shutil import copyfile


def extractFrames(videoPath):

    """
    Permet d'enregistrer toutes les frames d'une vidéo en format jpeg
    """

    videoName = os.path.basename(os.path.normpath(videoPath))
    outputDir = 'frames_' + videoName

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    cap = cv2.VideoCapture(videoPath)
    print(cap.isOpened())
    while(cap.isOpened()):
        frameExists, frame = cap.read()
        if frameExists:
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            cv2.imwrite("%s/%d.jpg" % (outputDir, timestamp), frame)
        else:
            break

    cap.release()


def copyFrame(filename, outputDir, framesDir):
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    copyfile(os.path.join(framesDir, filename),
             os.path.join(outputDir, filename))


def groupFrames(framesDir, labeledIntervals, copy=False):

    """
    Permet de grouper les frames selon leur label

    framesDir: dossier contenant les frames
    copy: booléen pour copier ou non les frames dans des dossiers correspondant aux labels

    format du dictionary labeledIntervals:

        labeledIntervals = {
            "goRight": [[0, 6375], [8750, 12750]], 
            "takeOff": [[12791, 13500]]
        }
        Ici les frames de [0 à 6375] et [8750 à 12750] sont des frames goRight,
        et celles de [12791 à 13500] son des frames takeOff. Tout le reste
        sera assigné dans default.

    La fonction renvoie un tableau des frames labelisées:
    labeledFrames = [[1, default], [2, default], [3, takeoff], ...]
    """

    labeledFrames = []
    files = os.listdir(framesDir)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for filename in files:
        timestamp, file_extension = os.path.splitext(filename)
        timestamp = int(timestamp)

        labeled = False

        for label, intervals in labeledIntervals.items():
            for interval in intervals:
                if interval[0] <= timestamp <= interval[1]:

                    labeled = True

                    labeledFrames.append([timestamp, label])

                    if copy:
                        copyFrame(filename, framesDir + "_%s" % label, framesDir)

                    break

        if not labeled:
            labeledFrames.append([timestamp, "default"])

            if copy:
                copyFrame(filename, framesDir + "_default", framesDir)

    return labeledFrames


#exemple pour labeliser les frames situées dans un dossier session_1,
#avec les intervalles situés dans labeled_intervals.py:
groupFrames("session_1", labeled_intervals.session_1, True)

#exemple pour extraire les frames d'une video située dans /home/uldrone/video.mp4
#extractFrames("/home/david/uldrone/output.mp4")
    
