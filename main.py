import albumentations as A
import cv2
import matplotlib.pyplot as plt
import os
import math
import xml.dom.minidom as minidom
import shutil
import time
import sys
import random
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

def visualize(image):
    plt.axis('off')
    plt.imshow(image)

def convertToJpg():
    path = "D:\\Users\\guilh\\PycharmProjects\\testingAugmentation\\personValidation\\train\\"

    for file in os.listdir(path):
        if not file.endswith(".jpg") and not file.endswith(".xml"):
            img = Image.open(path + file)

            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            file_name, file_ext = os.path.splitext(file)
            img.save('D:\\Users\\guilh\\PycharmProjects\\testingAugmentation\\personValidation\\png\\' + file_name + '.jpg'.format(file_name))
        time.sleep(0.3)


def splitDataset(trainPercent = 0.8, validatePercent = 0.2):
    folder_dir = "C:\\Users\\Guilherme\\OneDrive\\Área de Trabalho\\Combined\\train\\"
    validate_split = "C:\\Users\\Guilherme\\OneDrive\\Área de Trabalho\\Combined\\validate\\"

    files = len(os.listdir(folder_dir))
    images = files/2
    validate = math.floor(images * validatePercent)

    file_name_array = []

    i = 0
    #coloco em um array todos os arquivos a irem para validate
    while (i < validate):
        randomized_file = random.randint(0, files - 1)
        image_name = os.listdir(folder_dir)[randomized_file]

        if (image_name.endswith(".jpg")):
            if ("Augmented" not in image_name):
                if (image_name not in file_name_array):
                    file_name_array.append(image_name)
                    i = i + 1
    i = 0
    while (i < len(file_name_array)):
        fileName = file_name_array[i].split('.')[0]

        Path(folder_dir + fileName + '.jpg').rename(validate_split + fileName + '.jpg')
        Path(folder_dir + fileName + '.xml').rename(validate_split + fileName + '.xml')

        time.sleep(0.3)
        i = i + 1

def transformm():
    folder_dir = "C:\\Users\\Guilherme\\Dev\\CNN-HumanDetection\\newTruckValidation\\train\\"

    for file in os.listdir(folder_dir):
        if (file.endswith(".jpg")):
            image = cv2.imread('newTruckValidation\\train\\' + file)

            xml = minidom.parse('newTruckValidation\\train\\' + file.split('.')[0] + '.xml')
            bndbox = xml.getElementsByTagName('bndbox')
            classes = xml.getElementsByTagName('name')

            bboxes = []

            for box in bndbox:
                boxAux = [int(box.childNodes[1].firstChild.data),
                 int(box.childNodes[3].firstChild.data),
                 int(box.childNodes[5].firstChild.data),
                 int(box.childNodes[7].firstChild.data),
                 classes[0].firstChild.data]

                bboxes.append(boxAux)

            transform = A.Compose(
                [A.HorizontalFlip(),
                #A.RandomRotate90(),
                A.GaussNoise(),
                A.HueSaturationValue(),
                A.GridDistortion(),], bbox_params=A.BboxParams(format='pascal_voc'))


            augmented_image = transform(image=image, bboxes=bboxes)
            transformed_image = augmented_image['image']
            transformed_bboxes = augmented_image['bboxes']

            #before augmentation image
            #for box in bboxes:
            #    cv2.rectangle(image, (math.ceil(box[0]), math.ceil(box[1])), (math.ceil(box[2]), math.ceil(box[3])), (0, 255, 0), 3)

            #after augmentation image
            cv2.imwrite('newTruckValidation\\trainAugment\\' + file.split('.')[0] + 'Augmented3.jpg', transformed_image)
            source = 'newTruckValidation\\train\\' + file.split('.')[0] + '.xml'
            target = 'newTruckValidation\\trainAugment\\' + file.split('.')[0] + 'Augmented3.xml'
            shutil.copy(source, target)

            time.sleep(0.3)

            newXml = minidom.parse(target)
            newFilename = newXml.getElementsByTagName("filename")
            newFilename[0].firstChild.replaceWholeText(file.split('.')[0] + 'Augmented3.jpg')

            newBndBoxes = newXml.getElementsByTagName('bndbox')
            print(file)
            i = 0
            try:
                for box in newBndBoxes:
                    box.childNodes[1].firstChild.replaceWholeText(str(math.ceil(transformed_bboxes[i][0])))
                    box.childNodes[5].firstChild.replaceWholeText(str(math.ceil(transformed_bboxes[i][2])))
                    box.childNodes[3].firstChild.replaceWholeText(str(math.ceil(transformed_bboxes[i][1])))
                    box.childNodes[7].firstChild.replaceWholeText(str(math.ceil(transformed_bboxes[i][3])))
                    i= i + 1
            except:
                print (sys.exc_info()[0])
                print (" imagem: ", file)

            f = open(target, 'w', encoding="utf-8")
            newXml.writexml(f)
            f.close()

            #for box in transformed_bboxes:
            #    cv2.rectangle(transformed_image, (math.ceil(box[0]), math.ceil(box[1])), (math.ceil(box[2]), math.ceil(box[3])), (0, 255, 0), 3)

            time.sleep(0.3)

            '''fig = plt.figure(figsize=(10, 10))
            fig.add_subplot(2, 2, 1)

            plt.imshow(image)
            plt.axis('off')
            plt.title("Original")

            fig.add_subplot(2, 2, 2)
            plt.imshow(transformed_image)
            plt.axis('off')
            plt.title("Augmented")

            plt.show()'''

def cropTimeSelecionadas():
    selecionadas = "D:\\Users\\guilh\\PycharmProjects\\testingAugmentation\\Selecionadas\\"
    selecionadasCrop = "D:\\Users\\guilh\\PycharmProjects\\testingAugmentation\\Selecionadas Crop\\"

    i = 1
    for file in os.listdir(selecionadas):
        image = cv2.imread('Selecionadas\\' + file)
        height, width, channels = image.shape

        crop_img = image[0:height - 40, 0 + 54:0 + width]

        cv2.imwrite(selecionadasCrop + 'selecionadasCrop' + i.__str__() + '.jpg', crop_img)

        i = i + 1

def cropTimeSelecionadasCaminhao():
    selecionadasCaminhao = "D:\\Users\\guilh\\PycharmProjects\\testingAugmentation\\Selecionadas Caminhao\\"
    selecionadasCaminhaoCrop = "D:\\Users\\guilh\\PycharmProjects\\testingAugmentation\\Selecionadas Caminhao Crop\\"

    i = 1
    for file in os.listdir(selecionadasCaminhao):
        image = cv2.imread('Selecionadas Caminhao\\' + file)
        height, width, channels = image.shape

        crop_img = image[0:height - 40, 0 + 54:0 + width]

        cv2.imwrite(selecionadasCaminhaoCrop + 'selecionadasCaminhaoCrop' + i.__str__() + '.jpg', crop_img)

        i = i + 1

def cropTimeSelecionadasNoturnas():
    selecionadasNoturnas = "D:\\Users\\guilh\\PycharmProjects\\testingAugmentation\\Selecionadas Noturnas\\"
    selecionadasNoturnasCrop = "D:\\Users\\guilh\\PycharmProjects\\testingAugmentation\\Selecionadas Noturnas Crop\\"

    i = 1
    for file in os.listdir(selecionadasNoturnas):
        image = cv2.imread('Selecionadas Noturnas\\' + file)
        height, width, channels = image.shape

        crop_img = image[0:height - 40, 0 + 54:0 + width]

        cv2.imwrite(selecionadasNoturnasCrop + 'selecionadasNoturnasCrop' + i.__str__() + '.jpg', crop_img)

        i = i + 1

def cropTimeSelecionadasDiaA():
    selecionadasDiaA = "D:\\Users\\guilh\\PycharmProjects\\testingAugmentation\\Selecionadas Dia\\500a\\"
    selecionadasDiaACrop = "D:\\Users\\guilh\\PycharmProjects\\testingAugmentation\\Selecionadas Dia Crop\\500aCrop\\"

    i = 1
    for file in os.listdir(selecionadasDiaA):
        image = cv2.imread('Selecionadas Dia\\500a\\' + file)
        height, width, channels = image.shape

        crop_img = image[0:height - 40, 0 + 54:0 + width]

        cv2.imwrite(selecionadasDiaACrop + 'selecionadasDiaACrop' + i.__str__() + '.jpg', crop_img)

        i = i + 1

def cropTimeSelecionadasDiaB():
    selecionadasDiaB = "D:\\Users\\guilh\\PycharmProjects\\testingAugmentation\\Selecionadas Dia\\500b\\"
    selecionadasDiaBCrop = "D:\\Users\\guilh\\PycharmProjects\\testingAugmentation\\Selecionadas Dia Crop\\500bCrop\\"

    i = 1
    for file in os.listdir(selecionadasDiaB):
        image = cv2.imread('Selecionadas Dia\\500b\\' + file)
        height, width, channels = image.shape

        crop_img = image[0:height - 40, 0 + 54:0 + width]

        cv2.imwrite(selecionadasDiaBCrop + 'selecionadasDiaBCrop' + i.__str__() + '.jpg', crop_img)

        i = i + 1

def deleteFiles():
    folder_dir = "D:\\Users\\guilh\\PycharmProjects\\train"

    for file in os.listdir(folder_dir):
        if (file.__contains__("Augmented")):
            os.remove(folder_dir + "\\" + file)

def scaleUpImages():
    folder_path = 'C:\\Users\\Guilherme\\Dev\\CNN-HumanDetection\\truck dataset'

    kernel = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])

    k3 = np.ones((5, 5), np.float32) / 25

    for file in os.listdir(folder_path):
        img = cv2.imread('C:\\Users\\Guilherme\\Dev\\CNN-HumanDetection\\truck dataset\\' + file)
        img = cv2.pyrUp(img)
        img = cv2.pyrUp(img)

        cv2.imwrite("C:\\Users\\Guilherme\\Dev\\CNN-HumanDetection\\truck dataset resized\\" + file, img)

        image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

        cv2.imwrite("C:\\Users\\Guilherme\\Dev\\CNN-HumanDetection\\truck dataset resized\\2" + file, image_sharp)

def brightenUp():
    folder_path = 'C:\\Users\\Guilherme\\Dev\\CNN-HumanDetection\\2023-05-23\\'

    for file in os.listdir(folder_path):
        im = Image.open(folder_path + file)
        enhancer = ImageEnhance.Brightness(im)

        factor = 1.5  # brightens the image
        im_output = enhancer.enhance(factor)
        im_output.save(folder_path + file.split('.')[0] + 'brightened.jpg')


def floorBoundingBoxes():
    folder_path = 'C:\\Users\\Guilherme\\Dev\\CNN-HumanDetection\\personValidationNoturna\\validate\\'

    for file in os.listdir(folder_path):
        if ("xml" in file):
            xml = minidom.parse(folder_path + file)

            BndBoxes = xml.getElementsByTagName('bndbox')
            print(file)
            i = 0
            try:
                Filename = xml.getElementsByTagName("filename")
                Filename[0].firstChild.replaceWholeText(file.split('.xml')[0] + '.jpg')

                for box in BndBoxes:
                    box.childNodes[1].firstChild.replaceWholeText(str(int(float(box.childNodes[1].firstChild.data))))
                    box.childNodes[3].firstChild.replaceWholeText(str(int(float(box.childNodes[3].firstChild.data))))
                    box.childNodes[5].firstChild.replaceWholeText(str(int(float(box.childNodes[5].firstChild.data))))
                    box.childNodes[7].firstChild.replaceWholeText(str(int(float(box.childNodes[7].firstChild.data))))
                    i = i + 1
            except:
                print(sys.exc_info()[0])
                print(" imagem: ", file)

            f = open(folder_path + file, 'w', encoding="utf-8")
            xml.writexml(f)
            f.close()


def main():
    #cropTimeSelecionadas()
    #cropTimeSelecionadasCaminhao()
    #cropTimeSelecionadasNoturnas()
    #cropTimeSelecionadasDiaA()
    #cropTimeSelecionadasDiaB()

    #convertToJpg()
    #deleteFiles()
    splitDataset()
    #transformm()
    #scaleUpImages()
    #brightenUp()
    #floorBoundingBoxes()

    #image = cv2.imread('sample.jpg')
    #crop_img = image[0:0 + 560, 0 + 54:0 + 800]
    #cv2.imwrite('result.jpg', crop_img)

if __name__ == '__main__':
    main()

