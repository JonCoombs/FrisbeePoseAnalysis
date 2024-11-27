from ultralytics import YOLO
from PIL import Image
from os import listdir

#Begin modifiable global variables
dataset = "datasets/V4"   #Change path to match current dataset

training = True        #Are you training a new model?
numEpochs = 100         #Number of "epochs", or individual instances of training on a subset of the dataset. Generally from 100 to 500 for this project
batchSize = 16          #Number of images to train on per epoch. Generally between 12 and 20 for this project; my GPU ran out of memory while training on 20
patienceVal = 100       #Number of epochs w/o improvement before stopping training early. 100 is a good baseline, and you should change it based on numEpochs and your computer

visualizeModel = False  #Are you visualizing an existing model?
modelNum = 0            #0 if visualizing the most recent model, otherwise visualizes the chosen model number.
                        #This variable does nothing if visualizeModel is False.
#End modifiable global variables


def train():
    model = YOLO("yolov8n-pose.pt")

    #Generate a model
    model.train(
        data = dataset + "/data.yaml",
        imgsz = 1280,   #Default, can change if the images are pre-processed differently
        epochs=numEpochs,
        batch=batchSize,
        device="cuda",  #Switch to "cpu" if you don"t have both a GPU and some form of CUDA installed. CPU takes much longer per epoch, and is not recommended
        patience=patienceVal
    )

def visualize(modelNum):
    #Select the best training weights from the correct model. If modelNum = 1, select "train" instead of "train1"
    model = YOLO("runs/pose/train" + ("" if modelNum == 1 else str(modelNum)) + "/weights/best.pt")

    #Run the model on all images in the most recent test folder
    test_folder = "./test/images"

    testing_count = 1

    for item in listdir(test_folder):

        #Ignore other files. Currently, all images will always be .jpg files
        if not item.endswith(".jpg"):
            print("Not a test image.")
            continue
    
        annotated = model(test_folder + "/" + item)

        #Use Pillow to plot annotations to images
        r = annotated[0]
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

        #Save the annotated images to a folder for later review
        im.save("annotations/temp" + str(testing_count) + ".jpg")

        testing_count += 1

#Prevent other branches from running training (when used with GPU)
if __name__ == "__main__" and training:
    train()

#Test out the trained model
if visualizeModel:
    #Select the most recent model if modelNum is 0
    if modelNum == 0:
        for item in listdir("runs/pose"):
            trainNum = item[5:]
            trainNum = (1 if trainNum == "" else int(trainNum))
            if trainNum > modelNum:
                modelNum = trainNum
    
    #Call visualize on the correct model
    visualize(modelNum)
    
# model = YOLO("runs/pose/train25/weights/best.pt")

# test_folder = "datasets/Frisbee Form Analysis.v4i.yolov8/test/images"

# testimg_count = 1

# for item in listdir(test_folder):
#     print(item)

#     if not item.endswith(".jpg"):
#         print("Not jpg")
#         continue
    
#     annotated = model(test_folder + "/" + item)

#     r = annotated[0]
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.save("annotations/temp" + str(testimg_count) + ".jpg")

#     testimg_count += 1