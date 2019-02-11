from DeepVOG_model import load_DeepVOG
import skimage.io as ski
import numpy as np


def test_if_model_work():
    model = load_DeepVOG()
    img = np.zeros((1, 240, 320, 3))
    img[:,:,:,:] = (ski.imread("test_image.png")/255).reshape(1, 240, 320, 1)
    prediction = model.predict(img)
    ski.imsave("test_prediction.png", prediction[0,:,:,1])

if __name__ == "__main__":
    # If model works, the "test_prediction.png" should show the segmented area of pupil from "test_image.png"
    test_if_model_work()

