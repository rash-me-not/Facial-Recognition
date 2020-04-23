import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from knn_sklearn import KNNSklearn


class Main:
    def __init__(self, file):
        self.file = file

    def generate_dataset(self):
        """Read the source file and return a dictionary with x_train, y_train, x_val, y_yal, x_test and y_test"""

        df = pd.read_csv(file, header=0)
        train_samples = df[df.Usage == "Training"]
        val_samples = df[df.Usage == "PrivateTest"]
        test_samples = df[df.Usage == "PublicTest"]

        samples_dict = {"train": train_samples, "val": val_samples, "test": test_samples}
        data_dict = {}
        for type, samples in samples_dict.items():
            data_dict["x_" + type], data_dict["y_" + type] = self.preprocess(samples)
        self.visualize(data_dict["x_train"], data_dict["y_train"])
        return data_dict

    def visualize(self, x, y):
        """Visualize some examples from the dataset"""

        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        # Take a look at the first 25 images of the dataset
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            image = x[i].reshape(48, 48)
            plt.imshow(image, cmap='gray')
            plt.xlabel(emotion_labels[y[i]])
        plt.show()

    def preprocess(self, samples):
        """format the features and labels as necessary for processing"""

        y = []
        x = []
        for idx, image in samples.iterrows():
            y.append(int(image.emotion))
            image_pixel = np.asarray([float(pix) for pix in image.pixels.split(" ")])
            x.append(image_pixel)
        return np.asarray(x), np.asarray(y)


if __name__ == "__main__":

    file = "../../fer2013/fer2013.csv"
    main = Main(file)
    data = main.generate_dataset()
    knn = KNNSklearn(data)

    k_list = [1,2]
    knn.train_and_validate(k_list, "Manhattan")