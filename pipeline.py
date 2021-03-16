from custom_model import CustomModel
from model import ImageCaptioningModel


def main():

    # model = ImageCaptioningModel("train_set/annotations.json")
    model = CustomModel("train_set/annotations.json")
    model.train(1, 1)


if __name__ == "__main__":
    main()
