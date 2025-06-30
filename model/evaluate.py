from keras.models import load_model
from utils.datasets import load_data_from_csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def evaluate_model(model_path="saved_model/digit_model.h5"):
    _, x_test, _, y_test = load_data_from_csv()
    model = load_model(model_path)
    loss, acc = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    return loss, acc

if __name__ == "__main__":
    evaluate_model()
