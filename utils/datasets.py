import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data_from_csv(train_csv="data/train.csv", test_size=0.1):
    df = pd.read_csv(train_csv)
    X = df.drop(columns=["label"]).values / 255.0
    y = df["label"].values

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, x_test, y_train, y_test
