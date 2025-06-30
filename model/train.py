from utils.datasets import load_data_from_csv
from model.model import build_model
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def analyse_fit(
    train_acc_hist, val_acc_hist,
    train_loss_hist, val_loss_hist,
    cfg_name="run"
):
    train_acc_hist = np.asarray(train_acc_hist)
    val_acc_hist = np.asarray(val_acc_hist)
    train_loss_hist = np.asarray(train_loss_hist)
    val_loss_hist = np.asarray(val_loss_hist)

    gap_acc = train_acc_hist - val_acc_hist

    verdict = "     No strong signs of overfitting or underfitting."
    if train_acc_hist[-1] > 0.85 and gap_acc[-1] > 0.10:
        verdict = (
            f"      Overfitting detected ({cfg_name}): "
            f"train acc {train_acc_hist[-1]:.2%} "
            f"vs val acc {val_acc_hist[-1]:.2%}."
        )
    elif train_acc_hist[-1] < 0.60 and val_acc_hist[-1] < 0.60:
        verdict = (
            f"      Underfitting detected ({cfg_name}): "
            f"train acc {train_acc_hist[-1]:.2%}, "
            f"val acc {val_acc_hist[-1]:.2%}."
        )

    return verdict

def cross_validate_model(save_final_model_to="saved_model/digit_model.h5", folds=5, epochs=20):
    x_train, x_test, y_train, y_test = load_data_from_csv()
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
        print(f"\n Fold {fold + 1}/{folds}")
        x_tr, x_val = x_train[train_idx], x_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = build_model()

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        history = model.fit(
            x_tr, y_tr,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )

        val_acc = history.history['val_accuracy'][-1]
        fold_accuracies.append(val_acc)
        print(f" Fold {fold + 1} Validation Accuracy: {val_acc:.4f}")
        verdict = analyse_fit(
            history.history['accuracy'],
            history.history['val_accuracy'],
            history.history['loss'],
            history.history['val_loss'],
            cfg_name=f"fold_{fold+1}"
        )
        print(verdict)

        

    print("\n Cross-validation results:")
    print("Accuracies per fold:", fold_accuracies)
    print(f"Mean accuracy: {np.mean(fold_accuracies):.4f}")
    print()
    print("\n Training on full training set...")
    final_model = build_model()

    early_stop_final = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    final_model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[early_stop_final]
    )


    os.makedirs(os.path.dirname(save_final_model_to), exist_ok=True)
    final_model.save(save_final_model_to)
    print(f" Final model saved to {save_final_model_to}")

if __name__ == "__main__":
    cross_validate_model()
