from tensorflow.keras.models import load_model
from data import load_data, data_preprocessing
from evaluate import evaluate_model

x, y = load_data()

X_train_scaled, X_test_scaled, y_train, y_test = data_preprocessing(x, y)

print('data loading and preprocessing successfully')


ann = load_model("models/ann.keras")
dnn = load_model("models/dnn.keras")

print("models loaded successfully")

ann_results = evaluate_model(ann, X_test_scaled, y_test)

dnn_results = evaluate_model(dnn, X_test_scaled, y_test)


print("ANN Results")
print(ann_results)

print("\nDNN Results")
print(dnn_results)