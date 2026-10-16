
import tensorflow as tf
import numpy as np

print("Project: TFLite Micro Inference")
print("Author: SELVASH")

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="sgd", loss="mse")

x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

model.fit(x, y, epochs=100, verbose=0)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(
    model_content=tflite_model
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(
    input_details[0]["index"],
    np.array([[6.0]], dtype=np.float32)
)

interpreter.invoke()

result = interpreter.get_tensor(
    output_details[0]["index"]
)

print("Input:", 6)
print("Predicted output:", result[0][0])
