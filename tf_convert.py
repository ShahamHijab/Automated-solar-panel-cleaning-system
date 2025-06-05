import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('logistic_model_savedmodel')
tflite_model = converter.convert()

with open('logistic_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved as 'logistic_model.tflite'")
