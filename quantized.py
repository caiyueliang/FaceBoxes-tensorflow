import tensorflow as tf


def quantized(saved_model_dir):
    converter = tf.contrib.lite.TocoConverter()
    # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    # tflite_quant_model = converter.convert()


if __name__ == "__main__":
    quantized("./export/run00/1555830494/")
