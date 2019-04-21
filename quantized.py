import tensorflow as tf

# 1.12	    tf.contrib.lite.TFLiteConverter
# 1.9-1.11	tf.contrib.lite.TocoConverter
# 1.7-1.8	tf.contrib.lite.toco_convert


def quantized(saved_model_dir, export_model_name="./models/faceboxes_quant.tflite"):
    # converter = tf.contrib.lite.Interpreter()
    converter = tf.contrib.lite.TocoConverter.from_saved_model(saved_model_dir)
    # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.contrib.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()
    open(export_model_name, "wb").write(tflite_quant_model)


if __name__ == "__main__":
    quantized("./export/run00/1555830494/")
