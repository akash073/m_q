import os


# NUM_TEST_SAMPLES

# os.environ["NUM_TEST_SAMPLES"] = "10000"
# os.environ["TRAIN_EPOCHS"] = "10"

# CNN DNN interface
import classical_cnn_dnn
import quantum_dnn_cnn

# YOLO interface
import classical_yolo
import quantum_yolo


classical_cnn_dnn.main()
classical_yolo.main()

quantum_dnn_cnn.main()
quantum_yolo.main()

