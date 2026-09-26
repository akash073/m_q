
import os

# CNN DNN interface
import classical_cnn_dnn
import quantum_dnn_cnn

# YOLO interface
import classical_yolo
import quantum_yolo


# NUM_TEST_SAMPLES

os.environ["NUM_TEST_SAMPLES"] = "3"
os.environ["TRAIN_EPOCHS"] = "2"


classical_cnn_dnn.main()
quantum_dnn_cnn.main()



classical_yolo.main()
quantum_yolo.main()

