import tensorflow as tf
from time import time

# Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
from tensorflow.python.compiler.mlcompute import mlcompute

# Select device.
device = "cpu"  # Available options are 'cpu', 'gpu', and 'any'.
if device != "cpu":
    # for GPU device eager execution must be disabled
    tf.compat.v1.disable_eager_execution()
mlcompute.set_mlc_device(device_name=device)

# timer utility
def timer(start: float, end: float) -> str:
    """
    Timer function. Compute execution time from strart to end (end - start).
    :param start: start time
    :param end: end time
    :return: end - start
    """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

# params
num_words = 50000
embedding_size = 100
maxlen = 250

# load the dataset
train_data, test_data = tf.keras.datasets.imdb.load_data(
    path="imdb.npz",
    num_words=num_words,
    skip_top=0,
    maxlen=maxlen,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
)
x_train, y_train = train_data
x_test, y_test = test_data

# quickly pad the datasets
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Define the model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Embedding(num_words, embedding_size, input_length=maxlen),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train
start = time()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)
end = time()
print("Training time:", timer(start, end))

# Evaluation just for fun (bugged right now)
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
