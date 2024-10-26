import numpy as np
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
from tkinter import filedialog
import base64


n_steps = 2
batch_size = 32
temperature = 1.0
train_size_fraction = 0.31
shakespeare_url = "data:text/plain;charset=utf-8;base64,aGVsbG8hIHRoaXMgaXMgYW4gYWkgdGVzdA=="


tokenizer = None
max_id = None
encoded = None
dataset_size = None
dataset = None
model = None


def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, "rb") as file:
            file_content = file.read()
            encoded_content = base64.b64encode(file_content).decode('utf-8')
            global shakespeare_url
            shakespeare_url = f"data:text/plain;charset=utf-8;base64,{encoded_content}"
            load_data()


def load_data():
    global tokenizer, max_id, encoded, dataset_size
    filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
    with open(filepath) as f:
        shakespeare_text = f.read()
    
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts([shakespeare_text])
    max_id = len(tokenizer.word_index)
    [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
    dataset_size = len(encoded)

    print(f"Encoded Size: {dataset_size}")
    create_model()
    update_params()


def update_params():
    global n_steps, batch_size, temperature, train_size_fraction
    n_steps = int(n_steps_slider.get())
    batch_size = max(1, int(batch_size_slider.get()))
    temperature = temperature_slider.get()
    train_size_fraction = train_size_slider.get()

    train_size = int(dataset_size * train_size_fraction)

    print(f"Updated Train Size: {train_size}")
    print(f"Updated Batch Size: {batch_size}")

    global dataset
    dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
    window_length = n_steps + 1
    dataset = dataset.window(window_length, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    dataset = dataset.shuffle(10000).batch(batch_size)
    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
    dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
    dataset = dataset.repeat().prefetch(1)

    num_batches = (train_size - n_steps) // batch_size
    print(f"Number of Batches: {num_batches}")


def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)

def next_char(text):
    X_new = preprocess([text])
    y_proba = model(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]


def create_model():
    global model
    model = keras.models.Sequential([
        keras.layers.Input(shape=(None, max_id)),
        keras.layers.GRU(128, return_sequences=True, dropout=0.2),
        keras.layers.GRU(128, return_sequences=True, dropout=0.2),
        keras.layers.GRU(128, return_sequences=True, dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax"))
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")


def complete_text(text, n_chars=50):
    for _ in range(n_chars):
        text += next_char(text)
    return text


def run_model():
    global dataset
    update_params()
    train_size = int(dataset_size * train_size_fraction)

    steps_per_epoch = max(1, (train_size - n_steps) // batch_size)

    print(f"Steps per Epoch: {steps_per_epoch}")

    if dataset:
        try:
            model.fit(dataset, epochs=1, steps_per_epoch=steps_per_epoch)
            result = complete_text("t", n_chars=50)  # Only pass text and n_chars
            result_label.config(text="Generated Text: " + result)
        except Exception as e:
            print(f"Error during model fitting: {e}")
    else:
        print("Error: Dataset is None. Cannot run the model.")


def generate_text():
    try:
        seed_text = root_text_entry.get()  # Get the initial text from the text entry box
        if not seed_text:
            seed_text = "The "  # Default text if entry is empty
        result = complete_text(seed_text, n_chars=100)  # Generate 100 characters of text
        result_label.config(text="Generated Text: " + result)
    except Exception as e:
        print(f"Error generating text: {e}")


root = tk.Tk()
root.title("Flight")
root.configure(bg="black")
root.geometry("600x500")


n_steps_slider = tk.Scale(root, from_=1, to=10, orient="horizontal", label="n_steps", bg="black", fg="white")
n_steps_slider.set(n_steps)
n_steps_slider.pack()

batch_size_slider = tk.Scale(root, from_=1, to=100, orient="horizontal", label="Batch Size", bg="black", fg="white")
batch_size_slider.set(batch_size)
batch_size_slider.pack()

temperature_slider = tk.Scale(root, from_=0.1, to=2.0, resolution=0.1, orient="horizontal", label="Temperature", bg="black", fg="white")
temperature_slider.set(temperature)
temperature_slider.pack()

train_size_slider = tk.Scale(root, from_=0.1, to=1.0, resolution=0.01, orient="horizontal", label="Train Size Fraction", bg="black", fg="white")
train_size_slider.set(train_size_fraction)
train_size_slider.pack()

upload_button = tk.Button(root, text="Upload File", command=upload_file, bg="white", fg="black")
upload_button.pack()

run_button = tk.Button(root, text="Run Model", command=run_model, bg="white", fg="black")
run_button.pack()

root_text_label = tk.Label(root, text="Root Text:", bg="black", fg="white")
root_text_label.pack()

root_text_entry = tk.Entry(root, width=40)
root_text_entry.pack()


generate_button = tk.Button(root, text="Generate Text", command=generate_text, bg="white", fg="black")
generate_button.pack()

result_label = tk.Label(root, text="Generated Text: ", bg="black", fg="white", wraplength=500)
result_label.pack()


load_data()


root.mainloop()
