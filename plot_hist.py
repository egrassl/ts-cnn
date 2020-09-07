import matplotlib.pyplot as plt
import pandas as pd
import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='path to the history file')
args = parser.parse_args()

# Loads history file from disk
history = pd.read_csv(args.path)
fig, axs = plt.subplots(1, 2)

# Plot accuracies
axs[0].set_title('Accuracy')
axs[0].plot(history['accuracy'])
axs[0].plot(history['val_accuracy'])
axs[0].legend(['train', 'validation'], loc='upper left')
axs[0].set(ylabel='accuracy', xlabel='epoch')

# Plot losses
axs[1].set_title('Loss')
axs[1].plot(history['loss'])
axs[1].plot(history['val_loss'])
axs[1].legend(['train', 'validation'], loc='upper left')
axs[1].set(ylabel='loss', xlabel='epoch')

plt.show()