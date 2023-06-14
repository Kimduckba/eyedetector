import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Graph:
    history: float

    def plot_loss_and_accuracy(self):
        fig, ax1 = plt.subplots()

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        epochs = range(1, len(loss) + 1)

        ax1.plot(epochs, loss, 'b-', label='Training loss')
        ax1.plot(epochs, val_loss, 'g-', label='Validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(epochs, acc, 'b+', label='Training accuracy')
        ax2.plot(epochs, val_acc, 'g+', label='Validation accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.title('Training and Validation Loss / Accuracy')
        plt.show()
