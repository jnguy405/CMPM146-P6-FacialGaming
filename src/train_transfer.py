from preprocess import get_transfer_datasets
from models.transfered_model import TransferedModel
from models.random_model import RandomModel
from config import image_size, transfer_categories
import matplotlib.pyplot as plt
import time
import numpy as np

# Your code should change these values based on your choice of dataset for the transfer task
# -------------
input_shape = (image_size[0], image_size[1], 3)
categories_count = len(transfer_categories) # defined in config.py
# -------------

models = {
    'transfered_model': TransferedModel,
    'random_model': RandomModel,
}

def plot_history_diff(initial_hist, transfered_hist):
    val_acc_initial = initial_hist.history['val_accuracy']
    val_acc_transfered = transfered_hist.history['val_accuracy']

    epochs_initial = range(1, len(val_acc_initial) + 1)
    epochs_transfered = range(1, len(val_acc_initial) + 1)
    assert epochs_initial == epochs_transfered, "The two models have been tried with different epochs"

    plt.figure(figsize = (24, 6))
    plt.plot(epochs_initial, val_acc_initial, 'b', label = 'Random Model Accuracy (without Transfer)')
    plt.plot(epochs_initial, val_acc_transfered, 'r', label = 'Transfered Model Accuracy (with Transfer)')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Far Transfer from Facial Recognition to Sushi vs. Sandwich')
    plt.show()

if __name__ == "__main__":
    # Your code should change the number of epochs
    epochs = 20
    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_transfer_datasets()
    histories = []
    
    for name, model_class in models.items():
        print('* Training {} for {} epochs'.format(name, epochs))
        model = model_class(input_shape, categories_count)
        model.print_summary()
        history = model.train_model(train_dataset, validation_dataset, epochs)
        histories.append(history)
        
        print('* Evaluating {}'.format(name))
        model.evaluate(test_dataset)
        print('* Confusion Matrix for {}'.format(name))
        print(model.get_confusion_matrix(test_dataset))
        
        # Save model and history
        model_name = '{}_{}_epochs_timestamp_{}'.format(name, epochs, int(time.time()))
        filename = 'results/{}.keras'.format(model_name)
        model.save_model(filename)
        np.save('results/{}.npy'.format(model_name), history.history)
        print('* Model saved as {}'.format(filename))
        print('* History saved as results/{}.npy'.format(model_name))
        print('-' * 60)
    
    assert len(histories) == 2, "The number of trained models is not equal to two"
    plot_history_diff(histories[1], histories[0])  # Random first, Transfer second