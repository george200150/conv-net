import pickle
import random

from CNN.dataloader import MNISTDataLoader
from CNN.network import *
from CNN.utils import *

from tqdm import tqdm
import matplotlib.pyplot as plt


def fix_seeds():
    seed = 10
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    fix_seeds()

    num_classes = 10
    img_dim_x = 28
    img_dim_y = 28
    img_depth = 1
    train_sample_path = 'train-images-idx3-ubyte.gz'
    train_label_path = 'train-labels-idx1-ubyte.gz'
    test_sample_path = 't10k-images-idx3-ubyte.gz'
    test_label_path = 't10k-labels-idx1-ubyte.gz'
    save_path = "C:/Users/George/Downloads/repos/conv-net/output/output.txt"
    no_training_samples = 500  # 50000 max
    no_testing_samples = 100  # 10000 max
    batch_size = 32

    dataloader = MNISTDataLoader(img_dim_x, img_dim_y, train_sample_path, train_label_path, batch_size, no_training_samples)
    model = build_model(num_classes=num_classes, img_depth=img_depth, img_dim_x=img_dim_x, img_dim_y=img_dim_y,
                        save_path=save_path)
    cost = train(model, dataloader)

    data_file = open(save_path, 'rb')
    parameter_list, cost = pickle.load(data_file)
    data_file.close()
    model.load_model(parameter_list)

    # Plot cost 
    plt.plot(cost, 'r')
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend('Loss', loc='upper right')
    plt.show()

    # Get test data
    test_data = dataloader.load_data()

    X = test_data[:, 0:-1]
    X = X.reshape(len(test_data), 1, 28, 28)
    y = test_data[:, -1]

    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]

    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]
        outputs = model.full_forward(x)
        pred = np.argmax(outputs[-1])
        prob = np.max(outputs[-1])
        digit_count[int(y[i])] += 1
        if pred == y[i]:
            corr += 1
            digit_correct[pred] += 1

        t.set_description("Acc:%0.2f%%" % (float(corr / (i + 1)) * 100))

    print("Overall Accuracy: %.2f" % (float(corr / len(test_data) * 100)))
    x = np.arange(10)
    digit_recall = [safe_division(good, total) for good, total in zip(digit_correct, digit_count)]
    plt.xlabel('Digits')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    plt.bar(x, digit_recall)
    plt.show()
