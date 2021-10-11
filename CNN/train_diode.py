import pickle
import random

from CNN.dataloader import DIODEDataLoader
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

    num_classes = 2

    # img_dim_x = 512  # 1024 / 2
    # img_dim_y = 384  # 768 / 2
    img_dim_x = 64
    img_dim_y = 48  # 768 / 16

    img_depth = 3
    save_path = "C:/Users/George/Downloads/repos/conv-net/output/output_DIODE.txt"
    meta_fname = "C:/datasets/DIODE/diode_meta.json"
    data_root = "C:/datasets/DIODE/Depth/"
    num_images_train = 25458
    num_images_test = 771
    batch_size = 32

    dataloader = DIODEDataLoader(img_dim_x, img_dim_y, meta_fname, data_root, splits=['train'], scene_types=['indoors', 'outdoor'], num_images=num_images_train, batch_size=batch_size)
    model = build_model_DIODE(num_classes=num_classes, img_depth=img_depth, img_dim_x=img_dim_x, img_dim_y=img_dim_y, batch_size=batch_size, save_path=save_path)

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
    dataloader = DIODEDataLoader(img_dim_x, img_dim_y, meta_fname, data_root, splits=['val'], scene_types=['indoors', 'outdoor'], num_images=num_images_test)
    test_data = dataloader.load_data()

    X = test_data[:, 0:-1]
    X = X.reshape(len(test_data), img_depth, img_dim_x, img_dim_y)
    y = test_data[:, -1]

    corr = 0
    digit_count = [0 for i in range(num_classes)]
    digit_correct = [0 for i in range(num_classes)]

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
    x = np.arange(num_classes)
    digit_recall = [safe_division(good, total) for good, total in zip(digit_correct, digit_count)]
    plt.xlabel('Digits')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    plt.bar(x, digit_recall)
    plt.show()
