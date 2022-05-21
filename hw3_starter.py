"""
Starter code for CS 281 HW3 Sections 2 and 3.
"""

import numpy as np
import opacus
from torch import nn, optim
import torch.utils.data
import tqdm

# Don't modify these; especially the seed!
np.random.seed(42)
n = 10000
D = 2000
C = 6
sensitivity = 2. * C / n

mean = np.zeros((1, D)) + np.random.randn(1, D)
std = np.ones((1, D))
mean[:, 10:] = 0.
std[:, 10:] = 0.
X = mean + np.random.randn(n, D) * std


def sample_data(d: int) -> np.ndarray:
    """Sample private examples.

    Given the global seed, this function should be fully deterministic to facilitate reproducibility.
    Do not modify.

    Args:
        d: Dimensionality of the data.

    Returns:
        A np.ndarray with shape (n, d).
    """
    return X[:, :d]


def gaussian_mechanism(
    private_data: np.array, epsilon: float, delta: float
) -> np.array:
    # TODO:
    raise NotImplementedError("Your implementation goes here.")


def section_2_1():
    """This question is about the basics."""
    epsilon, delta = 0.2, 1e-5
    x = sample_data(d=10)
    # TODO:
    raise NotImplementedError("Your implementation goes here.")


def section_2_2():
    """This question is about dimension scaling."""
    epsilon, delta = 0.2, 1e-5
    reps = 10  # Number of times to run to estimate expected squared error.
    errors = []
    dims = (10, 20, 50, 100, 200, 500, 1000)
    for d in dims:
        x = sample_data(d=d)
        # TODO:
        raise NotImplementedError("Your implementation goes here.")


def _preprocess(batch_size=2000):
    # auto (1) and cat (3) have label 0
    # plane (0) and bird (2) have label 1
    # auto (1) and plane (0) are group 0, cat (3) and bird (2) are group 1
    TRAIN_PATH = "./simclr_r50_1x_sk0_train.npz"
    TEST_PATH = "./simclr_r50_1x_sk0_test.npz"

    train_dump = np.load(TRAIN_PATH)
    test_dump = np.load(TEST_PATH)

    loaders = []
    for dump, (maj_size, min_size) in zip(
        (train_dump, test_dump),
        ((10000, 500,), (2000, 2000)),
    ):
        features, labels = dump["features"], dump["labels"]
        xmaj, ymaj, xmin, ymin = [], [], [], []
        for feature, label in tqdm.tqdm(zip(features, labels)):
            if label == 0:  # plane, group 0, label 1.
                xmaj.append(feature)
                ymaj.append(1)
            if label == 1:  # auto, group 0, label 0.
                xmaj.append(feature)
                ymaj.append(0)
            elif label == 2:
                xmin.append(feature)
                ymin.append(1)
            elif label == 3:
                xmin.append(feature)
                ymin.append(0)

        splits = []
        for xsplit, ysplit, size in ((xmaj, ymaj, maj_size), (xmin, ymin, min_size)):
            xsplit = np.stack(xsplit)
            ysplit = np.array(ysplit)

            indices = np.random.permutation(len(xsplit))[:size]
            xsplit = xsplit[indices]
            ysplit = ysplit[indices]
            splits.append((xsplit, ysplit))

        x = np.concatenate([splits[0][0], splits[1][0]])
        y = np.concatenate([splits[0][1], splits[1][1]])
        g = np.array([0] * maj_size + [1] * min_size)

        x, y, g = torch.tensor(x), torch.tensor(y), torch.tensor(g)

        dataset = torch.utils.data.TensorDataset(x, y, g)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        loaders.append(loader)

    return loaders


# Do not change the default kwargs except epsilon for your experiments!
def section_3(
    in_size=2048,
    out_size=2,
    batch_size=500,
    delta=1e-5,
    epochs=50,
    lr=1e-1,
    momentum=0.9,
    epsilon=0.5,  # This you may change for section 3.2.
):
    train_loader, test_loader = _preprocess(batch_size=batch_size)

    # Example of what's returned of the loader.
    # `features` is the input tensor (processed by SimCLR)
    # `labels` is the label tensor (each either 0 for land or 1 for air)
    # `groups` is the group tensor (each either 0 for majority group or 1 for minority group)
    batch = next(iter(train_loader))
    features, labels, groups = batch
    print(f'features size: {features.size()}, labels size: {labels.size()}, groups size: {groups.size()}')

    model = nn.Linear(in_size, out_size)
    optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
    privacy_engine = opacus.PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=1.0,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
    )
    criterion = nn.CrossEntropyLoss()

    # TODO:
    raise NotImplementedError("Your implementation goes here.")


def main():
    section_2_1()
    section_2_2()
    section_3()


if __name__ == "__main__":
    main()
