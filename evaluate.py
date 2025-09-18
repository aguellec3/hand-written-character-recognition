import torch


def evaluate(model, loader):
    model.eval()
    correct = 0
    for x_, y_ in loader:
        with torch.no_grad():
            output = model(x_)
            y_pred = torch.argmax(output)
            correct += (y_pred == y_[0]).float().sum()

    return 100 * correct / loader.__len__()
