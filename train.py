import time
import torch
from evaluate import evaluate


def training(model, num_epochs, optimizer, criterion, train_loader, val_loader):
    best_acc = 0
    best_model = model
    train_acc = []
    val_acc = []

    print("start of training")
    for epoch in range(num_epochs):
        epoch_time_start = time.time()
        print(f"epoch: {epoch}")
        correct = 0
        model.train()
        for x_, y_ in train_loader:
            optimizer.zero_grad()
            output = model(x_)
            y_pred = torch.argmax(output)
            loss = criterion(torch.reshape(output, (1, 62)), y_)
            loss.backward()
            optimizer.step()
            correct += (y_pred == y_[0]).float().sum()

        # Get the accuracies in terms of percentages
        train_acc_epoch = 100 * correct / train_loader.__len__()
        val_acc_epoch = evaluate(model, val_loader)

        if val_acc_epoch > best_acc:
            best_model = model
            best_acc = val_acc_epoch

        # Add the accuracy percentages to their respectful lists
        train_acc.append(train_acc_epoch)
        val_acc.append(val_acc_epoch)

        # Print train and val accuracys
        print(f"train accuracy: {train_acc_epoch}")
        print(f"validation accuracty: {val_acc_epoch}")
        print(f"epoch duration:{time.time()-epoch_time_start}")

    return train_acc, val_acc, best_model
