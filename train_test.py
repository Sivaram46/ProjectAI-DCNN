import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(dataloader, model, loss_fn, optimizer):
    """
    Train the given model for one epoch, record and return loss and accuracy
    at the interval of 100 batches.
    """
    size = len(dataloader.dataset)
    losses, accs = [], []

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # metrics
        crt = (pred.argmax(1) == y).type(torch.float).sum().item()
        acc = 100 * crt / len(X)
        loss = loss.item()

        if(batch % 100 == 0):
            accs.append(acc)
            losses.append(loss)
            
            current = batch * len(X)
            print(f"[{current:>5d}/{size:>5d}]", end=' ')
            print(f"Loss: {loss:>7f}\tAccuracy: {acc:>0.1f}%")

    return losses, accs

def test(dataloader, model, loss_fn):
    """
    Test the given model for one epoch in test set and return the loss and 
    accuracy calculated for the whole dataset.
    """
    size = len(dataloader.dataset)

    model.eval()
    test_loss, crt = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            crt += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    acc = crt * 100 / size
    print(f"\nTest Accuracy: {acc:>0.1f}%\t Test Loss: {test_loss:>8f} \n")

    return test_loss, acc