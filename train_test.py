import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    losses, accs = [], []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        pred = model(X)
        loss = loss_fn(pred, y)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluation metrics
        acc = (pred.argmax(dim=1) == y).sum().item() / len(X)
        loss = loss.item()

        losses.append(loss)
        accs.append(acc)

        if(batch % 30 == 0):
            cur_batch = batch * len(X)
            print(f"Training Accuracy: {acc * 100:3f}%", end='\t\t')
            print(f"Training Loss: {loss:>7f} [{cur_batch:>5d}/{size:>5d}]")

    return losses, accs

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)

    model.eval()
    loss, acc = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            loss += loss_fn(pred, y).item()
            acc += (pred.argmax(dim=1) == y).sum().item()
            
    loss /= size
    acc /= size

    print(f"Test Accuracy: {(acc * 100):3f}%, Test Loss: {loss:>7f}")
    return loss, acc