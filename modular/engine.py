import torch


def train_step(model, loss_fn, optimizer, dataloader,accuracy_fn, device: torch.device = device):

  train_loss, train_acc = 0, 0

  model.to(device)
  for batch, (X, y) in enumerate(dataloader):

    model.train()
    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  print(f'Training Loss: {train_loss} Training Accuracy: {train_acc}')


def test_step(model, loss_fn, optimizer, dataloader, accuracy_fn, device: torch.device = device):
    test_loss, test_acc = 0, 0

    model.to(device)
    model.eval()
    with torch.no_grad():
      for X, y in dataloader:
        X.to(device)
        y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)

        test_loss += loss
        test_acc += accuracy_fn(y, y_pred.argmax(dim=1))

      test_loss /= len(dataloader)
      test_acc /= len(dataloader)
      print(f'Testing Loss: {test_loss} Testing Accuracy: {test_acc}')


def train(model, train_dataloader, test_dataloader, optimizer, loss_fn,epochs, device):
    torch.manual_seed(42)
    from tqdm.auto import tqdm

    for epoch in tqdm(range(epochs)):
       print(f"Epoch: {epoch}")
       train_step(model, loss_fn, optimizer, train_dataloader, acc)
       test_step(model, loss_fn, optimizer, test_dataloader, acc)