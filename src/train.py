import torch


def train_one_epoch(model, dataloader, optimizer, criterion):
    total_loss = 0
    model.train()
    for images, labels in dataloader:
        labels = labels.float().unsqueeze(1)
        optimizer.zero_grad()
        ouputs = model(images)
        loss = criterion(ouputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            labels = labels.float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train(model, train_loader, val_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
        )
        val_loss = evaluate(model=model, dataloader=val_loader, criterion=criterion)
        print(f"Epoch {epoch}: train={train_loss:.3f} val={val_loss:.3f}")
