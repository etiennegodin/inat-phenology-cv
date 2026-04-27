import time

import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader


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


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> dict:
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            labels = labels.float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds)
            all_labels.append(labels)

    accuracy = correct / total
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    metrics = {"loss": total_loss / len(dataloader), "accuracy": accuracy, "cm": cm}
    return metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
):
    for epoch in range(epochs):
        start = time.time()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
        )
        evaluate_metrics = evaluate(
            model=model, dataloader=val_loader, criterion=criterion
        )
        elapsed = time.time() - start

        val_loss = evaluate_metrics["loss"]
        accuracy = evaluate_metrics["accuracy"]
        gap = val_loss - train_loss

        print(
            f"Epoch {epoch}: train={train_loss:.3f} val={val_loss:.3f} "
            f"gap={gap:.3f} accuracy ={accuracy:.3f} time ={elapsed:.3f}s"
        )

    evaluate_metrics = evaluate(model=model, dataloader=val_loader, criterion=criterion)
    train_report(evaluate_metrics=evaluate_metrics)
    torch.save(model.state_dict(), "checkpoints/model.pth")


def train_report(evaluate_metrics: dict):
    labels = ["flowering", "non_flowering"]
    df_cm = pd.DataFrame(evaluate_metrics["cm"], index=labels, columns=labels)
    print(df_cm)
