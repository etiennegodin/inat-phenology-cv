import time

import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score
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
    all_preds_raw = []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            labels = labels.float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds_raw = torch.sigmoid(outputs)
            preds = (preds_raw > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds)
            all_labels.append(labels)
            all_preds_raw.append(preds_raw)

    accuracy = correct / total
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_preds_raw = torch.cat(all_preds_raw)

    cm = confusion_matrix(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds_raw)
    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "cm": cm,
        "roc": roc_auc,
    }
    return metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    patience: int = 3,
):
    best_loss = 1e10
    patience_counter = 0
    for epoch in range(epochs):
        start = time.time()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
        )
        metrics = evaluate(model=model, dataloader=val_loader, criterion=criterion)
        elapsed = time.time() - start

        val_loss = metrics["loss"]
        gap = val_loss - train_loss
        print(
            f"Epoch {epoch}: train={train_loss:.3f} val={val_loss:.3f} "
            f"gap={gap:.3f} accuracy ={float(metrics['accuracy']):.3f} "
            f" roc = {float(metrics['roc']):.3f} time ={elapsed:.3f}s"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            # Save only if better
            torch.save(model.state_dict(), "checkpoints/model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    metrics = evaluate(model=model, dataloader=val_loader, criterion=criterion)
    train_report(metrics=metrics)


def train_report(metrics: dict):
    labels = ["flowering", "non_flowering"]
    df_cm = pd.DataFrame(metrics["cm"], index=labels, columns=labels)
    print(df_cm)
