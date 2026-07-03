import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import copy

class MLP_regresion(nn.Module):

    def __init__(self, input_size, hidden_layers=[128, 64], activation=nn.ReLU, dropout=0):
        super().__init__()

        # output_size = 1, no n_classes -> una sola neurona de salida
        layer_sizes = [input_size] + hidden_layers + [1]
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))

        self.mlp_stack = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for layer in self.mlp_stack:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.mlp_stack(x)

    def predict(self, X, device):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32).to(device, non_blocking = True)
        with torch.no_grad():
            out = self(X)
        return out.cpu().numpy().flatten()  # sin argmax, devolvemos el valor crudo

    def train_loop(self, dataloader, loss_fn, optimizer, device):
        self.train()
        total_loss = 0
        for X, y in dataloader:
            X, y = X.to(device, non_blocking = True), y.to(device, non_blocking = True)
            pred = self(X).squeeze(1)  # de (batch, 1) a (batch,) para que matchee con y
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * len(y)
        return total_loss / len(dataloader.dataset)

    def eval_loop(self, dataloader, loss_fn, device):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self(X).squeeze(1)
                test_loss += loss_fn(pred, y).item()
        num_batches = len(dataloader)
        return test_loss / num_batches

    def train_model(self, train_loader, val_loader, optimizer, loss_fn, device, epochs=100, waiting=5):
        train_losses, val_losses = [], []
        pbar = tqdm(range(1, epochs + 1), desc="PyTorch", unit="epoca")

        best_val_loss = np.inf
        best_weights = None
        #wait_count = 0

        for epoch in pbar:
            train_loss = self.train_loop(train_loader, loss_fn, optimizer, device)
            val_loss = self.eval_loop(val_loader, loss_fn, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            #if val_loss < best_val_loss:
            #    best_val_loss = val_loss
            #    best_weights = copy.deepcopy(self.state_dict())
            #    wait_count = 0
            #else:
            #    wait_count += 1
            #    if wait_count >= waiting:
            #        self.load_state_dict(best_weights)
            #        pbar.set_postfix_str(f"early stopping -> epoca {epoch - waiting}")
            #        break

            pbar.set_postfix({"epoch": f"{epoch}"})
        return train_losses, val_losses

class MLP_clasificacion(nn.Module):
    def __init__(self, input_size, hidden_layers=[128, 64], n_classes=6, activation = nn.ReLU, dropout = 0):
        super().__init__()
        layer_sizes = [input_size] + hidden_layers + [n_classes]
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2: #not the last one
                layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        self.mlp_stack = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for layer in self.mlp_stack:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu") #He incialization
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.mlp_stack(x)

    def predict(self, X, device):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32)
        X = X.to(device)

        with torch.no_grad():
            mlp_stack = self(X)
        return mlp_stack.argmax(dim=1).cpu().numpy() #numpy needs it to be in the cpu

    def train_loop(self, dataloader, loss_fn, optimizer, device):
        self.train()
        total_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = self(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * len(y)
        return total_loss / len(dataloader.dataset)

    def eval_loop(self, dataloader, loss_fn, device):
        self.eval()
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self(X)
                test_loss += loss_fn(pred, y).item()

        num_batches = len(dataloader)
        return test_loss / num_batches

    def train_model(self, train_loader, val_loader, optimizer, loss_fn, device, epochs=100, waiting = 5):
        train_losses, val_losses = [], []
        pbar = tqdm(range(1, epochs + 1), desc="PyTorch", unit="epoca")

        best_val_loss = np.inf
        best_weights = None
        #wait_count = 0

        for epoch in pbar:
            train_loss = self.train_loop(train_loader, loss_fn, optimizer, device)
            val_loss = self.eval_loop(val_loader, loss_fn, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            #early stopping:
            #if val_loss < best_val_loss:
            #    best_val_loss = val_loss
            #    best_weights = copy.deepcopy(self.state_dict())
            #    wait_count = 0
            #else:
            #    wait_count += 1
            #    if wait_count >= waiting:
            #        self.load_state_dict(best_weights)
            #        pbar.set_postfix_str(f"early stopping -> epoca {epoch - waiting}")
            #        break

            pbar.set_postfix({"epoch":  f"{epoch}"})
        return train_losses, val_losses
