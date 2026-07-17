import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Generar datos sintéticos
np.random.seed(42)
X = np.random.randn(1000, 10).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

# Convertir a tensores
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# Dataset y DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Definir red neuronal
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 2 clases
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(32)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

# Instanciar modelo
model = SimpleNN()

# Loss function y optimizador
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
epochs = 50
train_losses = []

for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        # Forward pass
        logits = model(batch_X)
        loss = loss_fn(logits, batch_y)
        
        # Backward pass
        optimizer.zero_grad()  # limpiar gradientes
        loss.backward()  # calcular gradientes
        optimizer.step()  # actualizar pesos
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Visualizar entrenamiento
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Predicción
model.eval()
with torch.no_grad():
    logits = model(X_tensor[:5])
    predictions = torch.argmax(logits, dim=1)
    print(f"Predicciones: {predictions}")