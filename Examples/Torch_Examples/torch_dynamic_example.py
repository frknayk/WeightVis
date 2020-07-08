import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from Libraries.Torch import Read_Torch
from Visualizer.Brain import Brain

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 32
hidden_size = 32
num_classes = 1
num_epochs = 5
learning_rate = 0.001


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

X_train = torch.randn(100,32).to(device)
Y_train = torch.randn(100,1).to(device)

# Read weights
torch_weights = Read_Torch(trained_weights=model.state_dict())
# Initate visualizer
brain = Brain(torch_weights.weights_list, torch_weights.biases_list)

# Train the model
for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Plot Brain
        brain.visualize()

# # Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')