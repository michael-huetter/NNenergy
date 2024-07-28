# simple NN with one hidden layer (universal approximator), however usually high density of sampling points needed and extrapolates poorly

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchviz import make_dot
from time import time


################ 1D sample PES ################

E = (np.array([-0.09107, -0.14992, -0.17130, -0.17311, -0.16448,
                         -0.15039, -0.13370, -0.11605, -0.09842, -0.08135,
                         -0.06515, -0.04997, -0.03590, -0.02294, -0.01109,
                         -0.00029, 0.00952, 0.01840, 0.02642, 0.03365,
                         0.04016, 0.04602, 0.05127, 0.05599, 0.06023,
                         0.06402, 0.06742, 0.07047, 0.07320, 0.07564,
                         0.07783, 0.07979, 0.08154, 0.08311, 0.08452,
                         0.08578, 0.08691, 0.08793, 0.08885, 0.08967,
                         0.09041, 0.09108, 0.09169, 0.09225, 0.09275]) ) * 27

# R values data
r = np.array([0.50000, 0.60000, 0.70000, 0.80000, 0.90000,
                     1.00000, 1.10000, 1.20000, 1.30000, 1.40000,
                     1.50000, 1.60000, 1.70000, 1.80000, 1.90000,
                     2.00000, 2.10000, 2.20000, 2.30000, 2.40000,
                     2.50000, 2.60000, 2.70000, 2.80000, 2.90000,
                     3.00000, 3.10000, 3.20000, 3.30000, 3.40000,
                     3.50000, 3.60000, 3.70000, 3.80000, 3.90000,
                     4.00000, 4.10000, 4.20000, 4.30000, 4.40000,
                     4.50000, 4.60000, 4.70000, 4.80000, 4.90000])

################ NN ################

#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu" 

t0 = time()

E = E.reshape(-1, 1).astype(np.float32) # vector of features as NN input
r = r.reshape(-1, 1).astype(np.float32)
E_tensor = torch.from_numpy(E).to(device)
r_tensor = torch.from_numpy(r).to(device)

# define the model
n = 20 # number of neurons in the hidden layer
d = 1 # number of features

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d, n)  # Input layer to hidden layer using linear transformation
        self.fc2 = nn.Linear(n, d)  # Hidden layer to output layer using linear transformation

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x)) # sigmoid activation function for hidden layer (introduces non-linearity)
        x = self.fc2(x)
        return x

model = Net().to(device)
x_dummy = torch.randn(1, 1, device=device)  # Dummy input for visualization
model_vis = make_dot(model(x_dummy), params=dict(model.named_parameters()))
model_vis.render("model_architecture", format="png") 



# Initialize the model and move it to the appropriate device
model = Net().to(device)

# define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()

# Training loop
epochs = 3000
loss_per_epoch = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(r_tensor)
    loss = loss_func(output, E_tensor)
    loss.backward()
    optimizer.step()

    loss_per_epoch.append(loss.item())
    if epoch % 100 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

# evaluate the model
r_eval = np.linspace(-2, 7, 1000).reshape(-1, 1).astype(np.float32)
r_eval_tensor = torch.from_numpy(r_eval).to(device)
model.eval()
with torch.no_grad():
    predictions = model(r_eval_tensor).cpu().numpy()

t1 = time()
print(f'Total runtime: {t1 - t0:.2f}s')

# Plot the results
plt.figure(figsize=(10, 5))
plt.title("Epoch loss")
plt.plot(loss_per_epoch, "bo")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.title("PES")
plt.scatter(r, E, color='blue', label='Actual Data')
plt.plot(r_eval, predictions, color='red', label='NN Predictions')
plt.legend()
plt.show()
