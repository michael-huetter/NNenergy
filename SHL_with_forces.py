# NN can also easily be used to fit a PES and forces simltaniously.

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

np.random.seed(111)

################ harmonic oscillator to define sample PES and forces (including some noise) ################

a = 0.1

def V(r):
    return 0.5 * r**2 + a * np.random.randn(*r.shape)
def F(r):
    return r + a * np.random.randn(*r.shape)

r = np.linspace(-5, 5, 9)
Vr = V(r)
Fr = F(r)

################ Neural Network ################

device = "cpu" 

Vr = Vr.reshape(-1, 1).astype(np.float32) # vector of features as NN input
Fr = Fr.reshape(-1, 1).astype(np.float32)
r = r.reshape(-1, 1).astype(np.float32)
dat = np.concatenate((Vr, Fr), axis=1)


dat_tensor = torch.from_numpy(dat).to(device)
r_tensor = torch.from_numpy(r).to(device)

# define the model
n = 50 
d_in = 1
d_out = 2 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d_in, n)  
        self.fc2 = nn.Linear(n, d_out)  

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()

# Training loop
epochs = 3000
loss_per_epoch = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(r_tensor)
    loss = loss_func(output, dat_tensor)
    loss.backward()
    optimizer.step()

    loss_per_epoch.append(loss.item())
    if epoch % 100 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

# evaluate the model
r_eval = np.linspace(-6, 6, 1000).reshape(-1, 1).astype(np.float32)
r_eval_tensor = torch.from_numpy(r_eval).to(device)
model.eval()
with torch.no_grad():
    predictions = model(r_eval_tensor).cpu().numpy()

# Plot the results
plt.figure(figsize=(10, 5))
plt.title("Epoch loss")
plt.plot(loss_per_epoch, "bo")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.title("PES and Forces")
plt.plot(r_eval, predictions, color='black', label='NN Predictions')
plt.plot(r, Vr, "ro", label='PES')
plt.plot(r, Fr, "bo", label='Forces')
plt.legend()
plt.show()

