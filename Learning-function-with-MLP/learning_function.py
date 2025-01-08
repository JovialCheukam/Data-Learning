from deeplearning import *
# Numpy library
import numpy as np
# Plotting library
import matplotlib.pyplot as plt


input_dim = 2
output_dim = 1
depth = 15
width_values = [5,10,20]
activation_functions = ['tanh','sin','relu']

def ML(predictor, input_dim, output_dim, width, depth, activation):
    mlp = predictor(input_dim,output_dim,width,depth,activation)
    return mlp

predictors_list = []

for width in width_values:
  for activation in activation_functions:
    mlp = ML(MLP, input_dim, output_dim, width, depth, activation)
    predictors_list.append([mlp.deep_copy(),width,activation])
    mlp.width = width
    num_params = mlp.num_network_params()
    print("le nombre de parametres de MLP avec", depth-1," couches cachées de ", width, "neurones activés par ", activation, " est: ", num_params)


# defining a function to generate a sample of data in a tensor format 
def gen_data():
    numbers = []
    abs = -1
    ord = -1
    for _ in range(201):
      for _ in range(201):
        numbers.append(abs)
        numbers.append(ord)
        ord = ord + 0.01
      abs = abs + 0.01
    return torch.tensor(numbers).reshape(40401,2)

data = gen_data()

output_list = []
for mlp in predictors_list:
    output_list.append([mlp[0].forward(gen_data()).detach().numpy(),mlp[1],mlp[2]])

len(output_list)


# plot the evaluation of the data sample by the Multi Layer Perceptron (MLP)  according to IT architecture 
fig, _axs = plt.subplots(nrows=3, ncols=3)
fig.subplots_adjust(hspace=0.6)
axs = _axs.flatten()

axs[0].contourf(output_list[0][0].reshape(201,201))
axs[0].set_title(str(output_list[0][1]) + " " + output_list[0][2])

axs[1].contourf(output_list[1][0].reshape(201,201))
axs[1].set_title(str(output_list[1][1]) + " " + output_list[1][2])

axs[2].contourf(output_list[2][0].reshape(201,201))
axs[2].set_title(str(output_list[2][1]) + " " + output_list[2][2])

axs[3].contourf(output_list[3][0].reshape(201,201))
axs[3].set_title(str(output_list[3][1]) + " " + output_list[3][2])

axs[4].contourf(output_list[4][0].reshape(201,201))
axs[4].set_title(str(output_list[4][1]) + " " + output_list[4][2])

axs[5].contourf(output_list[5][0].reshape(201,201))
axs[5].set_title(str(output_list[5][1]) + " " + output_list[5][2])

axs[6].contourf(output_list[6][0].reshape(201,201))
axs[6].set_title(str(output_list[6][1]) + " " + output_list[6][2])

axs[7].contourf(output_list[7][0].reshape(201,201))
axs[7].set_title(str(output_list[7][1]) + " " + output_list[7][2])

axs[8].contourf(output_list[8][0].reshape(201,201))
axs[8].set_title(str(output_list[8][1]) + " " + output_list[8][2])



# define the target function to learn : f(x) = tan(20(x-0.5)) + sin(10xpi) + normal noise
def target_func(x):
    val = np.tanh(20 * (x-0.5)) + np.sin(10 * np.pi * x)
    # Adding noise
    val += 0.3 * np.random.normal(size=val.shape)
    return val.to(torch.float32)

# generate a sample of data training from the target function
np.random.seed(1)
Ntrain = 100
Nval   = 50
N      = Ntrain + Nval
x      = torch.linspace(0, 1, N).reshape(N,1)
ind_all  = np.arange(0,N)
np.random.shuffle(ind_all)
ind_train = np.sort(ind_all[0:Ntrain])
ind_val  = np.sort(ind_all[Ntrain:])

x_train =  x[ind_train]
x_val = x[ind_val]
y_train = target_func(x_train)
y_val = target_func(x_val)

# plot the generated data sample with the target function
fig, ax = plt.subplots(figsize = (15,5))
ax.plot(x_train,y_train,'-o', label='Training points')
ax.plot(x_val,y_val,'-x', label='Validation points')
plt.legend()
plt.savefig('data_plot.png')

# create a wrapper for customizing the data set
training_samples = np.concatenate((x_train.reshape(-1,1),y_train.reshape(-1,1)),axis=1)
train_dataset = CustomDataset(samples=training_samples)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)


testing_samples = np.concatenate((x_val.reshape(-1,1),y_val.reshape(-1,1)),axis=1)
test_dataset = CustomDataset(samples=testing_samples)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)


# define a function for the customize the training architechture
def train_loop(epochs, Weight_decay, x_eval):
    mlp = mlp_initial.deep_copy()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay = Weight_decay)
    loss_train_epoch = []
    loss_test_epoch = []
    for epoch in range(epochs):
        for data_train in train_loader:
            optimizer.zero_grad()
            # Compute prediction and loss
            pred = mlp(data_train[:,input_dim-1].reshape(data_train.shape[0],1))
            loss = loss_fn(pred, data_train[:,input_dim:].reshape(data_train.shape[0],1))

            # Backpropagation
            loss.backward()
            optimizer.step()

        test_loss = 0
        with torch.no_grad():
             for data_test in test_loader:
                 pred_val = mlp(data_test[:,input_dim-1].reshape(data_test.shape[0],1))
                 test_loss += loss_fn(pred_val, data_test[:,input_dim:].reshape(data_test.shape[0],1)).item()
        loss_train_epoch.append(loss.item())
        loss_test_epoch.append(test_loss)
    with torch.no_grad():
         y_eval = mlp(x_eval)
    return loss_train_epoch, loss_test_epoch, y_eval


# initialize the hyperparamters
input_dim = 1
output_dim = 1
width = 45
depth = 8
activation = 'sin'
mlp_initial = ML(MLP, input_dim, output_dim, width, depth, activation)
learning_rate = 1e-3
epochs = 100
x_eval = torch.linspace(0, 1, 1000).reshape(1000,1)

# plot the training and validation error according to the MLP architechture
loss_train_epoch1, loss_test_epoch1, y_eval1 = train_loop(epochs, 0.0, x_eval)
loss_train_epoch2, loss_test_epoch2, y_eval2 = train_loop(epochs, 1e-3, x_eval)
loss_train_epoch3, loss_test_epoch3, y_eval3 = train_loop(epochs, 1e-2, x_eval)
loss_train_epoch4, loss_test_epoch4, y_eval4 = train_loop(epochs, 1e-1, x_eval)

fig, _axs = plt.subplots(nrows=2, ncols=2,figsize = (12,8))
fig.subplots_adjust(hspace=1)
axs = _axs.flatten()

axs[0].plot(range(epochs),loss_train_epoch1,'-o', label='Training error')
axs[0].plot(range(epochs),loss_test_epoch1,'-x', label='Validation error')
axs[0].axhline(y = sum(loss_train_epoch1)/epochs, color = 'g', linestyle = '--',label='Training error average')
axs[0].axhline(y = sum(loss_test_epoch1)/epochs, color = 'y', linestyle = '--',label='Validation error average')
axs[0].set_title("Weight_dacay 0.0")
#axs[0].legend()

axs[1].plot(range(epochs),loss_train_epoch2,'-o', label='Training error')
axs[1].plot(range(epochs),loss_test_epoch2,'-x', label='Validation error')
axs[1].axhline(y = sum(loss_train_epoch2)/epochs, color = 'g', linestyle = '--',label='Training error average')
axs[1].axhline(y = sum(loss_test_epoch2)/epochs, color = 'y', linestyle = '--',label='Validation error average')
axs[1].set_title("Weight_dacay 1e-3")
#axs[1].legend()

axs[2].plot(range(epochs),loss_train_epoch3,'-o', label='Training error')
axs[2].plot(range(epochs),loss_test_epoch3,'-x', label='Validation error')
axs[2].axhline(y = sum(loss_train_epoch3)/epochs, color = 'g', linestyle = '--',label='Training error average')
axs[2].axhline(y = sum(loss_test_epoch3)/epochs, color = 'y', linestyle = '--',label='Validation error average')
axs[2].set_title("Weight_dacay 1e-2")
#axs[2].legend()

axs[3].plot(range(epochs),loss_train_epoch4,'-o', label='Training error')
axs[3].plot(range(epochs),loss_test_epoch4,'-x', label='Validation error')
axs[3].axhline(y = sum(loss_train_epoch4)/epochs, color = 'g', linestyle = '--',label='Training error average')
axs[3].axhline(y = sum(loss_test_epoch4)/epochs, color = 'y', linestyle = '--',label='Validation error average')
axs[3].set_title("Weight_dacay 1e-1")
#axs[3].legend()

#plt.legend()
plt.show()

# plot the target function and his prediction for each MLP architechture
fig, ax = plt.subplots(nrows=4, ncols=1,figsize = (15,12))
fig.subplots_adjust(hspace=1)

ax[0].plot(x_eval,target_func(x_eval),'-o', label='Target')
ax[0].plot(x_eval,y_eval1,'-+', label='0.0')
ax[0].legend()

ax[1].plot(x_eval,target_func(x_eval),'-o', label='Target')
ax[1].plot(x_eval,y_eval2,'-+', label='1e-3')
ax[1].legend()

ax[2].plot(x_eval,target_func(x_eval),'-o', label='Target')
ax[2].plot(x_eval,y_eval3,'-+', label='1e-2')
ax[2].legend()

ax[3].plot(x_eval,target_func(x_eval),'-o', label='Target')
ax[3].plot(x_eval,y_eval4,'-+', label='1e-1')
ax[3].legend()