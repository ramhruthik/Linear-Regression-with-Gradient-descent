import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.figure
import time

# ground truth line y = 10x + 3
true_slope = 10
true_intercept = 3
x = np.random.randint(0,50, size = 20)
y = list()
true_y = list()
for i in range(len(x)):
    if random.randint(0,1) == 0:
        t = true_slope*x[i] + true_intercept
        y.append(t - random.uniform(0,t*0.1))
    else:
        t = true_slope*x[i] + true_intercept 
        y.append(t + random.uniform(0,t*0.1))
    true_y.append(t)

plt.plot(x,true_y,color="green")
plt.scatter(x,y)

weight = random.uniform(0,1)
bias = random.uniform(0,1)
Learning_rate = 0.000001
epochs = 500
n = float(len(x))
def GradientDescent(x,y,Learning_rate,weight,bias):
    y_pred = weight * x + bias 
    acc_bias = (1/n) * sum(y - y_pred) 
    acc_weight = (1/n) * sum(y * (y - y_pred)) 
    weight = weight + Learning_rate * acc_weight  # Update the weight
    bias = bias + Learning_rate * acc_bias  # Update the bias
    MSE = (1/n) * sum((y-y_pred)**2) #Mean squared Error
    return (weight,bias,MSE)
for i in range(epochs):
    weight,bias,MSE = GradientDescent(x,y,Learning_rate,weight,bias)
    y_values = x*weight + bias
    plt.ion()
    fig,ax = plt.subplots(figsize=(20,8))
    plt.rcParams.update({'figure.max_open_warning': 0})
    line2, = ax.plot(x,true_y,color = "Green")
    line3, = ax.plot(x,y_values,color = "Orange",markersize=3)
    print("iteration:",i,"Mean Squared Error:",MSE)
    ax.scatter(x,y)
    line2.set_xdata(x)
    line2.set_ydata(true_y)
    line3.set_xdata(x)
    line3.set_ydata(y_values)
    ax.set_title(f'Epochs : {i}', fontsize=15)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)
