from object import Value
import matplotlib.pyplot as plt
import numpy as np

class training:

    def __init__(self , net , X , y , learning_rate , num_epochs):
        self.net = net
        self.X = X
        self.y = y
        self.lr = learning_rate
        self.num_epochs = num_epochs


    def __call__(self):
        for epoch in range(self.num_epochs):

            total_loss = Value(0.0)

            for xi, yi in zip(self.X, self.y):
                inputs = [Value(xi[0]), Value(xi[1])]
                pred = self.net(inputs)
                target = Value(yi)
                loss = (pred - target) * (pred - target)
            
                # zero gradients before backward
                for p in self.net.parameters():
                    p.grad = 0.0
            
                # backpropagate on the single-sample loss
                loss.backward()
            
                # update parameters
                for p in self.net.parameters():
                    p.data -= self.lr * p.grad
            
                # accumulate for logging only
            total_loss = total_loss + loss


            # print loss
            if epoch % 20  == 0:
                avg_loss = total_loss.data / len(self.X)
                print(f"Epoch {epoch:03d} : avg loss = {avg_loss:.4f}")


    def plot(self , smoothness):
        h = smoothness
        x_min , x_max = self.X[: , 0].min() - 1 , self.X[: , 0].max() +1
        y_min , y_max = self.X[: , 1].min() - 1 , self.X[: , 1].max() +1

        xx , yy = np.meshgrid(np.arange(x_min , x_max , h) , np.arange(y_min , y_max , h))

        x_cor = np.c_[xx.ravel() , yy.ravel()]

        # converting the x_cor into Value object for our own NN
        inputs = [list(map(Value , xrow)) for xrow in x_cor]

        predictions = list(map(self.net , inputs))

        z = np.array([s.data > 0 for s in predictions])
        # print(" z : - \n", z)

        new_z = z.reshape(xx.shape)
        # print(" \n\n  new_z  : - \n\n" , new_z)

        fig = plt.figure()
        plt.contourf(xx , yy , new_z , cmap = plt.cm.Spectral , alpha = 0.8)
        plt.scatter(self.X[: , 0] , self.X[:,1] , c = self.y , cmap = plt.cm.Spectral )
        plt.xlim(xx.min() , xx.max())
        plt.ylim(yy.min() , yy.max())


                