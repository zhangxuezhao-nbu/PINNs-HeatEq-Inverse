################################### libs ######################################
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from matplotlib import cm
import time
from scipy.signal import savgol_filter
 ##### Modify loss #########33

#############################################      Net for u(t,x,y)    #######################################

class Net(nn.Module):   
    # NL: the number of hidden layers
    # NN: the number of vertices in each layer
    def __init__(self,dimension, NL, NN):   
        """"net = Net(dimension=3, NL=2, NN=20) # net(x).size()=[2**8,1]
            f_NN= Net(dimension=2, NL=2, NN=20) # f_NN(x).size()=[2**8,1]
        """
        super(Net, self).__init__()  

        self.input_layer = nn.Linear(dimension, NN)  

        self.hidden_layers = nn.ModuleList([nn.Linear(NN, NN) for i in range(NL)])

        self.output_layer = nn.Linear(NN, 1)

        self.a = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        o = self.act(self.input_layer(x))

        for i, li in enumerate(self.hidden_layers):
            o = self.act(li(o))

        out = self.output_layer(o) 

        return out

    def act(self, x):
        return x * torch.tanh(self.a * x)  


############################################### Heat #######################################

class Heat():
    """"
    te = 4/3
    xe = 1
    ye = 1
    heatequation = Heat(net,f_NN, te, xe, ye)
    """

    def __init__(self, net,f_NN, te, xe):
        self.net = net
        self.f_NN = f_NN
        self.te = te  
        self.xe = xe  


    def sample(self, size=2**8):
        te = self.te
        xe = self.xe
        x = torch.cat((torch.rand([size, 1]) * te, torch.rand([size, 1]) * xe), dim=1)  
        x_initial = torch.cat((torch.zeros(size, 1), torch.rand([size, 1]) * xe), dim=1) 
        x_boundary_left = torch.cat((torch.rand([size, 1]) * te, torch.zeros([size, 1]) * xe), dim=1)  
        x_boundary_right = torch.cat((torch.rand([size, 1]) * te, torch.ones([size, 1]) * xe), dim=1) 
        x_terminal_1 = torch.cat(
            (torch.ones(size, 1) * te, torch.rand([size, 1]) * xe * 0.5), dim=1)
        x_terminal_2 = torch.cat(
            (torch.ones(size, 1) * te * 0.5, torch.rand([size, 1]) * xe * 0.5), dim=1)
        x_bar = x[:, 1].reshape(-1, 1)  


        return x, x_initial, x_boundary_left, x_boundary_right, x_terminal_1,x_terminal_2,x_bar


    def absolute_error_f(self, size=2**8):  
        x, x_initial, x_boundary_left, x_boundary_right,x_terminal_1,x_terminal_2,x_bar = self.sample(size=size)
        predict_f = f_NN(x_bar) 
        exact_f = f_exact(x_bar)
        value_f = torch.sqrt(torch.sum((predict_f - exact_f )**2))/torch.sqrt(torch.sum((exact_f)**2))
        return value_f

    def absolute_error_u(self, size=2**8):   
        x, x_initial, x_boundary_left, x_boundary_right,x_terminal_1,x_terminal_2,x_bar = self.sample(size=size)
        predict_u =net(x)   
        exact_u = u_exact(x)
        value_u = torch.sqrt(torch.sum((predict_u - exact_u )**2))/torch.sqrt(torch.sum((exact_u)**2))
        return value_u

    def loss_func(self, size=2 ** 8):
        x, x_initial, x_boundary_left, x_boundary_right, x_terminal_1, x_terminal_2, x_bar = self.sample(size=size)

        ############################## interior diff equation ###############################
        x = Variable(x, requires_grad=True) 

        d = torch.autograd.grad(self.net(x), x, grad_outputs=torch.ones_like(self.net(x)),
                                create_graph=True) 
        dx = d[0][:, 1].reshape(-1, 1)

        dxx = torch.autograd.grad(dx, x, grad_outputs=torch.ones_like(dx), create_graph=True)[0][:, 1].reshape(-1, 1)
 

        Ff = (torch.exp(-0.5 * x[:, 0]).view(-1, 1)) * f_NN(x_bar)  

        diff_error = torch.mean((dt - dxx - Ff.reshape(-1, 1)) ** 2)

        ############################# initial condition ##########################
        u_0 = 2 * ((torch.exp(-0.5 * x_initial[:, 0])).view(-1, 1) * torch.sin(x_initial[:, 1]).view(-1, 1))
          
        init_error = torch.mean((self.net(x_initial) - u_0.reshape(-1, 1)) ** 2)
        ######################### boundary condition ############################

        bd_left_error = torch.mean((self.net(x_boundary_left)) ** 2)  
        bd_right_error = torch.mean((self.net(x_boundary_right)) ** 2)

        x_boundary_left = Variable(x_boundary_left, requires_grad=True)
        d_boundary_left = torch.autograd.grad(self.net(x_boundary_left), x_boundary_left,
                                              grad_outputs=torch.ones_like(net(x_boundary_left)), create_graph=True)
        dt_boundary_left = d_boundary_left[0][:, 0].reshape(-1, 1)
        dt_bd_left_error = torch.mean((dt_boundary_left) ** 2)  
        x_boundary_right = Variable(x_boundary_right, requires_grad=True)
        d_boundary_right = torch.autograd.grad(self.net(x_boundary_right), x_boundary_right,
                                               grad_outputs=torch.ones_like(net(x_boundary_right)), create_graph=True)
        dt_boundary_right = d_boundary_right[0][:, 0].reshape(-1, 1)
        dt_bd_right_error = torch.mean((dt_boundary_right) ** 2)  

        ################## Measurement Data ########################

        delta = 0
        delt = 0.1
        varphi_1 = net(x_terminal_1) * (1 + delta * torch.randn(size, 1))  
        varphi_2 = net(x_terminal_2) * (1 + delta * torch.randn(size, 1))  

        u_exact_1 = ((torch.exp(-0.5 * x_terminal_1[:, 0])).view(-1, 1) * (
            torch.sin(x_terminal_1[:, 1]).view(-1, 1))) * 2
        u_exact_2 = ((torch.exp(-0.5 * x_terminal_2[:, 0])).view(-1, 1) * (
            torch.sin(x_terminal_2[:, 1]).view(-1, 1))) * 2

        u_exact_11 = u_exact_1 * (1 + delt * torch.randn(size, 1))
        u_exact_22 = u_exact_2 * (1 + delt * torch.randn(size, 1))

        data_error_1 = torch.mean((varphi_1 - u_exact_11.reshape(-1, 1)) ** 2)
        data_error_2 = torch.mean((varphi_2 - u_exact_22.reshape(-1, 1)) ** 2)

        sigma = 0.1 
    
        loss=sigma * diff_error +  1 * init_error +0.1*(bd_left_error +bd_right_error  + dt_bd_left_error+dt_bd_right_error)+(data_error_1 + data_error_2)

        return loss,0.1* diff_error ,1 * init_error , 0.1*bd_left_error ,0.1* bd_right_error,0.1*dt_bd_left_error,0.1*dt_bd_right_error,data_error_1 , data_error_2

##########################################  train  #########################################

class Train():
    def __init__(self, net, f_NN, heateq, BATCH_SIZE):
        self.errors = []
        self.errors_f = []
        self.errors_u = []
        self.net_a1 = []
        self.fNN_a2 = []
        self.BATCH_SIZE = BATCH_SIZE
        self.net = net
        self.f_NN = f_NN
        self.model = heateq

    def train(self, epoch, lr):
        optimizer = optim.Adam([{"params": net.parameters(), 'lr': lr}, {"params": f_NN.parameters(), 'lr': lr}],weight_decay=0)
       
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40000, gamma=1, last_epoch=-1)
       
        avg_loss = 0
        time_start = time.time()

        diff_error_list = []
        init_error_list = []
        bd_left_error_list = []
        bd_right_error_list = []
        dt_bd_left_error_list = []
        dt_bd_right_error_list = []
        bd_up_error_list = []
        bd_down_error_list = []
        data_error_1_list = []
        data_error_2_list = []
      
        for e in range(epoch):

            if e % 10000 == 0:
                torch.save(diff_error_list, 'diff_error_list_{}.pkl'.format(e))
                torch.save(init_error_list, 'init_error_list_{}.pkl'.format(e))
                torch.save(bd_left_error_list, 'bd_left_error_list_{}.pkl'.format(e))
                torch.save(bd_right_error_list, 'bd_right_error_list_{}.pkl'.format(e))
                torch.save(bd_up_error_list, 'bd_up_error_list_{}.pkl'.format(e))
                torch.save(bd_down_error_list, 'bd_down_error_list_{}.pkl'.format(e))
                torch.save(dt_bd_left_error_list, 'dt_bd_left_error_list_{}.pkl'.format(e))
                torch.save(dt_bd_right_error_list, 'dt_bd_right_error_list_{}.pkl'.format(e))
                
                torch.save(data_error_1_list, 'data_error_1_list_{}.pkl'.format(e))
                torch.save(data_error_2_list, 'data_error_2_list_{}.pkl'.format(e))
                torch.save(f_NN, 'DGM1_fNN_model_{}.pkl'.format(e))
                torch.save(net, 'DGM1_net_model_{}.pkl'.format(e))
                errors = train.get_errors()
                errors_f = train.get_errors_f()
                errors_u = train.get_errors_u()
                torch.save(errors, 'DGM1_errors_model_{}.pkl'.format(e))
                torch.save(errors_f, 'DGM1_errors_model_f_{}.pkl'.format(e))
                torch.save(errors_u, 'DGM1_errors_model_u_{}.pkl'.format(e))
                torch.save(self.net_a1, 'net_a_{}.pkl'.format(e))
                torch.save(self.fNN_a2, 'fNN_a_{}.pkl'.format(e))
               
                print('Saved Epoch:', format(e))

            optimizer.zero_grad()
          
            loss, diff_error, init_error, bd_left_error, bd_right_error, dt_bd_left_error,dt_bd_right_error,data_error_1, data_error_2 = self.model.loss_func(
                self.BATCH_SIZE)
            avg_loss = avg_loss + loss
            loss.backward()  
           
            optimizer.step() 
         
            scheduler.step()
            lr_r = scheduler.get_lr()

            # scheduler_1.step()
            lr_r_1 = scheduler.get_lr() 

            error_f = self.model.absolute_error_f()
            error_u = self.model.absolute_error_u()

            if e % 50 == 49:
                loss = avg_loss / 50
                print("Epoch {} - lr {} -  loss: {}".format(e, lr_r, loss))
                print("current absolute error for f is: ", error_f.detach())
                print("current absolute error for u is: ", error_u.detach())
                print()
                print("Epoch {} - lr {}".format(e, lr_r_1))
                print("net.a",net.a)
                print("f_NN.a",f_NN.a)
                print()
                avg_loss = 0
                error = self.model.loss_func(self.BATCH_SIZE)[0]
                self.errors.append(error.detach())
                error_f = self.model.absolute_error_f()
                self.errors_f.append(error_f.detach())
                error_u = self.model.absolute_error_u()
                self.errors_u.append(error_u.detach())

                diff_error_list.append(diff_error)
                init_error_list.append(init_error)
                bd_left_error_list.append(bd_left_error)
                bd_right_error_list.append(bd_right_error)
                dt_bd_left_error_list.append(dt_bd_left_error)
                dt_bd_right_error_list.append(dt_bd_right_error)
                data_error_1_list.append(data_error_1)
                data_error_2_list.append(data_error_2)

                self.net_a1.append(self.net.a.item())
                self.fNN_a2.append(self.f_NN.a.item())


        torch.save(diff_error_list, 'diff_error_list.pkl')
        torch.save(init_error_list, 'init_error_list.pkl')
        torch.save(bd_left_error_list, 'bd_left_error_list.pkl')
        torch.save(bd_right_error_list, 'bd_right_error_list.pkl')
        torch.save(dt_bd_left_error_list, 'dt_bd_left_error_list.pkl')
        torch.save(dt_bd_right_error_list, 'dt_bd_right_error_list.pkl')
        torch.save(data_error_1_list, 'data_error_1_list.pkl')
        torch.save(data_error_2_list, 'data_error_2_list.pkl')

        torch.save(self.net_a1, 'net_a.pkl')
        torch.save(self.fNN_a2, 'fNN_a.pkl')
      
        time_end = time.time()
        print('total time is: ', time_end - time_start, 'seconds')


    def get_errors(self):
        return self.errors

    def get_errors_f(self):
        return self.errors_f

    def get_errors_u(self):
        return self.errors_u




def u_exact(x):
   
    u_exact = (torch.exp(-0.5 * x[:, 0]).view(-1, 1) * torch.sin(x[:, 1]).view(-1, 1)) * 2
    return u_exact


def f_exact(x_bar):
    
    f_exact = torch.sin(x_bar[:, 0].view(-1,1))
    return f_exact

##################################################        main       #######################################


net = Net(dimension=2, NL=3, NN=20)  
f_NN= Net(dimension=1, NL=3, NN=20)  

te = 1
xe = np.pi

heatequation = Heat(net,f_NN, te, xe)
epoch=10*(10**5)

lr=0.001
lr_1=0.001
BATCH_SIZE=2**8

#################################### Training for  u(t,x,y),f(x,y) ################################
train = Train(net, f_NN, heatequation, BATCH_SIZE)
train.train(epoch, lr)
torch.save(f_NN, 'DGM1_fNN_model.pkl')
torch.save(net, 'DGM1_net_model.pkl')

errors = train.get_errors()
torch.save(errors, 'DGM1_errors_model.pkl')
errors_f = train.get_errors_f()
torch.save(errors_f, 'DGM1_errors_model_f.pkl')
errors_u = train.get_errors_u()
torch.save(errors_u, 'DGM1_errors_model_u.pkl')

##################################   plot training loss   #####################

errors=torch.load('DGM1_errors_model.pkl')
fig = plt.figure()
plt.plot(np.log(errors), '-b', label='Errors')
plt.title('Training Loss', fontsize=10)
path = "./pictures/DGM_Eg1_TrainingLoss.png"

plt.savefig(path)

############################  plot RelativeError for f  ###########################

errors=torch.load('DGM1_errors_model_f.pkl')
fig = plt.figure()
plt.plot(np.log(errors), '-b', label='Errors')
plt.title('RelativeError for f', fontsize=10)
path = "./pictures/DGM_Eg1_RelativeError_f.png"

plt.savefig(path)

############################  plot RelativeError for u ###########################

errors=torch.load('DGM1_errors_model_u.pkl')
fig = plt.figure()
plt.plot(np.log(errors), '-b', label='Errors')
plt.title('RelativeError for u', fontsize=10)
path = "./pictures/DGM_Eg1_RelativeError_u.png"

plt.savefig(path)


############################  plot net.a for u  ###########################

a=torch.load('net_a.pkl')
fig = plt.figure()
plt.plot(a, '-b', label='net.a')
plt.title('net.a for u', fontsize=10)
path = "./pictures/net_a.png"

plt.savefig(path)

############################  plot fNN.a for f  ###########################

a=torch.load('fNN_a.pkl')
fig = plt.figure()
plt.plot(a, '-b', label='fNN.a')
plt.title('fNN.a for f', fontsize=10)
path = "./pictures/fNN_a.png"

plt.savefig(path)


##################################################  定义一个函数来加载数据并绘制图形
def plot_error(file_path, label,maker, color):
    errors = torch.load(file_path)
    errors_tensor = torch.tensor(errors)

    log_errors = np.log(errors_tensor.detach().numpy())

    x = np.arange(len(log_errors))

    plt.plot(x, log_errors, color=color, marker=maker, label=label,linestyle='-',linewidth=0.8, ms=1.5)

fig = plt.figure(figsize=(12, 6))
plot_error('diff_error_list.pkl', 'Diff Errors','o', 'b')
plot_error('init_error_list.pkl', 'Init Errors', 'o','r')
plot_error('bd_left_error_list.pkl', 'BD Left Errors','o', 'g')
plot_error('bd_right_error_list.pkl', 'BD Right Errors', 'o','y')
plot_error('dt_bd_left_error_list.pkl', 'BD Left Errors','o', 'g')
plot_error('dt_bd_right_error_list.pkl', 'BD Right Errors', 'o','y')
plot_error('data_error_1_list.pkl', 'Data Error 1', 'o','k')
plot_error('data_error_2_list.pkl', 'Data Error 2', '^','b')


plt.title('Error Comparison', fontsize=10)
plt.xlabel('Iteration')
plt.ylabel('Log Error')
plt.legend()


path = "./pictures/combined_error.png"

plt.savefig(path)
plt.show()


########################################## After Training: plot exact and numerical (test) sol. for u(t,x,y) ##################################


from matplotlib import ticker, cm

net = torch.load('DGM1_net_model.pkl')


t_range = np.linspace(0, te, 100, dtype=np.float64)
x_range = np.linspace(0, xe, 100, dtype=np.float64)


data = np.empty((2, 1)) 

TrueZ = []  
Z = []
for _t in t_range:
    data[0] = _t
    for _x in x_range:
        data[1] = _x
        indata = torch.Tensor(data.reshape(1, -1))  
        Zdata = net(indata).detach().cpu().numpy()   
        Z.append(Zdata)  
        delt=0
        u_exact_1 = (np.exp(-0.5 * _t)) * (np.sin(_x)) * 2
        u_exact = u_exact_1 * (1 + delt * torch.randn(size, 1))
        print(delt * torch.randn(size, 1))
        TrueZ.append(u_exact)


torch.save(Z, 'DGM1_net_u_test.pkl')
torch.save(TrueZ, 'DGM1_exact_u_test.pkl')


##########################################  After Training:  exact and numerical sol. u(t,x,y) ##################################

u=torch.load('DGM1_net_u_test.pkl')
u=np.array(u)
u_surface=u.reshape(100,100)

exact_u=torch.load('DGM1_exact_u_test.pkl')
exact_u=np.array(exact_u)
exact_u_surface=exact_u.reshape(100,100)

#####################  time series L2 error ###############
error_save_u = np.zeros(t_range.shape[0])

for k in range(t_range.shape[0]):
    error_save_u[k] =(np.linalg.norm(exact_u_surface[k,:]- u_surface[k,:]))/np.linalg.norm(exact_u_surface[k,:])
print(error_save_u)

fig = plt.figure()
plt.plot(t_range, error_save_u, '-b', label='Errors')

plt.title('Time series relative error for u', fontsize=10)
path = "./pictures/DGM_Eg1_TimeSeries_RelativeError_u.png"
plt.savefig(path)

##########################################  After Training:  exact and numerical sol. f(x) ##################################

f_NN = torch.load('DGM1_fNN_model.pkl')

x_range = np.linspace(0, xe, 100, dtype=np.float64)  
Truef = []
f = []

for _x in x_range:
    data_f = np.array([[ _x ]]) 
    indata = torch.Tensor(data_f)
    fdata = f_NN(indata).detach().cpu().numpy()
    f.append(fdata[0, 0]) 
    f_exact = np.sin(1 * _x)  
    Truef.append(f_exact)


f_surface = np.array(f)
exact_f_surface = np.array(Truef)


Error_f = np.sqrt(np.sum((f_surface - exact_f_surface)**2)) / np.sqrt(np.sum((exact_f_surface)**2))
print(Error_f)


plt.figure(figsize=(10, 5))
plt.plot(x_range, f_surface, label='Approximation f_NN(x)', color='b')
plt.plot(x_range, exact_f_surface, label='Exact f(x)', color='r', linestyle='--')
plt.title('Approximation vs Exact Values')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()

plt.savefig('./pictures/DGM_Eg1_f_NN.png')
plt.show()
