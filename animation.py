import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from policy import action_sample, get_cost, Policy, my_get_cost
from ortools_mtsp import my_solve_mtsp
import random
import os
import pickle
from functools import partial
from matplotlib.animation import PillowWriter


class AnimationDrawer(object):
    def __init__(self, dataset, n_agent, model, dev, fig, ax) -> None:
        self.dataset = dataset
        self.dev = dev
        self.n_agent = n_agent
        self.model = model
        
        self.fig, self.ax =  fig, ax
        
        self.routes = None

    def solve(self):
        '''
        Only need to solve when using learning-base methods, routes for ORTools are stored.
        '''
        shape = self.dataset.shape
        adj = torch.ones([shape[0],shape[1], shape[1]])
        data_list = [Data(x=self.dataset[i],edge_index=torch.nonzero(adj[i],as_tuple=False).t(),as_tuple=False) for i in range(shape[0])]
        batch_graph = Batch.from_data_list(data_list=data_list).to(self.dev)

        pi = self.model(batch_graph, n_nodes=shape[1], n_batch=shape[0])
        action, log_prob = action_sample(pi)
        reward, routes_idx, route_coords, all_length = my_get_cost(action, self.dataset, self.n_agent)
        self.routes = route_coords
    
    def interpolate(self, route):
            '''
            In: route, torch tensor: (n_city, 2)
            Our : route_inter, numpy array:(n,2)
            '''
            route = route.numpy()
            def inter(start, end):
                n = int(np.linalg.norm([end[0]-start[0], end[1]-start[1]])//0.01)
                return np.linspace(start, end, n, endpoint=False)
            route_inter = np.array([route[0, :]])
            for i in range(route.shape[0]-1):
                route_inter = np.append(route_inter, inter(route[i], route[i+1]),axis=0)
            route_inter = np.append(route_inter, np.array([route[-1]]),axis=0)
            return route_inter[1:]

    
    def draw(self):    
        '''
        i iter over batch, j iter over agent, fr for frame.
        '''
        routes = self.routes

        # for i in range(len(routes)):

        route_length = np.zeros((len(routes),), dtype=int)
        # Interpolate all the routes
        for i in range(len(routes)):
            temp_length = np.zeros((self.n_agent,), dtype=int)
            for j in range(self.n_agent):
                routes[i][j] = self.interpolate(routes[i][j])
                temp_length[j] = int(routes[i][j].shape[0])
            route_length[i] = np.max(temp_length)
            
        
        # Function for plotting one frame
        def draw_one(fr, i):
            print(f'frame: {fr}')
            colors = ['b', 'g', 'r', 'c', 'm']
            for j in range(self.n_agent):
                if fr <= routes[i][j].shape[0]:
                    self.ax.plot(routes[i][j][:fr,0], routes[i][j][:fr,1], color=colors[j])
            self.ax.set_title(name)
            self.ax.set_xlim([0,1])
            self.ax.set_ylim([0,1])

        for i in range(len(routes)):
            self.ax.clear()
            ax.scatter(self.dataset[i][:,0], self.dataset[i][:,1], s=10)
            ani = FuncAnimation(self.fig, partial(draw_one, i=i), frames=route_length[i], interval=10, repeat=False)
            ani.save(f"imgs/route_ani_{name}_{n_node_train}_{i}.gif", dpi=300,writer=PillowWriter(fps=40))



if __name__ == '__main__':
    manual_seed = 1
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmarks = False
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_agent = 5
    n_node_train = 100
    n_nodes = 500
    batch_size = 2
    seed = 1
    time_limit = 1800
    names = ['iMTSP', 'RL', 'ORTools', 'Var']
    name = 'ORTools'
    if name == 'iMTSP':
        policy = Policy(in_chnl=2, hid_chnl=64, n_agent=n_agent, key_size_embd=32,
            key_size_policy=128, val_size=16, clipping=10, dev=dev)
        path = './saved_model/iMTSP_{}.pth'.format(str(n_node_train) + '_' + str(n_agent))
        policy.load_state_dict(torch.load(path, map_location=torch.device(dev)))
    elif name == 'RL':
        policy = Policy(in_chnl=2, hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                        key_size_policy=64, val_size=64, clipping=10, dev=dev)
        path = './saved_model/RL_{}.pth'.format(str(n_node_train) + '_' + str(n_agent))
        policy.load_state_dict(torch.load(path, map_location=torch.device(dev)))
    elif name in ['ORTools', 'Var']:
        policy = None
        pass
    else:
        raise KeyError('name not defined')

    testing_data = torch.load('./testing_data/testing_data_' + str(n_nodes) + '_' + str(batch_size))
    print(testing_data.shape)
    fig, ax = plt.subplots()
    drawer = AnimationDrawer(testing_data, n_agent, policy, dev, fig, ax)
    if name in ['RL', 'iMTSP']:
        drawer.solve()
        drawer.draw()
    else:
        route_idx = pickle.load(open('./results/route.p','rb'))
        route_coords = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(len(route_idx)):
                route_coords.append(testing_data[i][route_idx[i], :])
        drawer.draw()