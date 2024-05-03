from network import Net
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.distributions import Categorical
from ortools_tsp import solve, my_solve
from ortools_mtsp import my_solve_mtsp


class Agentembedding(nn.Module):
    def __init__(self, node_feature_size, key_size, value_size):
        super(Agentembedding, self).__init__()
        self.key_size = key_size
        self.q_agent = nn.Linear(2 * node_feature_size, key_size)
        self.k_agent = nn.Linear(node_feature_size, key_size)
        self.v_agent = nn.Linear(node_feature_size, value_size)

    def forward(self, f_c, f):
        q = self.q_agent(f_c)
        k = self.k_agent(f)
        v = self.v_agent(f)
        u = torch.matmul(k, q.transpose(-1, -2)) / math.sqrt(self.key_size)
        u_ = F.softmax(u, dim=-2).transpose(-1, -2)
        agent_embedding = torch.matmul(u_, v)

        return agent_embedding


class AgentAndNode_embedding(torch.nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_agent, key_size, value_size, dev):
        super(AgentAndNode_embedding, self).__init__()

        self.n_agent = n_agent

        # gin
        self.gin = Net(in_chnl=in_chnl, hid_chnl=hid_chnl).to(dev)
        # agent attention embed
        self.agents = torch.nn.ModuleList()
        for i in range(n_agent):
            self.agents.append(Agentembedding(node_feature_size=hid_chnl, key_size=key_size, value_size=value_size).to(dev))

    def forward(self, batch_graphs, n_nodes, n_batch):

        # get node embedding using gin
        nodes_h, g_h = self.gin(x=batch_graphs.x, edge_index=batch_graphs.edge_index, batch=batch_graphs.batch)
        nodes_h = nodes_h.reshape(n_batch, n_nodes, -1)
        g_h = g_h.reshape(n_batch, 1, -1)

        depot_cat_g = torch.cat((g_h, nodes_h[:, 0, :].unsqueeze(1)), dim=-1)
        # output nodes embedding should not include depot, refer to paper: https://www.sciencedirect.com/science/article/abs/pii/S0950705120304445
        nodes_h_no_depot = nodes_h[:, 1:, :]

        # get agent embedding
        agents_embedding = []
        for i in range(self.n_agent):
            agents_embedding.append(self.agents[i](depot_cat_g, nodes_h_no_depot))

        agent_embeddings = torch.cat(agents_embedding, dim=1)

        return agent_embeddings, nodes_h_no_depot


class Policy(nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_agent, key_size_embd, key_size_policy, val_size, clipping, dev):
        super(Policy, self).__init__()
        self.c = clipping
        self.key_size_policy = key_size_policy
        self.key_policy = nn.Linear(hid_chnl, self.key_size_policy).to(dev)
        self.q_policy = nn.Linear(val_size, self.key_size_policy).to(dev)

        # embed network
        self.embed = AgentAndNode_embedding(in_chnl=in_chnl, hid_chnl=hid_chnl, n_agent=n_agent,
                                            key_size=key_size_embd, value_size=val_size, dev=dev)

    def forward(self, batch_graph, n_nodes, n_batch):  # batch?

        agent_embeddings, nodes_h_no_depot = self.embed(batch_graph, n_nodes, n_batch)

        k_policy = self.key_policy(nodes_h_no_depot)
        q_policy = self.q_policy(agent_embeddings)
        u_policy = torch.matmul(q_policy, k_policy.transpose(-1, -2)) / math.sqrt(self.key_size_policy)
        imp = self.c * torch.tanh(u_policy)
        prob = F.softmax(imp, dim=-2)

        return prob


def action_sample(pi):
    dist = Categorical(pi.transpose(2, 1))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob


def get_log_prob(pi, action_int):
    dist = Categorical(pi.transpose(2, 1))
    log_prob = dist.log_prob(action_int)
    return log_prob


def get_cost(action, data, n_agent):
    subtour_max_lengths = [0 for _ in range(data.shape[0])]
    data = data * 1000 
    depot = data[:, 0, :].tolist()
    sub_tours = [[[] for _ in range(n_agent)] for _ in range(data.shape[0])]
    for i in range(data.shape[0]):
        for tour in sub_tours[i]:
            tour.append(depot[i])
        for n, m in zip(action.tolist()[i], data.tolist()[i][1:]):
            sub_tours[i][n].append(m)

    for k in range(data.shape[0]):
        for a in range(n_agent):
            instance = sub_tours[k][a]
            sub_tour_length = solve(instance)/1000
            if sub_tour_length >= subtour_max_lengths[k]:
                subtour_max_lengths[k] = sub_tour_length
    return subtour_max_lengths


def my_get_cost(action, data, n_agent):
    subtour_max_lengths = [0 for _ in range(data.shape[0])]
    routes_idx = [[] for _ in range(data.shape[0])]
    routes_coords = [[] for _ in range(data.shape[0])]
    all_length = [[] for _ in range(data.shape[0])]
    data = data * 1000  # why?
    depot = data[:, 0, :].tolist()
    sub_tours = [[[] for _ in range(n_agent)] for _ in range(data.shape[0])]
    city_indices = [[[0] for _ in range(n_agent)] for _ in range(data.shape[0])]
    for i in range(data.shape[0]):
        for tour in sub_tours[i]:
            tour.append(depot[i])
        for idx, (n, m) in enumerate(zip(action.tolist()[i], data.tolist()[i][1:])):
            sub_tours[i][n].append(m)
            city_indices[i][n].append(idx+1)

    for k in range(data.shape[0]):
        for a in range(n_agent):
            instance = sub_tours[k][a]
            route, sub_tour_length = my_solve(instance, city_indices[k][a])
            routes_idx[k].append(route)
            routes_coords[k].append(data[k,route,:]/1000)
            all_length[k].append(sub_tour_length)
            if sub_tour_length >= subtour_max_lengths[k]:
                subtour_max_lengths[k] = sub_tour_length
    return subtour_max_lengths, routes_idx, routes_coords, all_length

class Surrogate(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_hidden: int = 64, nonlin: str = 'relu', dev='cpu', **kwargs):
        super(Surrogate, self).__init__()
        nlist = dict(relu=nn.ReLU(), tanh=nn.Tanh(),
                     sigmoid=nn.Sigmoid(), softplus=nn.Softplus(), lrelu=nn.LeakyReLU(),
                     elu=nn.ELU())

        self.layer = nn.Linear(in_dim, n_hidden).to(dev)
        self.layer2 = nn.Linear(n_hidden, n_hidden).to(dev)
        self.out = nn.Linear(n_hidden, out_dim, bias=False).to(dev)
        self.nonlin = nlist[nonlin]

    def forward(self, x, **kwargs):
        x = self.layer(x)
        x = self.nonlin(x)
        x = self.layer2(x)
        x = self.nonlin(x)
        x = self.out(x)

        return x


if __name__ == '__main__':
    from torch_geometric.data import Data
    from torch_geometric.data import Batch

    dev = 'cpu'
    torch.manual_seed(2)

    n_agent = 4
    n_nodes = 6
    n_batch = 3
    # get batch graphs data list
    fea = torch.randint(low=0, high=100, size=[n_batch, n_nodes, 2]).to(torch.float)  # [batch, nodes, fea]
    adj = torch.ones([fea.shape[0], fea.shape[1], fea.shape[1]])
    data_list = [Data(x=fea[i], edge_index=torch.nonzero(adj[i]).t()) for i in range(fea.shape[0])]
    # generate batch graph
    batch_graph = Batch.from_data_list(data_list=data_list).to(dev)

    # test policy
    policy = Policy(in_chnl=fea.shape[-1], hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)

    pi = policy(batch_graph, n_nodes, n_batch)

    grad = torch.autograd.grad(pi.sum(), [param for param in policy.parameters()])

    action, log_prob = action_sample(pi)
    # print(log_prob)

    rewards = get_cost(action, fea, n_agent)

    print(rewards)
