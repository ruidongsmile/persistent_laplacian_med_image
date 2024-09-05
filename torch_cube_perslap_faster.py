# A class to find connected components w.r.t. a regular D.
import torch


class ColumnRearrangeRegular:
    def __init__(self, D):
        self.edges = None
        self.vertices = None
        self.D = D
        self.D_nonzero = D.nonzero()
        self.visited = None
        self.CC = []
        # self.count = 0
        # self.label = torch.zeros(D.shape[1])

    def find_vertices(self):
        self.vertices = (torch.nonzero(self.D.abs().sum(axis=0))).squeeze().tolist()

    def vertices_visited(self):
        # self.visited = torch.zeros(len(self.vertices), dtype=torch.bool)
        self.visited = {i: False for i in self.vertices}

    def find_edges(self):
        self.edges = {i.item(): (self.D_nonzero[self.D_nonzero[:, 0] == i][:, 1]).tolist()
                      for i in self.D_nonzero[:, 0].unique()}

    def dfs(self, u, temp):
        self.visited[u] = True
        
        temp.append(u)

        for edge in self.edges.values():
            if u in edge:
                v = edge[edge != u]
                if not self.visited[v]:
                    temp = self.dfs(v, temp)
        return temp

    def connected_cmpts(self):

        for v in self.vertices:
            if not self.visited[v]:
                temp = []
                self.CC.append(self.dfs(v, temp))


class ColumnRearrangeGeneral:
    def __init__(self, D):
        self.zero_column = None
        self.edges = dict()
        self.hyperedges = dict()
        self.vertices = None
        self.D = D
        self.D_nonzero = D.nonzero()
        self.visited = None
        self.CC = []
        # self.count = 0
        # self.label = torch.zeros(D.shape[1])

    def find_vertices(self):
        self.vertices = (torch.nonzero(self.D.abs().sum(axis=0))).squeeze().tolist()

    def vertices_visited(self):
        # self.visited = torch.zeros(len(self.vertices), dtype=torch.bool)
        self.visited = {i: False for i in self.vertices}

    def find_edges(self):
        v_count = self.D.abs().sum(axis=1)

        # if len((v_count == 1).nonzero()) != 1:
        #     hyperedge_idx = (v_count == 1).nonzero().squeeze().tolist()
        # elif len((v_count == 1).nonzero()) == 1:
        #     hyperedge_idx = [(v_count == 1).nonzero().squeeze().tolist()]

        hyperedge_idx = (v_count == 1).nonzero().squeeze().tolist()
        if not isinstance(hyperedge_idx, list):
            hyperedge_idx = [hyperedge_idx]

        # if len((v_count == 2).nonzero()) != 1:
        #     edge_idx = (v_count == 2).nonzero().squeeze().tolist()
        # elif len((v_count == 2).nonzero()) == 1:
        #     edge_idx = [(v_count == 2).nonzero().squeeze().tolist()]

        edge_idx = (v_count == 2).nonzero().squeeze().tolist()
        if not isinstance(edge_idx, list):
            edge_idx = [edge_idx]

        self.edges = {i: (self.D_nonzero[self.D_nonzero[:, 0] == i][:, 1]).tolist()
                      for i in edge_idx}
        self.hyperedges = {i: (self.D_nonzero[self.D_nonzero[:, 0] == i][:, 1]).tolist()
                           for i in hyperedge_idx}

    def dfs(self, u, temp):
        self.visited[u] = True

        temp.append(u)

        for edge in self.edges.values():
            if u in edge:
                v = edge[edge != u]
                if not self.visited[v]:
                    temp = self.dfs(v, temp)
        return temp

    def connected_components(self):

        for v in self.vertices:
            if not self.visited[v]:
                temp = []
                self.CC.append(self.dfs(v, temp))

        if len(self.hyperedges) != 0:
            temp = []
            # if len(list(self.hyperedges.values())) == 1:
            #     hyperedges = list(self.hyperedges.values())[0]
            # elif len(list(self.hyperedges.values())) > 1:
            #     hyperedges = torch.tensor(list(self.hyperedges.values())).squeeze().tolist()

            hyperedges = torch.tensor(list(self.hyperedges.values())).squeeze().tolist()
            if not isinstance(hyperedges, list):
                hyperedges = [hyperedges]

            for i in range(len(self.CC)):
                if len(set(hyperedges) & set(self.CC[i])) == 0:
                    temp.append(i)
            self.CC = [self.CC[i] for i in temp]

    def find_zero_column(self):
        self.zero_column = (self.D.abs().sum(axis=0) == 0).nonzero().squeeze().tolist()
        if not isinstance(self.zero_column, list):
            self.zero_column = [self.zero_column]