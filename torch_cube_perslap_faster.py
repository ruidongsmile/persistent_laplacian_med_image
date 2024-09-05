# A class to find connected components w.r.t. a regular D.
import torch


class ColumnRearrangeRegular:
    def __init__(self, D):
        self.edges = None
        self.vertices = dict()
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
        self.edges = {i.item(): (self.D_nonzero[self.D_nonzero[:, 0] == i][:, 1]).tolist() for i in self.D_nonzero[:, 0].unique()}

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
