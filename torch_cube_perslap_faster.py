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
        self.hyperedges = None
        self.vertices = None
        self.D = D
        self.D_nonzero = D.nonzero()
        self.visited = None
        self.CC = []
        self.CC_reg = []
        self.D_reg = None
        self.edge_idx = None
        self.hyperedge_idx = None
    
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

        self.hyperedge_idx = (v_count == 1).nonzero().squeeze().tolist()
        if not isinstance(self.hyperedge_idx, list):
            self.hyperedge_idx = [self.hyperedge_idx]

        self.edge_idx = (v_count == 2).nonzero().squeeze().tolist()
        if not isinstance(self.edge_idx, list):
            self.edge_idx = [self.edge_idx]

        self.edges = {i: (self.D_nonzero[self.D_nonzero[:, 0] == i][:, 1]).tolist()
                      for i in self.edge_idx}
        self.hyperedges = {i: (self.D_nonzero[self.D_nonzero[:, 0] == i][:, 1]).tolist()
                           for i in self.hyperedge_idx}

        self.D_reg = self.D[self.edge_idx, :]

      

    def nbhd(self, u):
        nb = (self.D_reg[self.D_reg[:, u] != 0, :]).abs().sum(axis=0).nonzero().squeeze().tolist()
        if not isinstance(nb, list):
            nb = [nb]
        nb.remove(u)
        
        return nb

    def dfs(self, u, temp):
        self.visited[u] = True
        temp.append(u)
        nb = self.nbhd(u)
        
        for v in nb:
            if not self.visited[v]:
                temp = self.dfs(v, temp)

        return temp

    def connected_components(self):
        v_reg = self.D_reg.abs().sum(axis=0).nonzero().squeeze().tolist()

        for v in v_reg:
            if not self.visited[v]:
                temp = []
                self.CC.append(self.dfs(v, temp))

        if len(self.hyperedge_idx) != 0:
            cmpt_reg_idx = []
            hyper_v = set(sum(self.hyperedges.values(), []))

            for i in range(len(self.CC)):
                if len(hyper_v & set(self.CC[i])) == 0:
                    cmpt_reg_idx.append(i)
            self.CC_reg = [self.CC[i] for i in cmpt_reg_idx]
                    

        # if len(self.hyperedges) != 0:
        #     temp = []
        #     # if len(list(self.hyperedges.values())) == 1:
        #     #     hyperedges = list(self.hyperedges.values())[0]
        #     # elif len(list(self.hyperedges.values())) > 1:
        #     #     hyperedges = torch.tensor(list(self.hyperedges.values())).squeeze().tolist()

        #     hyperedges = torch.tensor(list(self.hyperedges.values())).squeeze().tolist()
        #     if not isinstance(hyperedges, list):
        #         hyperedges = [hyperedges]

            # for i in range(len(self.CC)):
            #     if len(set(hyperedges) & set(self.CC[i])) == 0:
            #         temp.append(i)
            # self.CC = [self.CC[i] for i in temp]

    def find_zero_column(self):
        self.zero_column = (self.D.abs().sum(axis=0) == 0).nonzero().squeeze().tolist()
        if not isinstance(self.zero_column, list):
            self.zero_column = [self.zero_column]               


# class ColumnRearrangeGeneral:
#     def __init__(self, D):
#         self.zero_column = None
#         self.edges = dict()
#         self.hyperedges = dict()
#         self.vertices = None
#         self.D = D
#         self.D_nonzero = D.nonzero()
#         self.visited = None
#         self.CC = []
#         # self.count = 0
#         # self.label = torch.zeros(D.shape[1])

#     def find_vertices(self):
#         self.vertices = (torch.nonzero(self.D.abs().sum(axis=0))).squeeze().tolist()

#     def vertices_visited(self):
#         # self.visited = torch.zeros(len(self.vertices), dtype=torch.bool)
#         self.visited = {i: False for i in self.vertices}

#     def find_edges(self):
#         v_count = self.D.abs().sum(axis=1)

#         # if len((v_count == 1).nonzero()) != 1:
#         #     hyperedge_idx = (v_count == 1).nonzero().squeeze().tolist()
#         # elif len((v_count == 1).nonzero()) == 1:
#         #     hyperedge_idx = [(v_count == 1).nonzero().squeeze().tolist()]

#         hyperedge_idx = (v_count == 1).nonzero().squeeze().tolist()
#         if not isinstance(hyperedge_idx, list):
#             hyperedge_idx = [hyperedge_idx]

#         # if len((v_count == 2).nonzero()) != 1:
#         #     edge_idx = (v_count == 2).nonzero().squeeze().tolist()
#         # elif len((v_count == 2).nonzero()) == 1:
#         #     edge_idx = [(v_count == 2).nonzero().squeeze().tolist()]

#         edge_idx = (v_count == 2).nonzero().squeeze().tolist()
#         if not isinstance(edge_idx, list):
#             edge_idx = [edge_idx]

#         self.edges = {i: (self.D_nonzero[self.D_nonzero[:, 0] == i][:, 1]).tolist()
#                       for i in edge_idx}
#         self.hyperedges = {i: (self.D_nonzero[self.D_nonzero[:, 0] == i][:, 1]).tolist()
#                            for i in hyperedge_idx}

#     def dfs(self, u, temp):
#         self.visited[u] = True

#         temp.append(u)

#         for edge in self.edges.values():
#             if u in edge:
#                 v = edge[edge != u]
#                 if not self.visited[v]:
#                     temp = self.dfs(v, temp)
#         return temp

#     def connected_components(self):

#         for v in self.vertices:
#             if not self.visited[v]:
#                 temp = []
#                 self.CC.append(self.dfs(v, temp))

#         if len(self.hyperedges) != 0:
#             temp = []
#             # if len(list(self.hyperedges.values())) == 1:
#             #     hyperedges = list(self.hyperedges.values())[0]
#             # elif len(list(self.hyperedges.values())) > 1:
#             #     hyperedges = torch.tensor(list(self.hyperedges.values())).squeeze().tolist()

#             hyperedges = torch.tensor(list(self.hyperedges.values())).squeeze().tolist()
#             if not isinstance(hyperedges, list):
#                 hyperedges = [hyperedges]

#             for i in range(len(self.CC)):
#                 if len(set(hyperedges) & set(self.CC[i])) == 0:
#                     temp.append(i)
#             self.CC = [self.CC[i] for i in temp]

#     def find_zero_column(self):
#         self.zero_column = (self.D.abs().sum(axis=0) == 0).nonzero().squeeze().tolist()
#         if not isinstance(self.zero_column, list):
#             self.zero_column = [self.zero_column]


class UPPERSLAP:
    def __init__(self, X, channel=None, filt_1=None, filt_2=None, device="cpu"):
        # cell_dim is the dim of up persistent Laplacian
        self.X = X
        self.channel = channel
        self.filt_1 = filt_1
        self.filt_2 = filt_2
        self.device = device

        self.filt_idx = None

    def torch_chains_count(self):  # X is a tensor of shape: (Channel, Row, Column)
        c, m, n = self.X.shape
        num_v = int((m + 1) / 2 * (n + 1) / 2)
        num_s = int((m - 1) / 2 * (n - 1) / 2)
        num_e = int(m * n - num_v - num_s)
        return c, num_v, num_e, num_s

    def torch_cell_filt(self, filt, cell_dim):  # Return the coordinates of filtered dim-dimensional cubes.
        global cells
        X_idx = torch.where(self.X[self.channel, :, :] <= filt)
        X_idx = torch.stack(X_idx, dim=1).to(torch.float32)

        # X_idx = np.array([(X_idx[0]).tolist(), (X_idx[1]).tolist()]).transpose()
        # X_idx = X_idx.tolist()
        if len(X_idx) != 0:
            if cell_dim == 2:
                X_idx_odds = (X_idx[:, 0] % 2 == 1) & (X_idx[:, 1] % 2 == 1)
                cells = X_idx[X_idx_odds, :].to(torch.float32)

            elif cell_dim == 1:
                X_idx_sum = X_idx.sum(dim=1)
                cells = X_idx[X_idx_sum % 2 == 1, :].to(torch.float32)

            elif cell_dim == 0:
                X_idx_evens = (X_idx[:, 0] % 2 == 0) & (X_idx[:, 1] % 2 == 0)
                cells = X_idx[X_idx_evens, :].to(torch.float32)

            if len(cells) != 0:
                return cells
            elif len(cells) == 0:
                return torch.tensor([]).to(torch.float32)

        elif len(X_idx) == 0:
            return torch.tensor([]).to(torch.float32)

    def torch_bdry_idx(self, cell_coords, cell_dim):  # cell_coords are a tensor of cell coordinates,
        global bdry_idxs
        c, m, n = self.X.shape
        if len(cell_coords) != 0:
            if cell_dim == 2:  # Square cells
                x_0s = ((cell_coords[:, 0] * n + cell_coords[:, 1]) / 2 - 1).to(torch.float32)
                y_0s = ((n * cell_coords[:, 0] - n + cell_coords[:, 1] + 1) / 2 - 1).to(torch.float32)
                bdry_idxs = torch.transpose(torch.stack([y_0s, x_0s, x_0s + 1, y_0s + self.X.shape[2]]), 0, 1)

            elif cell_dim == 1:  # Edge cells
                cell_coords_v = cell_coords[cell_coords[:, 0] % 2 == 1, :]  # coordinates of vertical edges
                cell_coords_h = cell_coords[cell_coords[:, 1] % 2 == 1, :]  # coordinates of horizontial edges

                v_x_0s = ((cell_coords_v[:, 0] - 1) * (n + 1) / 4 + cell_coords_v[:, 1] / 2).to(torch.float32)
                v_x_1s = (v_x_0s + (n + 1) / 2).to(int)

                # x_0 = p * (n + 1) / 4 + (q + 1) / 2 - 1
                # x_1 = x_0 + 1
                h_x_0s = (cell_coords_h[:, 0] * (n + 1) / 4 + (cell_coords_h[:, 1] + 1) / 2 - 1).to(torch.float32)
                h_x_1s = h_x_0s + 1

                bdry_idxs = torch.zeros(cell_coords.shape[0], 2).to(dtype=torch.float32, device=self.device)

                bdry_idxs[cell_coords[:, 0] % 2 == 1, :] = torch.transpose(torch.stack((v_x_0s, v_x_1s)), 0, 1)
                bdry_idxs[cell_coords[:, 1] % 2 == 1, :] = torch.transpose(torch.stack((h_x_0s, h_x_1s)), 0, 1)

            elif cell_dim == 0:  # Vertices cells
                bdry_idxs = torch.tensor([]).to(torch.float32)

        elif len(cell_coords) == 0:
            bdry_idxs = torch.tensor([]).to(torch.float32)

        return bdry_idxs

    def find_boundary_opr(self):
        # c, num_v, num_e, num_s = self.torch_chains_count()
        chain_space_dim = self.torch_chains_count()[2]
        cell_coords = self.torch_cell_filt(filt=self.filt_2, cell_dim=2)
        bdry_coords = self.torch_bdry_idx(cell_coords=cell_coords, cell_dim=2)

        if len(bdry_coords) != 0:
            num_cells, bdry_cells = bdry_coords.shape
            which_column = torch.tensordot(torch.arange(num_cells), torch.ones(bdry_cells).to(int), dims=0).to(
                self.device)
            bdry_indices = torch.transpose(torch.stack([bdry_coords.flatten(), which_column.flatten()]), 0, 1).to(int)
            bdry_opr = torch.zeros([chain_space_dim, num_cells]).to(dtype=torch.float32, device=self.device)

            bdry_opr[bdry_indices[:, 0], bdry_indices[:, 1]] = torch.tensor([-1., 1., -1., 1.]).repeat(num_cells).to(
                dtype=torch.float32, device=self.device)
            bdry_opr = bdry_opr[~torch.all(bdry_opr == 0, dim=1)]

        else:
            bdry_opr = torch.zeros(chain_space_dim, 1).to(dtype=torch.float32, device=self.device)
        # return bdry_opr # a tensor object
        self.bdry_opr = bdry_opr

    def find_cell_idx_check(self):

        cell_filt_2 = self.torch_cell_filt(filt=self.filt_2, cell_dim=1)
        cell_filt_1 = self.torch_cell_filt(filt=self.filt_1, cell_dim=1)

        try:
            all_indices = cell_filt_1.unsqueeze(0) == cell_filt_2.unsqueeze(1)
            filt_idx = all_indices[:, :, :].all(dim=2).any(dim=1)
        except (RuntimeError, IndexError):
            filt_idx = torch.zeros(cell_filt_2.shape[0], dtype=torch.bool)

        # all_indices = cell_filt_1.unsqueeze(0) == cell_filt_2.unsqueeze(1)
        # filt_idx = all_indices[:, :, :].all(dim=2).any(dim=1)
        self.filt_idx = filt_idx

    def find_D(self):
        # bdry_opr = bdry_opr[~torch.all(bdry_opr == 0, axis=1)]

        # filt_idx = self.cell_idx_check()

        if self.filt_idx.numel() != 0:
            D_mtx = self.bdry_opr[~self.filt_idx, :]
        else:
            D_mtx = self.bdry_opr

        self.D_mtx = D_mtx

    def find_col_rearrange(self):
        graph_D = ColumnRearrangeGeneral(self.D_mtx)
        graph_D.find_vertices()
        graph_D.find_edges()
        graph_D.vertices_visited()
        graph_D.connected_components()
        graph_D.find_zero_column()

        self.graph_D = graph_D

    def find_rel_bdry_opr(self):
        if self.filt_idx.numel() != 0 and self.filt_idx.nonzero().numel() != 0:
            rel_bdry_opr = self.bdry_opr[self.filt_idx, :][:, self.graph_D.zero_column]
        else:
            rel_bdry_opr = torch.tensor([[0.]])

        for i in range(len(self.graph_D.CC_reg) - 1, -1, -1):
            cc = self.bdry_opr[self.filt_idx, :][:, self.graph_D.CC_reg[i]].sum(dim=1).unsqueeze(1)
            rel_bdry_opr = torch.cat((cc, rel_bdry_opr), 1)

        self.rel_bdry_opr = rel_bdry_opr

    def find_diag(self):
        _, cc_count = self.rel_bdry_opr.shape
        if cc_count != 0:
            diag = torch.eye(cc_count)
        else:
            diag = torch.eye(1)

        for i in range(len(self.graph_D.CC_reg)):
            diag[i, i] = 1 / len(self.graph_D.CC_reg[i])
        self.diag = diag

    def find_diag_sqrt(self):
        _, cc_count = self.rel_bdry_opr.shape
        if cc_count != 0:
            diag_sqrt = torch.eye(cc_count)
        else:
            diag_sqrt = torch.eye(1)

        for i in range(len(self.graph_D.CC_reg)):
            diag_sqrt[i, i] = 1 / torch.sqrt(torch.tensor(len(self.graph_D.CC_reg[i])))
        self.diag_sqrt = diag_sqrt

    def up_persistent_Laplacian(self):
        self.find_boundary_opr()
        self.find_cell_idx_check()
        self.find_D()
        self.find_col_rearrange()
        self.find_rel_bdry_opr()
        self.find_diag()

        up_lap = self.rel_bdry_opr @ self.diag @ torch.transpose(self.rel_bdry_opr, 0, 1)
        return up_lap

    def dual_up_persistent_Laplacian(self):
        self.find_boundary_opr()
        self.find_cell_idx_check()
        self.find_D()
        self.find_col_rearrange()
        self.find_rel_bdry_opr()
        self.find_diag_sqrt()

        dual_up_lap = self.diag_sqrt @ torch.transpose(self.rel_bdry_opr, 0, 1) @ self.rel_bdry_opr @ self.diag_sqrt
        return dual_up_lap
