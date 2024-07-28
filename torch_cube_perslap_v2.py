import numpy as np
import torch

def torch_chains_count(X): # X is a tensor of shape: (Channel, Row, Column)
    c, m, n = X.shape
    num_v = int((m+1) / 2 * (n+1) / 2)
    num_s = int((m-1) / 2 * (n-1) / 2)
    num_e = int(m * n - num_v - num_s)
    return c, num_v, num_e, num_s

def torch_bdry_idx_v2(X, channel, cell_coords, cell_dim, device="cpu"): # cell_coords are a tensor of cell coordinates, 
    c, m, n = X.shape
    if len(cell_coords) != 0:
        if cell_dim == 2: # Square cells
            x_0s = ((cell_coords[:, 0] * n + cell_coords[:, 1]) / 2 - 1).to(torch.float32)
            y_0s = ((n * cell_coords[:, 0] - n + cell_coords[:, 1] + 1) / 2 - 1).to(torch.float32)
            bdry_idxs = torch.transpose(torch.stack([y_0s, x_0s, x_0s + 1, y_0s + X.shape[2]]), 0, 1)
    
        elif cell_dim == 1: # Edge cells
            cell_coords_v = cell_coords[cell_coords[:, 0] % 2 == 1, :] # coordinates of vertical edges
            cell_coords_h = cell_coords[cell_coords[:, 1] % 2 == 1, :] # coordinates of horizontial edges
    
            v_x_0s = ((cell_coords_v[:, 0] - 1) * (n + 1) / 4 + cell_coords_v[:, 1] / 2).to(torch.float32)
            v_x_1s = (v_x_0s + (n + 1) / 2).to(int)
    
            # x_0 = p * (n + 1) / 4 + (q + 1) / 2 - 1
            # x_1 = x_0 + 1
            h_x_0s = (cell_coords_h[:, 0] * (n + 1) / 4 + (cell_coords_h[:, 1] + 1) / 2 - 1).to(torch.float32)
            h_x_1s = h_x_0s + 1
    
            bdry_idxs = torch.zeros(cell_coords.shape[0], 2).to(dtype=torch.float32, device=device)
    
            bdry_idxs[cell_coords[:, 0] % 2 == 1, :] = torch.transpose(torch.stack((v_x_0s, v_x_1s)), 0, 1)
            bdry_idxs[cell_coords[:, 1] % 2 == 1, :] = torch.transpose(torch.stack((h_x_0s, h_x_1s)), 0, 1)
    
        elif cell_dim == 0: # Vertices cells
            bdry_idxs = torch.tensor([]).to(torch.float32)

    elif len(cell_coords) == 0:
        bdry_idxs = torch.tensor([]).to(torch.float32)

    return bdry_idxs

def torch_cell_filt_v2(X, channel, filt, cell_dim=2): # Return the coordinates of filtered dim-dimensional cubes.
    X_idx = torch.where(X[channel, :, :] <= filt)
    X_idx = torch.stack(X_idx, dim=1).to(torch.float32)
    # X_idx = np.array([(X_idx[0]).tolist(), (X_idx[1]).tolist()]).transpose()
    # X_idx = X_idx.tolist()
    if len(X_idx) != 0:
        if cell_dim == 2:
            X_idx_odds = (X_idx[:, 0] % 2 == 1) & (X_idx[:, 1] % 2 == 1)
            cells = X_idx[X_idx_odds, :].to(torch.float32)
            
        elif cell_dim == 1:
            X_idx_sum = X_idx.sum(axis=1)
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

def torch_boundary_opr_v2(X, channel, filt, cell_dim=2, device="cpu"):
    c, num_v, num_e, num_s = torch_chains_count(X)
    chain_space_dim = torch_chains_count(X)[cell_dim]
    cell_coords = torch_cell_filt_v2(X, channel=channel, filt=filt, cell_dim=cell_dim)
    bdry_coords = torch_bdry_idx_v2(X=X, channel=channel, cell_coords=cell_coords, cell_dim=cell_dim, device=device)

    if cell_dim == 0:
        bdry_opr = torch.zeros(1, num_v).to(dtype=torch.float32, device=device)
    else:
        if len(bdry_coords) != 0:
            num_cells, bdry_cells = bdry_coords.shape
            which_column = torch.tensordot(torch.arange(num_cells), torch.ones(bdry_cells).to(int), dims=0).to(device)
            bdry_indices = torch.transpose(torch.stack([bdry_coords.flatten(), which_column.flatten()]), 0, 1).to(int)
            bdry_opr = torch.zeros([chain_space_dim, num_cells]).to(dtype=torch.float32, device=device)
            if cell_dim == 2:
                bdry_opr[bdry_indices[:, 0], bdry_indices[:, 1]] = torch.tensor([-1., 1., -1., 1.]).repeat(num_cells).to(dtype=torch.float32, device=device)
            elif cell_dim == 1:
                bdry_opr[bdry_indices[:, 0], bdry_indices[:, 1]] = torch.tensor([-1., 1.]).repeat(num_cells).to(dtype=torch.float32, device=device)
            
        else:
            bdry_opr = torch.zeros(chain_space_dim, 1).to(dtype=torch.float32, device=device)
    return bdry_opr # a tensor object


def torch_laplacian_up_v2(X, channel, filt, cell_dim=2, device="cpu"):
    if cell_dim == 0:
        bdry_opr_1 = torch_boundary_opr_v2(X, channel=channel, filt=filt, cell_dim=1, device=device) #.to(torch.float32)
        bdry_opr_1 = bdry_opr_1[~torch.all(bdry_opr_1 == 0, axis=1)]
        lap_up = bdry_opr_1 @ torch.transpose(bdry_opr_1, 0, 1) 
        
    elif cell_dim == 1:
        bdry_opr_2 = torch_boundary_opr_v2(X, channel=channel, filt=filt, cell_dim=2, device=device) #.to(torch.float32)
        bdry_opr_2 = bdry_opr_2[~torch.all(bdry_opr_2 == 0, axis=1)]
        lap_up = bdry_opr_2 @ torch.transpose(bdry_opr_2, 0, 1)
        
    elif cell_dim == 2:
        len_dim_2 = len(torch_cell_filt_v2(X, channel=channel, filt=filt, cell_dim=2))
        lap_up = torch.zeros([len_dim_2, len_dim_2]).to(dtype=torch.float32, device=device)

    if len(lap_up) == 0:
        lap_up = torch.tensor([0]).to(dtype=torch.float32, device=device)
    return lap_up


def torch_laplacian_down_v2(X, channel, filt, cell_dim=2):
    if cell_dim == 0:
        len_dim_0 = len(torch_cell_filt_v2(X, channel=channel, filt=filt, cell_dim=0))
        lap_down = torch.zeros([len_dim_0, len_dim_0])
        
    elif cell_dim == 1:
        bdry_opr_1 = torch_boundary_opr_v2(X, channel=channel, filt=filt, cell_dim=1)
        bdry_opr_1 = bdry_opr_1[~torch.all(bdry_opr_1 == 0, axis=1)]
        lap_down = torch.transpose(bdry_opr_1, 0, 1) @ bdry_opr_1
        
    elif cell_dim == 2:
        bdry_opr_2 = torch_boundary_opr_v2(X, channel=channel, filt=filt, cell_dim=2)
        bdry_opr_2 = bdry_opr_2[~torch.all(bdry_opr_2 == 0, axis=1)]
        lap_down = torch.transpose(bdry_opr_2, 0, 1) @ bdry_opr_2
    return lap_down

def torch_persistence_lap_up_v2(X, filt_1, filt_2, channel, cell_dim, device="cpu"):# filt_1 < filt_2 
    if (cell_dim == 1) or (cell_dim == 0):
        lap_up = torch_laplacian_up_v2(X=X, channel=channel, filt=filt_2, cell_dim=cell_dim, device=device) 
        cell_filt_2 = torch_cell_filt_v2(X=X, channel=channel, filt=filt_2, cell_dim=cell_dim)
        cell_filt_1 = torch_cell_filt_v2(X=X, channel=channel, filt=filt_1, cell_dim=cell_dim)

        # all_indices = cell_filt_1.unsqueeze(0) == cell_filt_2.unsqueeze(1)
        # filt_idx = all_indices[:, :, :].all(axis=2).any(axis=1)
        filt_idx = cell_idx_check(cell_filt_1, cell_filt_2)
        
        #perslap_up = lap_up[filt_idx, :][:, filt_idx] - lap_up[filt_idx, :][:, ~filt_idx] @ torch.linalg.pinv(lap_up[~filt_idx, :][:, ~filt_idx]) @ lap_up[~filt_idx, :][:, filt_idx]
        perslap_up = Schur_cmpl_v2(lap_up, filt_idx)

    elif cell_dim == 2:
        cell_filt_1 = torch_cell_filt_v2(X=X, channel=channel, filt=filt_1, cell_dim=cell_dim)
        lap_dim = len(cell_filt_1)
        perslap_up = torch.zeros(lap_dim, lap_dim).to(dtype=torch.float32, device=device)

    return perslap_up

def torch_persistence_laplacian_v2(X, filt_1, filt_2, channel, cell_dim):# filt_1 < filt_2 
    if (cell_dim == 1) or (cell_dim == 0):
        perslap_up = torch_persistence_lap_up_v2(X=X, filt_1=filt_1, filt_2=filt_2, channel=channel, cell_dim=cell_dim)
        if cell_dim == 1:
            perslap = perslap_up + torch_laplacian_down_v2(X=X, channel=channel, filt=filt_1, cell_dim=cell_dim)
        elif cell_dim == 0:
            perslap = perslap_up

    elif cell_dim == 2:
        perslap = torch_laplacian_down_v2(X=X, channel=channel, filt = filt_1, cell_dim=cell_dim)
    return perslap

def Schur_cmpl_v2(M, filt_idx):
    try:
        schur = M[filt_idx, :][:, filt_idx] - M[filt_idx, :][:, ~filt_idx] @ torch.linalg.pinv(M[~filt_idx, :][:, ~filt_idx]) @ M[~filt_idx, :][:, filt_idx]
    except:
        d = len(filt_idx)
        schur = torch.full([d, d], torch.inf)
    return schur

def cell_idx_check(cell_filt_1, cell_filt_2):
    all_indices = cell_filt_1.unsqueeze(0) == cell_filt_2.unsqueeze(1)
    filt_idx = all_indices[:, :, :].all(axis=2).any(axis=1)
    return filt_idx

def torch_persistence_lap_up_filtration_v2(X, channel, cell_dim, start=0., end=1.0, steps=5):
    # times = np.flip(np.linspace(start=start, stop=stop, num=num))
    # lap_up = torch_laplacian_up(X=X, channel=channel, filt=stop, cell_dim=cell_dim)
    # idx_stop = [torch_cell_idx(X, coord) for coord in torch_cell_filt(X, filt=stop, channel=channel, cell_dim=cell_dim)]
    times = torch.flip(torch.linspace(start=start, end=end, steps=steps), dims=[0])
    # times = 1 - torch.exp(-times) # nonlinear steps
    
    # lap_ups = []
    # lap_ups.append(lap_up)
    
    for t in range(len(times)):
        if t == 0:
            lap_up = torch_laplacian_up_v2(X=X, channel=channel, filt=end, cell_dim=cell_dim)
        else:
            idx_1 = torch_cell_filt_v2(X, filt=times[t], channel=channel, cell_dim=cell_dim)
            idx_2 = torch_cell_filt_v2(X, filt=times[t-1], channel=channel, cell_dim=cell_dim)
            if len(idx_1) != 0:
                filt_idx = cell_idx_check(idx_1, idx_2)
                try:
                    lap_up = Schur_cmpl_v2(lap_up, filt_idx)
                except:
                    lap_up = torch.tensor(tensor.inf)
            elif len(idx_1) == 0:
                lap_up = torch.tensor([0])
        yield lap_up

def torch_persistence_lap_down_filtration_v2(X, channel, cell_dim, start=0., end=1.0, steps=5):
    times = torch.flip(torch.linspace(start=start, end=end, steps=steps), dims=[0])
    # times = 1 - torch.exp(-times) # nonlinear steps
    
    for t in range(len(times)):
        lap_down = torch_laplacian_down_v2(X=X, channel=channel, filt=times[t], cell_dim=cell_dim)
        yield lap_down

def torch_persistence_laplacian_filtration_v2(X, channel, cell_dim, start=0., end=1.0, steps=5):
    lap_ups = torch_persistence_lap_up_filtration_v2(X=X, channel=channel, cell_dim=cell_dim,start=start, end=end, steps=steps)
    lap_downs = torch_persistence_lap_down_filtration_v2(X=X, channel=channel, cell_dim=cell_dim,start=start, end=end, steps=steps)

    laps = (lap_up + lap_down for lap_up, lap_down in zip(lap_ups, lap_downs))
    return laps









