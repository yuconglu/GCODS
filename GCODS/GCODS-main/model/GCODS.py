import torch
import torch.nn as nn
import torch.nn.functional as F
from afnonet import Block as AFNOBlock
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear
import timm
from torchdiffeq import odeint_adjoint as odeint
from .mcdp_layers import gcn, MCDP_Net

class CliffordReLU(nn.Module):
    def forward(self, x):
        return F.relu(x)


class HGCE(nn.Module):
    def __init__(self, args):
        super(HGCE, self).__init__()
        self.args = args
        self.lag = args.lag
        self.num_node = args.num_nodes
        self.H = args.height
        self.W = args.width

        self.num_multivector_components = 4
        clifford_output_dim = (args.hidden_dim // 2 // 4) * 4
        clifford_out_channels = clifford_output_dim // self.num_multivector_components
        
        self.clifford_enc = nn.Sequential(
            CliffordLinear(g=[1,1], in_channels=args.lag, out_channels=clifford_out_channels),
            CliffordReLU()
        )
        self.clifford_output_dim = clifford_output_dim
        
        self.num_scalar_features = args.input_dim - self.num_multivector_components
        scalar_input_channels = self.num_scalar_features * self.lag
        
        self.cnn_enc = timm.create_model(
            'mobilenetv3_small_100',
            pretrained=False,
            in_chans=scalar_input_channels,
            features_only=True
        )
        
        cnn_feature_info = self.cnn_enc.feature_info[-1]
        cnn_out_channels = cnn_feature_info['num_chs']
        scalar_output_dim = args.hidden_dim - self.clifford_output_dim
        
        self.feature_align = nn.Sequential(
            nn.Conv2d(in_channels=cnn_out_channels, out_channels=scalar_output_dim, kernel_size=1),
            nn.ReLU()
        )
        self.scalar_output_dim = scalar_output_dim
        self.scalar_norm = nn.LayerNorm(scalar_output_dim)

    def forward(self, X):
        multivector_indices = [0, 7, 8, 2]
        scalar_indices = [1, 3, 4, 5, 6, 9, 10, 11]
        X_multivector = X[..., multivector_indices]
        X_scalar = X[..., scalar_indices]

        X_mv_permuted = X_multivector.permute(0, 2, 1, 3)
        X_mv_reshaped = X_mv_permuted.reshape(-1, self.lag, self.num_multivector_components)
        H_mv_raw = self.clifford_enc(X_mv_reshaped)
        H_mv = H_mv_raw.reshape(X.shape[0], self.num_node, self.clifford_output_dim)
        
        X_scalar_grid = X_scalar.permute(0, 2, 1, 3).reshape(
            X.shape[0], self.num_node, -1
        ).view(
            X.shape[0], self.H, self.W, -1
        ).permute(0, 3, 1, 2).contiguous()
        
        X_scalar_grid_resized = F.interpolate(X_scalar_grid, size=(224, 224), mode='bilinear', align_corners=False)
        cnn_features = self.cnn_enc(X_scalar_grid_resized)[-1]
        aligned_features = self.feature_align(cnn_features)
        H_scalar_grid = F.interpolate(aligned_features, size=(self.H, self.W), mode='bilinear', align_corners=False)
        H_scalar_unnormed = H_scalar_grid.permute(0, 2, 3, 1).reshape(X.shape[0], self.num_node, self.scalar_output_dim)
        H_scalar = self.scalar_norm(H_scalar_unnormed)
        
        X_encoded = torch.cat([H_mv, H_scalar], dim=-1)
        return X_encoded


class HybridODEFunc(nn.Module):
    def __init__(self, edge_index, model_type, num_nodes, latent_dim, nhidden, 
                 alpha, embed_dim, height, width, num_afno_blocks, 
                 use_dgcrn=False, gcn_depth=2, dgcrn_dropout=0.3, 
                 dgcrn_alpha=0.05, dgcrn_beta=0.95, dgcrn_gamma=0.95, 
                 node_dim=40, hyperGNN_dim=16):
        super(HybridODEFunc, self).__init__()

        self.model_type = model_type
        self.use_dgcrn = use_dgcrn
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.H = height
        self.W = width

        if self.use_dgcrn:
            predefined_A = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
            predefined_A[edge_index[0], edge_index[1]] = 1.0
            predefined_A = predefined_A + torch.eye(num_nodes).to(edge_index.device)
            predefined_A = predefined_A / (torch.unsqueeze(predefined_A.sum(-1), -1) + 1e-7)

            self.mcdp_net = MCDP_Net(
                num_nodes=num_nodes,
                gcn_depth=gcn_depth,
                dropout=dgcrn_dropout,
                alpha=dgcrn_alpha, 
                beta=dgcrn_beta, 
                gamma=dgcrn_gamma,
                node_dim=node_dim,
                rnn_size=latent_dim,
                in_dim=latent_dim,
                hyperGNN_dim=hyperGNN_dim,
                predefined_A=predefined_A
            )
        else:
            self.alpha = alpha
            self.embed_dim = embed_dim
            self.adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
            self.A1 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.A2 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            if self.model_type == 'k':
                self.coeff = nn.Parameter(torch.tensor(1, dtype=torch.float32, requires_grad=True))

        self.afno_residual_model = AFNOBlock(
            dim=latent_dim,
            num_blocks=num_afno_blocks
        )
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1

        B, N, C = x.shape
        x_grid = x.view(B, self.H, self.W, C)
        F_residual_grid = self.afno_residual_model(x_grid)
        F_residual = F_residual_grid.view(B, N, C)

        if self.use_dgcrn:
            A_dynamic = self.mcdp_net(x, x)
            A_dynamic = A_dynamic + torch.eye(self.num_nodes, device=x.device).unsqueeze(0)
            D_for_norm = A_dynamic.sum(dim=-1, keepdim=True)
            A_dynamic = A_dynamic / (D_for_norm + 1e-7)
            D_out = torch.diag_embed(A_dynamic.sum(dim=-1))
            L = D_out - A_dynamic.transpose(1, 2)
            F_physical = -torch.matmul(L, x)
        else:
            if self.model_type == 'diff':
                A_out = F.relu(torch.tanh(self.alpha * torch.mm(self.A1, self.A1.T)))
            elif self.model_type == 'adv':
                A_out = F.relu(torch.tanh(self.alpha * (torch.mm(self.A1, self.A2.T) - torch.mm(self.A2, self.A1.T))))
            else: 
                A_out = F.relu(torch.tanh(self.alpha * torch.mm(self.A1, self.A2.T)))

            if self.model_type == 'pre':
                indices = self.edge_index
                values = torch.ones(indices.size(1), device=indices.device)
                size = (self.A1.size(0), self.A1.size(0))
                A_out = torch.sparse.FloatTensor(indices, values, size).to_dense()
            elif self.model_type == 'k':
                indices = self.edge_index
                values = torch.ones(indices.size(1), device=indices.device)
                size = (self.A1.size(0), self.A1.size(0))
                self.adj = torch.sparse.FloatTensor(indices, values, size).to_dense()
                A_out = self.coeff * self.adj
            else:
                indices = self.edge_index
                values = torch.ones(indices.size(1), device=indices.device)
                size = (self.A1.size(0), self.A1.size(0))
                self.adj = torch.sparse.FloatTensor(indices, values, size).to_dense()
                A_out = A_out * self.adj

            D_out = torch.diag(A_out.sum(1))
            L = D_out - A_out.T
            F_physical = -torch.matmul(L, x)

        if self.model_type == 'OnlyUncertainty':
            return F_residual
        elif self.model_type == 'WithoutUncertainty':
            return F_physical
        elif self.model_type == 'Full':
            return F_residual + F_physical
        else:
            raise ValueError(f"Unknown model_type: '{self.model_type}'")


class ODEBlock(nn.Module):
    def __init__(self,
                edge_index,
                model_type,
                num_nodes, 
                in_features,
                horizon,
                alpha,
                embed_dim,
                height,
                width,
                num_afno_blocks,
                use_dgcrn, gcn_depth, dgcrn_dropout, 
                dgcrn_alpha, dgcrn_beta, dgcrn_gamma, 
                node_dim, hyperGNN_dim,
                method='dopri5',
                adjoint=True,
                atol=1e-3,
                rtol=1e-3,
                time_step_hours=6,
                use_continuous_time=False):
        super(ODEBlock, self).__init__()

        self.odefunc = HybridODEFunc(edge_index, model_type, num_nodes, in_features, in_features*4, 
                               alpha, embed_dim, height, width, num_afno_blocks,
                               use_dgcrn, gcn_depth, dgcrn_dropout,
                               dgcrn_alpha, dgcrn_beta, dgcrn_gamma,
                               node_dim, hyperGNN_dim)

        self.horizon = horizon
        self.embed_dim = embed_dim
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.adjoint = adjoint
        self.time_step_hours = time_step_hours
        self.use_continuous_time = use_continuous_time
        self.time_scaling_factor = 0.01

    def forward(self, x, eval_times=None):
        if eval_times is None:  
            horizon_indices = torch.linspace(0, self.horizon, self.horizon+1).float().to(x.device)
        else:
            horizon_indices = eval_times.type_as(x).to(x.device)

        if self.use_continuous_time:
            time_hours = horizon_indices * self.time_step_hours
            init_hour = time_hours[0].item()
            final_hour = time_hours[-1].item()
            num_continuous_points = int(final_hour - init_hour) + 1
            continuous_hours = torch.linspace(
                init_hour, final_hour, 
                steps=num_continuous_points
            ).float().to(x.device)
            integration_time = continuous_hours * self.time_scaling_factor
        else:
            integration_time = horizon_indices

        if self.method == 'dopri5':
            out = odeint(self.odefunc, x, integration_time,
                                rtol=self.rtol, atol=self.atol, method=self.method,
                                options={'max_num_steps': 1000})
        else:
            out = odeint(self.odefunc, x, integration_time,
                                rtol=self.rtol, atol=self.atol, method=self.method)        

        if self.use_continuous_time:
            sample_indices = torch.arange(0, len(continuous_hours), self.time_step_hours).long()
            out = out[sample_indices]
        
        return out[1:]


class GCODS(nn.Module):
    def __init__(self, args, edge_index):
        super(GCODS, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.lag = args.lag
        self.horizon = args.horizon
        self.alpha = args.alpha
        self.dropout_rate = args.dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

        if args.use_cnn_encoder:
            print("Initializing GCODS with HGCE.")
            self.encoder = HGCE(args)
        else:
            raise ValueError("GCODS requires HGCE encoder. Set use_cnn_encoder=True in config.")

        if args.use_dgcrn:
            gcn_depth = args.gcn_depth
            dgcrn_dropout = args.dgcrn_dropout
            dgcrn_alpha = args.dgcrn_alpha
            dgcrn_beta = args.dgcrn_beta
            dgcrn_gamma = args.dgcrn_gamma
            node_dim = args.node_dim
            hyperGNN_dim = args.hyperGNN_dim
        else:
            gcn_depth = 2
            dgcrn_dropout = 0.0
            dgcrn_alpha = 0.0
            dgcrn_beta = 0.0
            dgcrn_gamma = 0.0
            node_dim = 1
            hyperGNN_dim = 1

        time_step_hours = getattr(args, 'time_step_hours', 6)
        use_continuous_time = getattr(args, 'use_continuous_time', False)
        
        self.ode_block = ODEBlock(
            edge_index, args.model_type, args.num_nodes, args.hidden_dim, 
            args.horizon, args.alpha, args.embed_dim, args.height, args.width, 
            args.num_afno_blocks, args.use_dgcrn, gcn_depth, 
            dgcrn_dropout, dgcrn_alpha, dgcrn_beta, 
            dgcrn_gamma, node_dim, hyperGNN_dim,
            time_step_hours=time_step_hours,
            use_continuous_time=use_continuous_time
        )
        
        self.dec = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(args.hidden_dim, args.output_dim)
        )

    def forward(self, X, targets, teacher_forcing_ratio=0.5, apply_r_drop=False):
        X_encoded = self.encoder(X)
        ode_out = self.ode_block(X_encoded)
        ode_out = ode_out.permute(1,0,2,3)

        ode_out_dropped = self.dropout(ode_out)
        out1 = self.dec(ode_out_dropped)

        if apply_r_drop:
            ode_out_dropped_2 = self.dropout(ode_out)
            out2 = self.dec(ode_out_dropped_2)
            return out1, out2
        else:
            return out1
