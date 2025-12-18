
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from pytorch3d.ops import knn_points,knn_gather
import einops
import math
from sparsemax import Sparsemax
from torch_geometric.data import Data
from torch_geometric.nn import PointGNNConv, SGConv
from torch_geometric.nn import knn_graph

class GraphFormer_positional_embedding(nn.Module):
    """
    with simple graph convolution
    https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.nn.conv.SGConv.html#torch_geometric.nn.conv.SGConv
    and https://arxiv.org/abs/1902.07153
    """
    def __init__(self, in_dim):
        super(GraphFormer_positional_embedding, self).__init__()

        self.att_layer = RNSA(in_dim)

        self.lnorm_att = nn.LayerNorm(in_dim)

        self.positional_embedding = nn.Sequential(
                nn.Linear(3, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, in_dim)
                )

        self.sgconv1 = SGConv(in_dim, in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.sgconv2 = SGConv(in_dim, in_dim)
        self.dropout = nn.Dropout(0.5)
        
        self.lnorm_ff = nn.LayerNorm(in_dim)

    def create_graph_data(self, coordinates, features):
        """
        Create a PyTorch Geometric Data object from coordinates and features using k-nearest neighbors.

        Parameters:
        - coordinates: Tensor of shape [n, 3], the XYZ coordinates of the nodes.
        - features: Tensor of shape [n, d], the features of the nodes.

        Returns:
        - data: A PyTorch Geometric Data object with node features and edge index [2, n x k].
        """
        edge_index = knn_graph(coordinates.float(), k=min(32, coordinates.size(0)), loop=False)
        # Create the PyTorch Geometric data object
        data = Data(x=features, edge_index=edge_index)
        
        return data

    def get_batch_splits(self, me_tensor):
        _, counts = torch.unique(me_tensor.C[:, 0], return_counts=True)
        splits_C = torch.split(me_tensor.C, counts.tolist(), dim=0)
        splits_C = [s[:, 1:] for s in splits_C]
        splits_F = torch.split(me_tensor.F, counts.tolist())

        return splits_C, splits_F
    
    def forward(self, x):
        ## Att
        x_att = self.att_layer(x)
        # x_att = x_att + x
        
        Batch_split_coordinates, Batch_split_features = self.get_batch_splits(x_att)
        batch = []
        for i in range (len(Batch_split_features)):
            
            features = Batch_split_features[i]
            coordinates = Batch_split_coordinates[i].float()
            coordinates_feature = self.positional_embedding(coordinates)
            x_input_ff = self.lnorm_att(features)
            x_input_ff_pe = x_input_ff + coordinates_feature
            ## FF
            graph_data = self.create_graph_data(coordinates,x_input_ff_pe)
            x_ff = self.sgconv1(graph_data.x, graph_data.edge_index)
            x_ff = self.relu(x_ff)
            x_ff = self.sgconv2(x_ff, graph_data.edge_index)
            x_ff = self.dropout(x_ff)

            x_out = x_ff + x_input_ff
            x_total = self.lnorm_ff(x_out)

            batch.append(x_total)

        x_total_batch = torch.cat(batch, dim=0)
        x_transformer = ME.SparseTensor(features=x_total_batch, tensor_stride=x_att.tensor_stride, 
                                            device = x_att.device, coordinate_map_key=x_att.coordinate_map_key, coordinate_manager=x_att.coordinate_manager)

        return x_transformer

class GraphFormer_PE(nn.Module):
    """
    with simple graph convolution
    https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.nn.conv.SGConv.html#torch_geometric.nn.conv.SGConv
    and https://arxiv.org/abs/1902.07153
    """
    def __init__(self, in_dim):
        super(GraphFormer_PE, self).__init__()

        self.att_layer = RNSA(in_dim)

        self.lnorm_att = nn.LayerNorm(in_dim)

        self.sgconv1 = SGConv(in_dim+3, in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(0.5)
        
        self.lnorm_ff = nn.LayerNorm(in_dim)

    def create_graph_data(self, coordinates, features):
        """
        Create a PyTorch Geometric Data object from coordinates and features using k-nearest neighbors.

        Parameters:
        - coordinates: Tensor of shape [n, 3], the XYZ coordinates of the nodes.
        - features: Tensor of shape [n, d], the features of the nodes.

        Returns:
        - data: A PyTorch Geometric Data object with node features and edge index [2, n x k].
        """
        edge_index = knn_graph(coordinates.float(), k=min(32, coordinates.size(0)), loop=False)
        # Create the PyTorch Geometric data object
        data = Data(x=features, edge_index=edge_index)
        
        return data

    def get_batch_splits(self, me_tensor):
        _, counts = torch.unique(me_tensor.C[:, 0], return_counts=True)
        splits_C = torch.split(me_tensor.C, counts.tolist(), dim=0)
        splits_C = [s[:, 1:] for s in splits_C]
        splits_F = torch.split(me_tensor.F, counts.tolist())

        return splits_C, splits_F
    
    def forward(self, x):
        ## Att
        x_att = self.att_layer(x)
        x_att = x_att + x
        
        Batch_split_coordinates, Batch_split_features = self.get_batch_splits(x_att)
        batch = []
        for i in range (len(Batch_split_features)):
            
            features = Batch_split_features[i]
            coordinates = Batch_split_coordinates[i]
            x_input_ff = self.lnorm_att(features)
            ## FF
            graph_data = self.create_graph_data(coordinates,x_input_ff)
            x_ff = self.sgconv1(torch.cat((graph_data.x,coordinates),dim=-1), graph_data.edge_index)
            x_ff = self.relu(x_ff)
            x_ff = self.linear(x_ff)
            x_ff = self.dropout(x_ff)

            x_out = x_ff + x_input_ff
            x_total = self.lnorm_ff(x_out)

            batch.append(x_total)

        x_total_batch = torch.cat(batch, dim=0)
        x_transformer = ME.SparseTensor(features=x_total_batch, tensor_stride=x_att.tensor_stride, 
                                            device = x_att.device, coordinate_map_key=x_att.coordinate_map_key, coordinate_manager=x_att.coordinate_manager)

        return x_transformer

class PointGraph(nn.Module):
    """
    https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.nn.conv.PointGNNConv.html#torch_geometric.nn.conv.PointGNNConv
    """
    def __init__(self, in_dim):
        super(PointGraph, self).__init__()

        self.att_layer = RNSA(in_dim)

        self.lnorm_att = nn.LayerNorm(in_dim)

        self.MLP_h = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 3)  # Outputs three-dimensional offsets
        )

        self.MLP_f = nn.Sequential(
            nn.Linear(in_dim + 3, in_dim),  # Include position vector
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )

        self.MLP_g = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )

        self.point_gnn_conv = PointGNNConv(mlp_h=self.MLP_h, mlp_f=self.MLP_f, mlp_g=self.MLP_g)

        self.lnorm_ff = nn.LayerNorm(in_dim)

    def create_graph_data(self, coordinates, features):
        """
        Create a PyTorch Geometric Data object from coordinates and features using k-nearest neighbors.

        Parameters:
        - coordinates: Tensor of shape [n, 3], the XYZ coordinates of the nodes.
        - features: Tensor of shape [n, d], the features of the nodes.

        Returns:
        - data: A PyTorch Geometric Data object with node features and edge index [2, n x k].
        """
        edge_index = knn_graph(coordinates.float(), k=min(32, coordinates.size(0)), loop=False)
        # Create the PyTorch Geometric data object
        data = Data(x=features, edge_index=edge_index)
        
        return data

    def get_batch_splits(self, me_tensor):
        _, counts = torch.unique(me_tensor.C[:, 0], return_counts=True)
        splits_C = torch.split(me_tensor.C, counts.tolist(), dim=0)
        splits_C = [s[:, 1:] for s in splits_C]
        splits_F = torch.split(me_tensor.F, counts.tolist())

        return splits_C, splits_F
    
    def forward(self, x):
        ## Att
        x_att = self.att_layer(x)
        
        Batch_split_coordinates, Batch_split_features = self.get_batch_splits(x_att)
        batch = []
        for i in range (len(Batch_split_features)):
            
            features = Batch_split_features[i]
            coordinates = Batch_split_coordinates[i]
            x_input_ff = self.lnorm_att(features)
            ## FF
            graph_data = self.create_graph_data(coordinates, x_input_ff)
            x_out = self.point_gnn_conv(graph_data.x, coordinates, graph_data.edge_index)
            x_total = self.lnorm_ff(x_out)

            batch.append(x_total)

        x_total_batch = torch.cat(batch, dim=0)
        x_transformer = ME.SparseTensor(features=x_total_batch, tensor_stride=x_att.tensor_stride, 
                                            device = x_att.device, coordinate_map_key=x_att.coordinate_map_key, coordinate_manager=x_att.coordinate_manager)

        return x_transformer

class Stacked3RNSAFormer(nn.Module):
    def __init__(self, in_dim):
        super(Stacked3RNSAFormer, self).__init__()
        
        # Define three RNSAFormer layers
        self.rnsaformer1 = RNSAFormer(in_dim)
        self.rnsaformer2 = RNSAFormer(in_dim)
        self.rnsaformer3 = RNSAFormer(in_dim)

    def forward(self, x):
        
        x = self.rnsaformer1(x)
        x = self.rnsaformer2(x)
        x = self.rnsaformer3(x)

        return x

class Stacked2RNSAFormer(nn.Module):
    def __init__(self, in_dim):
        super(Stacked2RNSAFormer, self).__init__()
        
        # Define two RNSAFormer layers
        self.rnsaformer1 = RNSAFormer(in_dim)
        self.rnsaformer2 = RNSAFormer(in_dim)

    def forward(self, x):
        
        x = self.rnsaformer1(x)
        x = self.rnsaformer2(x)

        return x

class RNSAFormer(nn.Module):
    """
    Just simple one layer rnsa with normalizing and nn.Linear as FF 
    """
    def __init__(self, in_dim):
        super(RNSAFormer, self).__init__()

        self.att_layer = RNSA(in_dim)

        self.lnorm_att = nn.LayerNorm(in_dim)

        self.ff = nn.Linear(in_dim,in_dim)
        
        self.lnorm_ff = nn.LayerNorm(in_dim)

    def get_batch_splits(self, me_tensor):
        _, counts = torch.unique(me_tensor.C[:, 0], return_counts=True)
        splits_C = torch.split(me_tensor.C, counts.tolist(), dim=0)
        splits_C = [s[:, 1:] for s in splits_C]
        splits_F = torch.split(me_tensor.F, counts.tolist())

        return splits_C, splits_F
    
    def forward(self, x):
        ## Att
        x_att = self.att_layer(x)
        # x_att = x_att + x
        
        _, Batch_split_features = self.get_batch_splits(x_att)
        batch = []
        for i in range (len(Batch_split_features)):

            x_input_ff = self.lnorm_att(Batch_split_features[i])
            ## FF
            x_ff = self.ff(x_input_ff)
            x_out = x_ff + x_input_ff
            x_total = self.lnorm_ff(x_out)

            batch.append(x_total)

        x_total_batch = torch.cat(batch, dim=0)
        x_transformer = ME.SparseTensor(features=x_total_batch, tensor_stride=x_att.tensor_stride, 
                                            device = x_att.device, coordinate_map_key=x_att.coordinate_map_key, coordinate_manager=x_att.coordinate_manager)

        return x_transformer

class GraphFormer(nn.Module):
    """
    with simple graph convolution
    https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.nn.conv.SGConv.html#torch_geometric.nn.conv.SGConv
    and https://arxiv.org/abs/1902.07153
    """
    def __init__(self, in_dim):
        super(GraphFormer, self).__init__()

        self.att_layer = RNSA(in_dim)

        self.lnorm_att = nn.LayerNorm(in_dim)

        self.positional_embedding = nn.Sequential(
                nn.Linear(3, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, in_dim),
                )

        self.sgconv1 = SGConv(in_dim, in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.sgconv2 = SGConv(in_dim, in_dim)
        self.dropout = nn.Dropout(0.5)
        
        self.lnorm_ff = nn.LayerNorm(in_dim)

    def create_graph_data(self, coordinates, features):
        """
        Create a PyTorch Geometric Data object from coordinates and features using k-nearest neighbors.

        Parameters:
        - coordinates: Tensor of shape [n, 3], the XYZ coordinates of the nodes.
        - features: Tensor of shape [n, d], the features of the nodes.

        Returns:
        - data: A PyTorch Geometric Data object with node features and edge index [2, n x k].
        """
        edge_index = knn_graph(coordinates.float(), k=min(32, coordinates.size(0)), loop=False) #used to be 32
        # Create the PyTorch Geometric data object
        data = Data(x=features, edge_index=edge_index)
        
        return data

    def get_batch_splits(self, me_tensor):
        _, counts = torch.unique(me_tensor.C[:, 0], return_counts=True)
        splits_C = torch.split(me_tensor.C, counts.tolist(), dim=0)
        splits_C = [s[:, 1:] for s in splits_C]
        splits_F = torch.split(me_tensor.F, counts.tolist())

        return splits_C, splits_F
    
    def forward(self, x):
        ## Att
        x_att = self.att_layer(x)
        x_att = x_att + x
        
        Batch_split_coordinates, Batch_split_features = self.get_batch_splits(x_att)
        batch = []
        for i in range (len(Batch_split_features)):
            
            features = Batch_split_features[i]
            coordinates = Batch_split_coordinates[i]
            x_input_ff = self.lnorm_att(features)
            ## FF
            graph_data = self.create_graph_data(coordinates,features)
            x_ff = self.sgconv1(graph_data.x, graph_data.edge_index)
            x_ff = self.relu(x_ff)
            x_ff = self.sgconv2(x_ff, graph_data.edge_index)
            x_ff = self.dropout(x_ff)

            x_out = x_ff + x_input_ff
            x_total = self.lnorm_ff(x_out)

            batch.append(x_total)

        x_total_batch = torch.cat(batch, dim=0)
        x_transformer = ME.SparseTensor(features=x_total_batch, tensor_stride=x_att.tensor_stride, 
                                            device = x_att.device, coordinate_map_key=x_att.coordinate_map_key, coordinate_manager=x_att.coordinate_manager)

        return x_transformer

class GraphFF_PreLN(nn.Module):
    """
    Utilizes PreLN for Graph based on Point Transformer v3
    """
    def __init__(self, input_channel):
        super(GraphFF_PreLN, self).__init__()

        self.att_layer = RNSA_PreLN(input_channel)

        self.lnorm_preff = nn.LayerNorm(input_channel)

        self.graph_ff = Sequential('x, edge_index', [
            (GCNConv(input_channel, input_channel), 'x, edge_index -> x'),
            torch.nn.ReLU(inplace=True),
            (GCNConv(input_channel, input_channel), 'x, edge_index -> x'),
            nn.Dropout(0.5)
        ])

    @torch.no_grad()
    def create_graph_data(self, coordinates, features):
        """
        Create a PyTorch Geometric Data object from coordinates and features using k-nearest neighbors.

        Parameters:
        - coordinates: Tensor of shape [n, 3], the XYZ coordinates of the nodes.
        - features: Tensor of shape [n, d], the features of the nodes.

        Returns:
        - data: A PyTorch Geometric Data object with node features and edge index [2, n x k].
        """
        edge_index = knn_graph(coordinates.float(), k=min(32, coordinates.size(0)), loop=False)
        # Create the PyTorch Geometric data object
        data = Data(x=features, edge_index=edge_index)
        
        return data

    def get_batch_splits(self, me_tensor):
        _, counts = torch.unique(me_tensor.C[:, 0], return_counts=True)
        splits_C = torch.split(me_tensor.C, counts.tolist(), dim=0)
        splits_C = [s[:, 1:] for s in splits_C]
        splits_F = torch.split(me_tensor.F, counts.tolist())

        return splits_C, splits_F
    
    def forward(self, x):

        ## Att
        x_att = self.att_layer(x)
        x_att = x_att + x
        
        Batch_split_coordinates, Batch_split_features = self.get_batch_splits(x_att)
        batch = []
        for i in range (len(Batch_split_features)):
            
            coordinates = Batch_split_coordinates[i]
            features = Batch_split_features[i]
            x_att_norm = self.lnorm_preff(features)
            ## FF
            data = self.create_graph_data(coordinates,x_att_norm).to(x_att.device)
            x_FF = self.graph_ff(data.x, data.edge_index)

            batch.append(x_FF)

        x_total_batch = torch.cat(batch, dim=0)
        x_feedforward = ME.SparseTensor(features=x_total_batch, tensor_stride=x_att.tensor_stride, 
                                            device = x_att.device, coordinate_map_key=x_att.coordinate_map_key, coordinate_manager=x_att.coordinate_manager)
        
        x_transformer = x_att + x_feedforward

        return x_transformer
            
class PreLNTransformer(nn.Module):
            """ Vanilla Transformer Optained from https://github.com/jadore801120/
                attention-is-all-you-need-pytorch/tree/master/transformer
                attention layer used: RNSA_PreLN_NoDropout in Feed Forward
                """
            def __init__(self, in_dim):
                super(PreLNTransformer, self).__init__()

                self.att_layer = RNSA_PreLN(in_dim)

                self.feedforward = nn.Sequential(
                        nn.Linear(in_dim, in_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(in_dim, in_dim)
                        # nn.Dropout(0.5)
                )

                self.lnorm_preff = nn.LayerNorm(in_dim)

            def get_batch_splits(self, me_tensor):
                _, counts = torch.unique(me_tensor.C[:, 0], return_counts=True)
                splits_C = torch.split(me_tensor.C, counts.tolist(), dim=0)
                splits_C = [s[:, 1:] for s in splits_C]
                splits_F = torch.split(me_tensor.F, counts.tolist())

                return splits_C, splits_F

            def forward(self, x):
                ## Att
                x_att = self.att_layer(x)
                x_att = x_att + x
                
                _, Batch_split_features = self.get_batch_splits(x_att)
                batch = []
                for i in range (len(Batch_split_features)):
                    
                    x_att_norm = self.lnorm_preff(Batch_split_features[i])
                    ## FF
                    x_FF = self.feedforward(x_att_norm)

                    batch.append(x_FF)

                x_total_batch = torch.cat(batch, dim=0)
                x_feedforward = ME.SparseTensor(features=x_total_batch, tensor_stride=x_att.tensor_stride, 
                                                 device = x_att.device, coordinate_map_key=x_att.coordinate_map_key, coordinate_manager=x_att.coordinate_manager)
                
                x_transformer = x_att + x_feedforward

                return x_transformer

class RNSA_PreLN(nn.Module):
            """ KNN Self Attention with Positional Embedding with k = 32"""
            def __init__(self, in_dim):
                super(RNSA_PreLN, self).__init__()

                self.query_conv = nn.Sequential(
                    nn.Linear(in_dim, in_dim, bias=True),
                    nn.ReLU(inplace=True)
                )

                self.key_conv =  nn.Sequential(
                    nn.Linear(in_dim, in_dim, bias=True),
                    nn.ReLU(inplace=True)
                )

                self.value_conv =  nn.Linear(in_dim, in_dim, bias=True)

                self.layernorm_q = nn.LayerNorm(in_dim)
                self.layernorm_kv = nn.LayerNorm(in_dim)

                self.linear = nn.Sequential(
                nn.Linear(3, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, in_dim),
                )

                self.normalized = Sparsemax(dim=1)

                self.weight_encoding = nn.Sequential(
                    nn.Linear(in_dim, in_dim, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_dim, in_dim)
                )

                self.attn_drop = nn.Dropout(0.5)

            def get_batch_splits(self, me_tensor):
                _, counts = torch.unique(me_tensor.C[:, 0], return_counts=True)
                splits_C = torch.split(me_tensor.C, counts.tolist(), dim=0)
                splits_C = [s[:, 1:] for s in splits_C]
                splits_F = torch.split(me_tensor.F, counts.tolist())

                return splits_C, splits_F

            def pytorchknn(self,x):

                x = x.unsqueeze(0)
                k = min (32,x.size(1))
                knn = knn_points(x, x, K=k)
                indices = knn[1]
                neighbors = knn_gather(x,indices)
                return neighbors.squeeze(0),indices.squeeze(0),k # return gathered points indices

            def forward(self, x):

                Batch_split_coordinate, Batch_split_features = self.get_batch_splits(x)
                projection = []
                for i in range (len(Batch_split_features)):

                    feature_point = Batch_split_features[i].to(x.device)
                    feature_point =  self.layernorm_q(feature_point)
                    coordinate_point = Batch_split_coordinate[i].to(x.device).float()

                    neighbors_coordinates, indices, _ = self.pytorchknn(coordinate_point)
                    differential_coordinates = neighbors_coordinates - coordinate_point.unsqueeze(1)
                    neighbors_features = feature_point[indices]
                    feature_coordinate = self.linear(differential_coordinates)
                    positional_embeding= feature_coordinate + neighbors_features
                    positional_embeding= self.layernorm_kv(positional_embeding)

                    Query = self.query_conv(feature_point)
                    Key = self.key_conv(positional_embeding)
                    Value = self.value_conv(positional_embeding)
                    
                    unsqueeze_query = Query.unsqueeze(1)
                    relation_qk = Key - unsqueeze_query
                    relation_qk = relation_qk * feature_coordinate
                    weight = self.weight_encoding(relation_qk)
                    weight = self.attn_drop(self.normalized(weight))

                    feat = torch.einsum("n k d, n k d -> n d", Value, weight)
                    projection.append(feat)

                scores = torch.cat(projection, dim=0)
                attended_input = ME.SparseTensor(features=scores, tensor_stride=x.tensor_stride, 
                                                 device = x.device, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
                # attended_input = x + attended_input 
                #moved to the transformer part
                return attended_input

class vTransformer(nn.Module):
            """ Vanilla Transformer Optained from https://github.com/jadore801120/
                attention-is-all-you-need-pytorch/tree/master/transformer
                attention layer used: RNSA
                """
            def __init__(self, in_dim):
                super(vTransformer, self).__init__()

                self.att_layer = RNSA(in_dim)

                self.lnorm_att = nn.LayerNorm(in_dim)

                self.feedforward = nn.Sequential(
                        nn.Linear(in_dim, in_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(in_dim, in_dim),
                        nn.Dropout(0.5)
                )

                self.lnorm_ff = nn.LayerNorm(in_dim)

            def get_batch_splits(self, me_tensor):
                _, counts = torch.unique(me_tensor.C[:, 0], return_counts=True)
                splits_C = torch.split(me_tensor.C, counts.tolist(), dim=0)
                splits_C = [s[:, 1:] for s in splits_C]
                splits_F = torch.split(me_tensor.F, counts.tolist())

                return splits_C, splits_F

            def forward(self, x):
                ## Att
                x_att = self.att_layer(x)
                #x_att = x_att + x
                
                _, Batch_split_features = self.get_batch_splits(x_att)
                batch = []
                for i in range (len(Batch_split_features)):

                    x_att_norm = self.lnorm_att(Batch_split_features[i])
                    ## FF
                    x_FF = self.feedforward(x_att_norm)
                    x_total = self.lnorm_ff(x_FF + x_att_norm)

                    batch.append(x_total)

                x_total_batch = torch.cat(batch, dim=0)
                x_transformer = ME.SparseTensor(features=x_total_batch, tensor_stride=x_att.tensor_stride, 
                                                 device = x_att.device, coordinate_map_key=x_att.coordinate_map_key, coordinate_manager=x_att.coordinate_manager)

                return x_transformer

class RNSA(nn.Module):
            """ KNN Self Attention with Positional Embedding with k = 32"""
            def __init__(self, in_dim):
                super(RNSA, self).__init__()

                self.query_conv = nn.Sequential(
                    nn.Linear(in_dim, in_dim, bias=True),
                    nn.ReLU(inplace=True)
                )

                self.key_conv =  nn.Sequential(
                    nn.Linear(in_dim, in_dim, bias=True),
                    nn.ReLU(inplace=True)
                )

                self.value_conv =  nn.Linear(in_dim, in_dim, bias=True)

                self.linear = nn.Sequential(
                nn.Linear(3, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, in_dim),
                )

                self.normalized = Sparsemax(dim=1)

                self.weight_encoding = nn.Sequential(
                    nn.Linear(in_dim, in_dim, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_dim, in_dim)
                )

                self.attn_drop = nn.Dropout(0.5)

            def get_batch_splits(self, me_tensor):
                _, counts = torch.unique(me_tensor.C[:, 0], return_counts=True)
                splits_C = torch.split(me_tensor.C, counts.tolist(), dim=0)
                splits_C = [s[:, 1:] for s in splits_C]
                splits_F = torch.split(me_tensor.F, counts.tolist())

                return splits_C, splits_F

            def pytorchknn(self,x):

                x = x.unsqueeze(0)
                k = min (32,x.size(1)) # used to be 32
                knn = knn_points(x, x, K=k)
                indices = knn[1]
                neighbors = knn_gather(x,indices)
                return neighbors.squeeze(0),indices.squeeze(0),k # return gathered points indices

            def forward(self, x):

                Batch_split_coordinate, Batch_split_features = self.get_batch_splits(x)
                projection = []
                for i in range (len(Batch_split_features)):

                    feature_point = Batch_split_features[i].to(x.device)
                    coordinate_point = Batch_split_coordinate[i].to(x.device).float()

                    neighbors_coordinates, indices, k = self.pytorchknn(coordinate_point)
                    differential_coordinates = neighbors_coordinates - coordinate_point.unsqueeze(1)
                    neighbors_features = feature_point[indices]
                    feature_coordinate = self.linear(differential_coordinates)
                    positional_embeding= feature_coordinate + neighbors_features
                    Query = self.query_conv(Batch_split_features[i])
                    Key = self.key_conv(positional_embeding)
                    Value = self.value_conv(positional_embeding)
                    unsqueeze_query = Query.unsqueeze(1)
                    relation_qk = Key - unsqueeze_query
                    relation_qk = relation_qk * feature_coordinate
                    weight = self.weight_encoding(relation_qk)
                    weight = self.attn_drop(self.normalized(weight))

                    feat = torch.einsum("n k d, n k d -> n d", Value, weight)
                    projection.append(feat)

                scores = torch.cat(projection, dim=0)
                attended_input = ME.SparseTensor(features=scores, tensor_stride=x.tensor_stride, 
                                                 device = x.device, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
                attended_input = x + attended_input 

                return attended_input

class RNSA_NoDrop(nn.Module):
            """ KNN Self Attention with Positional Embedding with k = 32"""
            def __init__(self, in_dim):
                super(RNSA_NoDrop, self).__init__()

                self.query_conv = nn.Sequential(
                    nn.Linear(in_dim, in_dim, bias=True),
                    nn.ReLU(inplace=True)
                )

                self.key_conv =  nn.Sequential(
                    nn.Linear(in_dim, in_dim, bias=True),
                    nn.ReLU(inplace=True)
                )

                self.value_conv =  nn.Linear(in_dim, in_dim, bias=True)

                self.linear = nn.Sequential(
                nn.Linear(3, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, in_dim),
                )

                self.normalized = Sparsemax(dim=1)

                self.weight_encoding = nn.Sequential(
                    nn.Linear(in_dim, in_dim, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_dim, in_dim)
                )

                self.attn_drop = nn.Dropout(0.5)

            def get_batch_splits(self, me_tensor):
                _, counts = torch.unique(me_tensor.C[:, 0], return_counts=True)
                splits_C = torch.split(me_tensor.C, counts.tolist(), dim=0)
                splits_C = [s[:, 1:] for s in splits_C]
                splits_F = torch.split(me_tensor.F, counts.tolist())

                return splits_C, splits_F

            def pytorchknn(self,x):

                x = x.unsqueeze(0)
                k = min (32,x.size(1))
                knn = knn_points(x, x, K=k)
                indices = knn[1]
                neighbors = knn_gather(x,indices)
                return neighbors.squeeze(0),indices.squeeze(0),k # return gathered points indices

            def forward(self, x):

                Batch_split_coordinate, Batch_split_features = self.get_batch_splits(x)
                projection = []
                for i in range (len(Batch_split_features)):

                    feature_point = Batch_split_features[i].to(x.device)
                    coordinate_point = Batch_split_coordinate[i].to(x.device).float()

                    neighbors_coordinates, indices, k = self.pytorchknn(coordinate_point)
                    differential_coordinates = neighbors_coordinates - coordinate_point.unsqueeze(1)
                    neighbors_features = feature_point[indices]
                    feature_coordinate = self.linear(differential_coordinates)
                    positional_embeding= feature_coordinate + neighbors_features
                    Query = self.query_conv(Batch_split_features[i])
                    Key = self.key_conv(positional_embeding)
                    Value = self.value_conv(positional_embeding)
                    unsqueeze_query = Query.unsqueeze(1)
                    relation_qk = Key - unsqueeze_query
                    relation_qk = relation_qk * feature_coordinate
                    weight = self.weight_encoding(relation_qk)
                    weight = self.normalized(weight)

                    feat = torch.einsum("n k d, n k d -> n d", Value, weight)
                    projection.append(feat)

                scores = torch.cat(projection, dim=0)
                attended_input = ME.SparseTensor(features=scores, tensor_stride=x.tensor_stride, 
                                                 device = x.device, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
                attended_input = x + attended_input 

                return attended_input
