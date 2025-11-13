import torch
import numpy as np
import igraph as ig
import networkx as nx
from collections import Counter, defaultdict
from torch_sparse import SparseTensor
from torch_geometric.utils import from_networkx, to_networkx
from methods.replay import Replay
from backbones.encoder import Encoder

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0

class GGReplay(Replay):
    def __init__(self, model, tasks, budget, m_update, device, minor_thres=100,
                 community_algo='multilevel', max_supernodes=100, top_k=100, 
                 fixed_supernode_count=100, preserve_community_structure=True,
                 community_weight=0.3, minority_ratio=0.5):
        super().__init__(model, tasks, budget, m_update, device)
        self.minor_thres = minor_thres
        self.community_algo = community_algo
        self.max_supernodes = max_supernodes
        self.top_k = top_k
        self.fixed_supernode_count = fixed_supernode_count
        self.preserve_community_structure = preserve_community_structure
        self.community_weight = community_weight
        self.minority_ratio = minority_ratio  
        self.comm_info = {}

    def memorize(self, task, budgets):
        pyg_supergraph = self._construct_global_graph(task)
        pyg_supergraph.train_mask = torch.ones(pyg_supergraph.num_nodes, dtype=torch.bool)
        return pyg_supergraph

    def _construct_global_graph(self, task):
        task.feat = task.x
        task.label = task.y
        graph_nx = to_networkx(task, node_attrs=['x', 'y', 'label', 'feat'])
        g_ig = ig.Graph.from_networkx(graph_nx).as_undirected()

        if self.community_algo == "multilevel":
            partitions = g_ig.community_multilevel()
        elif self.community_algo == 'leiden':
            partitions = g_ig.community_leiden("modularity")
        elif self.community_algo == 'LP':
            partitions = g_ig.community_label_propagation()
        elif self.community_algo=="infomap":
            partitions = g_ig.community_infomap()
        elif self.community_algo=="fastgreedy":
            partitions = g_ig.community_fastgreedy().as_clustering()
        else:
            raise NotImplementedError("Only 'multilevel' is supported.")

        subgraphs = partitions.subgraphs()
        membership = partitions.membership


        community_info = self._extract_community_info(g_ig, subgraphs, membership)

        if len(subgraphs) > self.max_supernodes:
            membership = community_aware_merge(
                subgraphs, membership, self.max_supernodes, self.top_k, community_info
            )
            clust_new = ig.VertexClustering(g_ig, membership=membership)
            subgraphs = clust_new.subgraphs()


        g_contracted = self._community_aware_contraction(g_ig, membership, subgraphs)

        g_final = hybrid_balance_with_community_and_ratio(
            g_contracted, self.fixed_supernode_count, community_info, 
            self.community_weight, self.minority_ratio
        )


        g_final.vs['x'] = g_final.vs['feat']
        g_final.vs['y'] = g_final.vs['label']
        

        for i in range(g_final.vcount()):
            g_final.vs[i]['_nx_name'] = i
        

        try:
            pyg_graph = from_networkx(g_final.to_networkx())
        except (ValueError, TypeError) as e:
            print(f"NetworkX conversion error: {e}")
       
            pyg_graph = create_pyg_graph_manually(g_final)
        
        pyg_graph.adj_t = SparseTensor(
            row=pyg_graph.edge_index[0],
            col=pyg_graph.edge_index[1],
            sparse_sizes=(pyg_graph.num_nodes, pyg_graph.num_nodes)
        ) + SparseTensor.eye(pyg_graph.num_nodes, pyg_graph.num_nodes)

        return pyg_graph

    def _extract_community_info(self, g_ig, subgraphs, membership):

        community_info = {
            'modularity': g_ig.modularity(membership),
            'sizes': [sg.vcount() for sg in subgraphs],
            'densities': [sg.density() if sg.vcount() > 1 else 0 for sg in subgraphs],
            'conductances': [],
            'internal_edges': [],
            'external_edges': []
        }
        
        # 각 커뮤니티의 conductance 계산
        for i, sg in enumerate(subgraphs):
            community_nodes = [j for j, m in enumerate(membership) if m == i]
            internal_edges = sg.ecount()
            
            # 외부로 나가는 엣지 수 계산
            external_edges = 0
            for node in community_nodes:
                for neighbor in g_ig.neighbors(node):
                    if membership[neighbor] != i:
                        external_edges += 1
            
            conductance = external_edges / (internal_edges + external_edges + 1e-8)
            community_info['conductances'].append(conductance)
            community_info['internal_edges'].append(internal_edges)
            community_info['external_edges'].append(external_edges)
        
        return community_info

    def _community_aware_contraction(self, g_ig, membership, subgraphs):
    
        g_contracted = g_ig.copy()
        
        g_contracted.contract_vertices(membership, combine_attrs="first")
        g_contracted.simplify(combine_edges="ignore")

        for i in range(g_contracted.vcount()):
            sub = subgraphs[i] if i < len(subgraphs) else None
            if sub:
                labels = [int(l) for l in sub.vs['label']] if 'label' in sub.vs.attributes() else [0]
                feats = np.array(sub.vs['feat']) if 'feat' in sub.vs.attributes() else np.zeros((1, 10))
                
                g_contracted.vs[i]['label'] = int(Counter(labels).most_common(1)[0][0])
                g_contracted.vs[i]['feat'] = np.mean(feats, axis=0)
                
                g_contracted.vs[i]['community_size'] = sub.vcount()
                g_contracted.vs[i]['community_density'] = sub.density() if sub.vcount() > 1 else 0
                g_contracted.vs[i]['class_diversity'] = len(set(labels))
        
        return g_contracted


def hybrid_balance_with_community_and_ratio(g_ig, target_total_nodes, community_info, 
                                           community_weight, minority_ratio):

    current_labels = [g_ig.vs[i]['label'] for i in range(g_ig.vcount())]
    class_counts = Counter(current_labels)

    if len(class_counts) < 2:
        return balance_with_minority_ratio(g_ig, target_total_nodes, minority_ratio)
    

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    major_class = sorted_classes[0][0]
    minor_class = sorted_classes[1][0]
    
    minor_nodes = int(target_total_nodes * minority_ratio)
    major_nodes = target_total_nodes - minor_nodes
    
    class_nodes = {}
    for i, label in enumerate(current_labels):
        if label not in class_nodes:
            class_nodes[label] = []
        
        attrs = g_ig.vs[i].attributes()
        community_quality = 0
        density = attrs.get('community_density', 0)
        size = attrs.get('community_size', 1)
        max_size = max(community_info['sizes']) if community_info['sizes'] else 1
        community_quality = 0.5 * density + 0.5 * (size / max_size)
        
        total_score = community_quality
        
        class_nodes[label].append((i, total_score))
    

    for label in class_nodes:
        class_nodes[label].sort(key=lambda x: x[1], reverse=True)
    

    g_new = ig.Graph()

    if minor_class in class_nodes:
        available_nodes = class_nodes[minor_class]
        for i in range(minor_nodes):
            node_idx = i % len(available_nodes)
            node_id, _ = available_nodes[node_idx]
            
            feat = g_ig.vs[node_id]['feat']
            if feat is None:
                feat = np.zeros(10)
            elif not isinstance(feat, (list, np.ndarray)):
                feat = [feat] * 10
            
            g_new.add_vertex(
                label=minor_class,
                feat=feat,
                x=feat,
                y=minor_class
            )
    
    if major_class in class_nodes:
        available_nodes = class_nodes[major_class]
        for i in range(major_nodes):
            node_idx = i % len(available_nodes)
            node_id, _ = available_nodes[node_idx]
            
            feat = g_ig.vs[node_id]['feat']
            if feat is None:
                feat = np.zeros(10)
            elif not isinstance(feat, (list, np.ndarray)):
                feat = [feat] * 10
            
            g_new.add_vertex(
                label=major_class,
                feat=feat,
                x=feat,
                y=major_class
            )
    
    return g_new

def preserve_original_edge_structure_with_ratio(g_original, g_new, class_nodes, 
                                               class_labels, node_counts, 
                                               community_weight=0):
    selected_nodes = []
    for i, class_label in enumerate(class_labels):
        if class_label in class_nodes:
            available_nodes = class_nodes[class_label]
            for j in range(node_counts[i]):
                node_idx = j % len(available_nodes)
                if isinstance(available_nodes[node_idx], tuple):
                    node_id, _ = available_nodes[node_idx]
                else:
                    node_id = available_nodes[node_idx]
                selected_nodes.append(node_id)
    
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(selected_nodes)}
    
    for old_src in selected_nodes:
        for neighbor in g_original.neighbors(old_src):
            if neighbor in node_mapping:
                new_src = node_mapping[old_src]
                new_dst = node_mapping[neighbor]
                
                if new_src != new_dst and not g_new.are_connected(new_src, new_dst):
                    g_new.add_edge(new_src, new_dst)
    
    np.random.seed(42)
    target_edges = min(g_original.ecount(), len(selected_nodes) * 2)
    current_edges = g_new.ecount()
    
    for _ in range(target_edges - current_edges):
        src = np.random.randint(0, g_new.vcount())
        dst = np.random.randint(0, g_new.vcount())
        if src != dst and not g_new.are_connected(src, dst):
            g_new.add_edge(src, dst)
    


def community_aware_merge(subgraphs, membership, max_supernodes, top_k, community_info):

    if len(subgraphs) <= max_supernodes:
        return membership

    embeddings = [np.mean(np.array(sg.vs['feat']), axis=0) if 'feat' in sg.vs.attributes() else np.zeros(10)
                  for sg in subgraphs]
    class_counts = [Counter(sg.vs['label']) if 'label' in sg.vs.attributes() else Counter()
                    for sg in subgraphs]
    sizes = [sg.vcount() for sg in subgraphs]

    
    scores = []
    for i, (cc, sz) in enumerate(zip(class_counts, sizes)):

        diversity_score = 0.5 * len(cc)
        size_score = 0.5 * (sz / max(sizes))
        
        total_score = diversity_score + size_score
        scores.append(total_score)

    keep = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    mapping = {}
    class_sum = {k: Counter(class_counts[k]) for k in keep}

    for i in range(len(subgraphs)):
        if i in keep:
            mapping[i] = i
        else:
            def score(k):

                dist = np.linalg.norm(embeddings[i] - embeddings[k])
                
                common = len(set(class_counts[i].keys()) & set(class_sum[k].keys()))
                


                combined = class_sum[k] + class_counts[i]
                freqs = np.array(list(combined.values()))
                imbalance = np.std(freqs)
                
                return dist - 0.1 * common + 0.2 * imbalance 

            best = min(keep, key=score)
            mapping[i] = best
            class_sum[best] += class_counts[i]

    new_membership = [mapping[m] for m in membership]
    new_membership = relabel_membership(new_membership)
    return new_membership


def create_pyg_graph_manually(g_ig):

    import torch
    from torch_geometric.data import Data

    node_features = []
    node_labels = []
    
    for i in range(g_ig.vcount()):
        feat = g_ig.vs[i]['feat']
        if isinstance(feat, (list, np.ndarray)):
            node_features.append(feat)
        else:

            node_features.append([feat] * 10) 
        
        label = g_ig.vs[i]['label']
        node_labels.append(label)
    

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.long)
    

    edges = g_ig.get_edgelist()
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:

        edge_index = torch.empty((2, 0), dtype=torch.long)
    

    pyg_graph = Data(x=x, y=y, edge_index=edge_index)
    pyg_graph.num_nodes = g_ig.vcount()
    
    return pyg_graph

def relabel_membership(new_membership):
    unique_labels = sorted(set(new_membership))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return [label_map[label] for label in new_membership]
