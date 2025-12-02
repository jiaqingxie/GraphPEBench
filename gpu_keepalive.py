#!/usr/bin/env python3
"""
GPU Keep-Alive Script
防止GPU因利用率过低而自动关机的简单脚本
"""

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import time
import argparse
import signal
import sys
import random

class SimpleGNN(torch.nn.Module):
    """简单的GNN模型用于GPU保活"""
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=64, num_layers=3):
        super(SimpleGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

def create_random_graph(num_nodes, num_edges, node_features, device):
    """创建随机图数据"""
    # 创建随机边
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    
    # 创建随机节点特征
    x = torch.randn(num_nodes, node_features, device=device)
    
    # 创建随机标签
    y = torch.randint(0, 10, (1,), device=device)
    
    return Data(x=x, edge_index=edge_index, y=y)

def signal_handler(sig, frame):
    print('\n正在安全退出GPU保活脚本...')
    sys.exit(0)

def gpu_keepalive(device_id=1, utilization=60, interval=60):
    """
    GPU保活函数
    
    Args:
        device_id (int): GPU设备ID，默认0
        utilization (int): 目标利用率百分比，默认30%
        interval (int): 检查间隔（秒），默认60秒
    """
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误: CUDA不可用，请检查PyTorch和CUDA安装")
        return
    
    # 检查指定的GPU是否存在
    if device_id >= torch.cuda.device_count():
        print(f"错误: GPU {device_id} 不存在，可用GPU数量: {torch.cuda.device_count()}")
        return
    
    device = torch.device(f'cuda:{device_id}')
    print(f"使用GPU: {torch.cuda.get_device_name(device_id)}")
    print(f"目标利用率: {utilization}%")
    print(f"检查间隔: {interval}秒")
    print("按 Ctrl+C 停止脚本")
    print("-" * 50)
    
    # 创建GNN模型和数据
    try:
        # 根据利用率调整图的大小
        base_nodes = max(100, int(50 * (utilization / 10)))
        base_edges = base_nodes * 4
        node_features = 128
        batch_size = max(1, int(utilization / 20))
        
        # 创建GNN模型
        model = SimpleGNN(input_dim=node_features, hidden_dim=256, output_dim=64, num_layers=4)
        model = model.to(device)
        model.train()
        
        # 创建优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        print(f"GNN模型参数: {sum(p.numel() for p in model.parameters())} 个参数")
        print(f"图规模: {base_nodes} 节点, {base_edges} 边, 批次大小: {batch_size}")
        
        while True:
            start_time = time.time()
            
            # 执行GNN训练来维持利用率
            with torch.cuda.device(device):
                # 创建随机图数据批次
                graph_list = []
                for _ in range(batch_size):
                    # 随机调整图大小
                    num_nodes = base_nodes + random.randint(-20, 20)
                    num_edges = base_edges + random.randint(-100, 100)
                    graph = create_random_graph(num_nodes, num_edges, node_features, device)
                    graph_list.append(graph)
                
                batch_data = Batch.from_data_list(graph_list)
                
                # 前向传播
                optimizer.zero_grad()
                output = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                
                # 创建目标标签
                target = torch.randint(0, 10, (output.size(0),), device=device)
                
                # 计算损失
                loss = criterion(output, target)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 额外的计算操作
                with torch.no_grad():
                    # 一些矩阵运算
                    temp = torch.mm(output, output.T)
                    temp = F.relu(temp)
                    temp = torch.softmax(temp, dim=1)
                
                # 确保计算完成
                torch.cuda.synchronize()
                
                # 清理变量
                del batch_data, output, target, loss, temp, graph_list
            
            # 显示状态
            gpu_mem = torch.cuda.memory_allocated(device) / 1024**3  # GB
            max_mem = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
            elapsed = time.time() - start_time
            
            print(f"时间: {time.strftime('%H:%M:%S')} | "
                  f"GPU内存: {gpu_mem:.2f}GB | "
                  f"最大内存: {max_mem:.2f}GB | "
                  f"GNN训练时间: {elapsed:.2f}s | "
                  f"批次: {batch_size}图")
            
            # 等待下一次循环
            time.sleep(max(0, interval - elapsed))
            
    except KeyboardInterrupt:
        print("\n收到中断信号，正在退出...")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("GPU保活脚本已退出")

def main():
    parser = argparse.ArgumentParser(description='GPU Keep-Alive Script')
    parser.add_argument('--gpu', type=int, default=1, 
                      help='GPU设备ID (默认: 1)')
    parser.add_argument('--util', type=int, default=60,
                      help='目标利用率百分比 (默认: 60)')
    parser.add_argument('--interval', type=int, default=60,
                      help='检查间隔秒数 (默认: 60)')
    
    args = parser.parse_args()
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=== GNN GPU保活脚本 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"PyTorch Geometric版本: {torch_geometric.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    
    if torch.cuda.device_count() > 0:
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("-" * 50)
    
    gpu_keepalive(device_id=args.gpu, 
                  utilization=args.util,
                  interval=args.interval)

if __name__ == "__main__":
    main()
