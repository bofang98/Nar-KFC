import os
import io
import pandas as pd
import numpy as np
import string
from uuid import uuid4
import os.path as osp
import base64
from PIL import Image
import sys
import random

Image.MAX_IMAGE_PIXELS = 1e9


def rescale_img(img, tgt=None):
    assert isinstance(tgt, tuple) and -1 in tgt
    w, h = img.size
    if tgt[0] != -1:
        new_w, new_h = tgt[0], int(tgt[0] / w * h)
    elif tgt[1] != -1:
        new_w, new_h = int(tgt[1] / h * w), tgt[1]
    img = img.resize((new_w, new_h))
    return img


def concat_images_vlmeval(images, target_size=-1, mode='h', return_image=False):
    from .file import md5

    ims = [Image.open(im) for im in images]
    if target_size != -1:
        ims = [
            rescale_img(im, (-1, target_size) if mode == 'h' else (target_size, -1))
            for im in ims
        ]

    ws, hs = [x.width for x in ims], [x.height for x in ims]
    if mode == 'h':
        new_w, new_h = sum(ws), max(hs)
        dst = Image.new('RGB', (new_w, new_h))
        for i, im in enumerate(ims):
            dst.paste(im, (sum(ws[:i]), 0))
    elif mode == 'v':
        new_w, new_h = max(ws), sum(hs)
        dst = Image.new('RGB', (new_w, new_h))
        for i, im in enumerate(ims):
            dst.paste(im, (sum(ws[:i], 0)))
    if return_image:
        return dst
    else:
        _str = '\n'.join(images)
        str_md5 = md5(_str)
        tgt = osp.join('/tmp', str_md5 + '.jpg')
        dst.save(tgt)
        return tgt


def mmqa_display(question, target_size=512):
    question = {k.lower(): v for k, v in question.items()}
    keys = list(question.keys())
    keys = [k for k in keys if k not in ['index', 'image']]

    images = question['image']
    if isinstance(images, str):
        images = [images]

    idx = question.pop('index', 'XXX')
    print(f'INDEX: {idx}')

    for im in images:
        image = decode_base64_to_image(im, target_size=target_size)
        display(image)  # noqa: F821

    for k in keys:
        try:
            if not pd.isna(question[k]):
                print(f'{k.upper()}. {question[k]}')
        except ValueError:
            if False in pd.isna(question[k]):
                print(f'{k.upper()}. {question[k]}')


def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    return ret


def encode_image_file_to_base64(image_path, target_size=-1):
    image = Image.open(image_path)
    return encode_image_to_base64(image, target_size=target_size)


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    image = decode_base64_to_image(base64_string, target_size=target_size)
    image.save(image_path)


def build_option_str(option_dict):
    s = 'There are several options: \n'
    for c, content in option_dict.items():
        if not pd.isna(content):
            s += f'{c}. {content}\n'
    return s


def isimg(s):
    return osp.exists(s) or s.startswith('http')


def read_ok(img_path):
    if not osp.exists(img_path):
        return False
    try:
        im = Image.open(img_path)
        assert im.size[0] > 0 and im.size[1] > 0
        return True
    except:
        return False


def gpt_key_set():
    openai_key = os.environ.get('OPENAI_API_KEY', None)
    return isinstance(openai_key, str) and openai_key.startswith('sk-')


def apiok(wrapper):
    s = wrapper.generate('Hello!')
    return wrapper.fail_msg not in s


def circular_pred(df, extract_func=None):
    if extract_func is None:
        extract_func = lambda x: x  # noqa: E731
    df = df.sort_values('index')
    from vlmeval.utils import can_infer_option

    shift = int(1e6)

    choices = [extract_func(x) for x in df['prediction']]
    pred_map = {i: c for i, c in zip(df['index'], choices)}
    flag_map = {i: True for i in pred_map if i < 1e6}
    valid_map = {i: True for i in pred_map if i < 1e6}
    for i in df['index']:
        if i >= shift and pred_map[i] and pred_map[i - shift]:
            if pred_map[i] not in list(
                string.ascii_uppercase
            ) or pred_map[  # noqa: W504
                i - shift
            ] not in list(
                string.ascii_uppercase
            ):

                valid_map[i % shift] = False
                continue
            if (ord(pred_map[i]) - ord(pred_map[i - shift])) % 4 == 1:
                continue
            else:
                flag_map[i % shift] = False
    flag_map = {k: v for k, v in flag_map.items() if valid_map[k]}
    flags = list(flag_map.values())
    return np.mean(flags)


import torch
def greedy(matrix, K, rank):
    device = matrix.device
    n = matrix.shape[0]

    # sym_matrix = torch.triu(matrix, 1) + torch.triu(matrix, 1).T
    sym_matrix = matrix + matrix.T - torch.diag(matrix.diagonal())
    sym_matrix = sym_matrix.to(device)
 
    mask = torch.zeros(n, dtype=torch.float32, device=device)
    # initialization
    max_frame_id = torch.argmax(sym_matrix.diagonal())
    mask[max_frame_id] = 1.0

    selected = []
    selected.append(max_frame_id.item())
    total_weight = 0.0

    for _ in range(K-1):
        delta = sym_matrix @ mask
        delta[mask.bool()] = -float('inf')
        i = torch.argmax(delta).item()
        selected.append(i)
        mask[i] = 1.0
        total_weight += delta[i].item()

    return selected



def dynamic_programming(matrix, K, rank):
    device = matrix.device
    n = matrix.shape[0]
    sym_matrix = torch.triu(matrix, 1) + torch.triu(matrix, 1).T
    sym_matrix = sym_matrix.to(device)
    
    dp = {0: 0.0}  # mask: total_weight
    
    for step in range(K):
        new_dp = {}
        for mask in dp:
            current_weight = dp[mask]
            for i in range(n):
                if not (mask & (1 << i)):
                    new_mask = mask | (1 << i)
                    mask_tensor = torch.zeros(n, device=device)
                    for j in range(n):
                        if mask & (1 << j):
                            mask_tensor[j] = 1.0
                    delta = torch.dot(sym_matrix[i], mask_tensor).item()
                    new_weight = current_weight + delta
                    if new_mask not in new_dp or new_weight > new_dp.get(new_mask, -float('inf')):
                        new_dp[new_mask] = new_weight
        dp = new_dp
        if not dp:
            break
    
    max_weight = -float('inf')
    best_mask = 0
    for mask in dp:
        if bin(mask).count('1') == K and dp[mask] > max_weight:
            max_weight = dp[mask]
            best_mask = mask
    
    selected = [i for i in range(n) if best_mask & (1 << i)]
    return max_weight, selected


def max_spanning_tree(matrix, K, rank):
    device = matrix.device
    matrix_cpu = matrix.cpu()
    n = matrix_cpu.shape[0]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            weight = matrix_cpu[i][j].item()
            if weight > 0:
                edges.append((-weight, i, j))
    edges.sort()
    
    parent = list(range(n))
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
    
    mst_edges = []
    for w, i, j in edges:
        if find(i) != find(j):
            union(i, j)
            mst_edges.append((-w, i, j))
        if len(mst_edges) == n-1:
            break
    
    mst_edges.sort(reverse=True, key=lambda x: x[0])
    selected = set()
    for w, i, j in mst_edges:
        selected.update({i, j})
        if len(selected) >= K:
            break
    
    selected = list(selected)[:K]
    # total = 0
    # for i in range(len(selected)):
    #     for j in range(i+1, len(selected)):
    #         a, b = sorted([selected[i], selected[j]])
    #         total += matrix_cpu[a][b].item()
    return sorted(selected)



def prim_max_spanning_tree(matrix, K, rank):
    """
    使用Prim算法生成最大生成树，并保留前K个节点
    
    Args:
        matrix (torch.Tensor): 邻接矩阵，应为对称矩阵
        K (int): 需要保留的节点数
        device (str): 计算设备，默认为'cuda'
    
    Returns:
        list: 最大生成树中的边列表，仅包含前K个节点形成的子树
    """
    # 确保矩阵在GPU上
    device = matrix.device
    N = matrix.size(0)
    
    # 初始化已选节点集合
    selected = torch.zeros(N, dtype=torch.bool, device=device)
    start = torch.argmax(matrix.diagonal()).item()
    selected[start] = True
    edges = []

    matrix.fill_diagonal_(0)
    sym_matrix = matrix + matrix.T 
    sym_matrix = sym_matrix.to(device)
    
    for _ in range(K-1):
        # 生成候选边掩码（已选节点到未选节点）
        mask = selected.unsqueeze(1) & (~selected).unsqueeze(0)
        
        # 将非候选边设为负无穷
        adjusted = torch.where(mask, sym_matrix, torch.tensor(-float('inf'), device=device))
        
        # 找到最大边的全局索引
        max_val, flat_idx = torch.max(adjusted.view(-1), 0)
        
        if max_val <= -float('inf'):
            break  # 无有效边时提前终止
        
        # 转换为行列索引
        row = flat_idx // N
        col = flat_idx % N
        
        # 记录边并更新已选节点
        edges.append((row.item(), col.item()))
        selected[col] = True
    # edges: [(38, 73), (38, 72), (38, 71), (38, 70), (73, 37), (73, 30), (38, 59)]
    return list(set(x for pair in edges for x in pair))


def uniform_select_matrix(matrix, M):
    n = matrix.shape[0]  # 假设矩阵是 n x n 的
    segment_size = n // M  # 计算每个 segment 的大小
    
    # 计算每个 segment 中央的索引
    indices = torch.tensor([(i * segment_size + (i + 1) * segment_size) // 2 for i in range(M)])

    # 选取对应的行和列
    reduced_matrix = matrix[indices][:, indices]

    # 构造索引映射关系
    index_mapping = {i: indices[i].item() for i in range(M)}

    return reduced_matrix, indices, index_mapping



def incremental_select_nodes(matrix, K, shrink_K, rank):
    """
    增量式选择：
    1. 首先计算每个节点的加权度（该节点与所有其他节点的边权之和）。
    2. 选择加权度最大的节点作为起点。
    3. 当选中节点数 < K 时，对于所有未选节点，
       计算它与当前已选节点之间的边权总和，选取贡献最大的节点加入。
    返回选中的节点（按升序排列）。
    """
    device = matrix.device
    n = matrix.shape[0]

    sym_matrix = matrix + matrix.T 
    sym_matrix = sym_matrix.to(device)

    if n <= 4*K:
        U, Sigma, Vt = torch.svd(sym_matrix)
        Sigma_k = torch.diag(Sigma[:n//4])
        U_k = U[:, :n//4]
        Vt_k = Vt[:, :n//4]
        S_k = U_k @ Sigma_k @ Vt_k.T

        sym_matrix = S_k            # 维度不变
        start = torch.argmax(S_k.diagonal()).item()
        selected = {start}
        indices_mapping = {i: i for i in range(n)}
    else:
        if n > 7200:   # if there are 2h-long videos, doing SVD may cause OOM
            S_k = sym_matrix        # directly downsampling
        else:
            # SVD sparsing
            U, Sigma, Vt = torch.svd(sym_matrix)
            Sigma_k = torch.diag(Sigma[: n//4])
            U_k = U[:, :n//4]
            Vt_k = Vt[:, :n//4]
            S_k = U_k @ Sigma_k @ Vt_k.T

        # low rank approximation
        sparse_matrix, true_indice, indices_mapping = uniform_select_matrix(S_k, M=shrink_K)  # sample 128 nodes
        start = torch.argmax(sparse_matrix.diagonal()).item()
        selected = {start}
        sym_matrix = sparse_matrix


    sym_matrix.fill_diagonal_(0)


    while len(selected) < K:
        best_increment = -float('inf')
        best_node = None
        for i in range(sym_matrix.shape[0]):
            if i in selected:
                continue
            # 计算 i 与当前 selected 中所有节点之间的边权总和
            inc = 0.0
            for j in selected:
                # 由于 A 是对称矩阵，统一取 A[min(i,j), max(i,j)]
                if i < j:
                    inc += sym_matrix[i, j].item()
                else:
                    inc += sym_matrix[j, i].item()
            if inc > best_increment:
                best_increment = inc
                best_node = i
        selected.add(best_node)

    true_select = [indices_mapping[i] for i in selected]
    return true_select, S_k



import torch
import cplex
# def init_cplex_frame_selection(A, K, init_ind, rank):
#     """
#     K: 需要选择的帧数
#     A: N x N 的得分矩阵 (对称矩阵，主对角线为0)
#     init_ind: 初始索引（贪心搜索得到）
#     """
#     N = A.shape[0]
#     device = A.device
    
#     # 创建 CPLEX 求解器
#     problem = cplex.Cplex()
#     # 关闭所有日志输出
#     problem.set_log_stream(None)
#     problem.set_error_stream(None)
#     problem.set_warning_stream(None)
#     problem.set_results_stream(None)

#     problem.parameters.threads.set(16) 
#     problem.parameters.parallel.set(1)
#     problem.parameters.timelimit.set(30)  # 限制 CPLEX 最多运行 30 秒
#     problem.parameters.mip.limits.nodes.set(40000)  # 限制最大搜索节点数
#     problem.parameters.mip.strategy.heuristicfreq.set(5)  # 让 CPLEX 更频繁使用启发式搜索
#     problem.parameters.mip.strategy.rinsheur.set(10)  # 尝试改进已有解
#     problem.parameters.mip.tolerances.mipgap.set(0.001)  # 允许更小误差，提高求解质量
#     problem.parameters.mip.limits.solutions.set(10)  # 让 CPLEX 至少找到 10 个解
#     problem.parameters.mip.strategy.variableselect.set(3)  # 让 CPLEX 优先选择重要变量

#     problem.set_problem_type(cplex.Cplex.problem_type.LP)
#     problem.objective.set_sense(problem.objective.sense.maximize)

#     # 定义变量 X（是否选择帧）
#     x_vars = [f"x_{i}" for i in range(N)]
#     problem.variables.add(names=x_vars, types=[problem.variables.type.binary] * N)

#     # **Step 3: 设置初始解（MIP Start）**
#     initial_solution = [0] * N
#     for idx in init_ind:
#         initial_solution[idx] = 1  # 选取得分最高的 K 帧

#     # 目标函数系数 (1/2 * X^T A X)
#     obj_coeffs = []
#     for i in range(N):
#         for j in range(N):
#             if i < j:  # 只存储上三角部分，避免重复
#                 obj_coeffs.append((f"x_{i}", f"x_{j}", float(A[i][j])))

#     # 添加目标函数（使用二次项）
#     problem.objective.set_quadratic_coefficients(obj_coeffs)

#     # 约束：选取 K 帧
#     var_indices = list(range(N))
#     problem.linear_constraints.add(
#         lin_expr=[cplex.SparsePair(ind=var_indices, val=[1] * N)],
#         senses=["E"],
#         rhs=[K]
#     )

#     is_mip_start_feasible = True  # 预设 MIP Start 是可行的
#     try:
#         problem.MIP_starts.add(
#             [(i, float(initial_solution[i])) for i in range(N)], 
#             problem.MIP_starts.effort_level.check_feasibility  # 仅检查可行性
#         )
#     except cplex.exceptions.errors.CplexSolverError:
#         print("初始解不可行，跳过 MIP Start")
#         is_mip_start_feasible = False

#     # **Step 7: 如果 MIP Start 可行，再真正添加**
#     if is_mip_start_feasible:
#         problem.MIP_starts.add(
#             [(i, float(initial_solution[i])) for i in range(N)], 
#             problem.MIP_starts.effort_level.auto  # 让 CPLEX 自动决定是否使用
#         )

#     # 求解
#     problem.solve()

#     # 输出结果
#     solution = problem.solution
#     selected_frames = [i for i in range(N) if solution.get_values(f"x_{i}") > 0.5]
#     max_score = solution.get_objective_value()

#     # 释放CPLEX资源
#     problem.end()
#     return selected_frames, max_score


def cplex_frame_selection(A, K, rank):
    """
    K: 需要选择的帧数
    A: N x N 的得分矩阵 (对称矩阵，主对角线为0)
    """
    N = A.shape[0]
    device = A.device
    
    # 创建 CPLEX 求解器
    problem = cplex.Cplex()
    # 关闭所有日志输出
    problem.set_log_stream(None)
    problem.set_error_stream(None)
    problem.set_warning_stream(None)
    problem.set_results_stream(None)

    problem.parameters.threads.set(16) 
    problem.parameters.parallel.set(1)
    problem.parameters.timelimit.set(30)  # 限制 CPLEX 最多运行 30 秒
    problem.parameters.mip.limits.nodes.set(30000)  # 限制最大搜索节点数
    problem.parameters.mip.strategy.heuristicfreq.set(5)  # 让 CPLEX 更频繁使用启发式搜索
    problem.parameters.mip.strategy.rinsheur.set(10)  # 尝试改进已有解
    problem.parameters.mip.tolerances.mipgap.set(0.001)  # 允许更小误差，提高求解质量
    problem.parameters.mip.limits.solutions.set(10)  # 让 CPLEX 至少找到 10 个解
    problem.parameters.mip.strategy.variableselect.set(3)  # 让 CPLEX 优先选择重要变量

    problem.set_problem_type(cplex.Cplex.problem_type.LP)
    problem.objective.set_sense(problem.objective.sense.maximize)

    # 定义变量 X（是否选择帧）
    x_vars = [f"x_{i}" for i in range(N)]
    problem.variables.add(names=x_vars, types=[problem.variables.type.binary] * N)
    
    # 目标函数系数 (1/2 * X^T A X)
    obj_coeffs = []
    for i in range(N):
        for j in range(N):
            if i < j:  # 只存储上三角部分，避免重复
                obj_coeffs.append((f"x_{i}", f"x_{j}", float(A[i][j])))

    # 添加目标函数（使用二次项）
    problem.objective.set_quadratic_coefficients(obj_coeffs)

    # 约束：选取 K 帧
    problem.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=x_vars, val=[1] * N)],
        senses=["E"],
        rhs=[K]
    )

    # 求解
    problem.solve()

    # 输出结果
    solution = problem.solution
    selected_frames = [i for i in range(N) if solution.get_values(f"x_{i}") > 0.5]
    max_score = solution.get_objective_value()

    # 释放CPLEX资源
    problem.end()
    return selected_frames, max_score


def ILP_select_nodes(matrix, K, shrink_K, rank):
    '''
    init: denotes for a specified initialization
    '''
    device = matrix.device
    n = matrix.shape[0]

    sym_matrix = matrix + matrix.T 
    sym_matrix = sym_matrix.to(device)

    if n <= 4*K:
        # start = torch.argmax(matrix.diagonal()).item()
        # selected = {start}
        indices_mapping = {i: i for i in range(n)}
    # elif n > 4*K and n <= 128:
    #     # SVD sparsing
    #     U, Sigma, Vt = torch.svd(sym_matrix)
    #     Sigma_k = torch.diag(Sigma[:n//4])
    #     U_k = U[:, :n//4]
    #     Vt_k = Vt[:, :n//4]
    #     S_k = U_k @ Sigma_k @ Vt_k.T

    #     # low rank approximation
    #     sparse_matrix, true_indice, indices_mapping = uniform_select_matrix(S_k, M=n//4)  # sample n//4 nodes
    #     sym_matrix = sparse_matrix
    else:       # > 128
        if n > 7200:   # if there are 2h-long videos, doing SVD may cause OOM
            S_k = sym_matrix
        else:
            # SVD sparsing
            U, Sigma, Vt = torch.svd(sym_matrix)
            Sigma_k = torch.diag(Sigma[:n//4])       # 128 main components
            U_k = U[:, :n//4]
            Vt_k = Vt[:, :n//4]
            S_k = U_k @ Sigma_k @ Vt_k.T

        # low rank approximation
        sparse_matrix, true_indice, indices_mapping = uniform_select_matrix(S_k, M=shrink_K)  # sample 128 nodes
        sym_matrix = sparse_matrix

    sym_matrix.fill_diagonal_(0)
    # ILP
    solution, score = cplex_frame_selection(sym_matrix, K, rank)    #

    # selected = x_relaxed.nonzero().squeeze().tolist()
    true_select = [indices_mapping[i] for i in solution]
    return sorted(true_select)




def local_search_indices(x_selected, A, search_radius=2, rank=0):
    """
    对选取的 K 帧索引进行局部搜索优化，寻找更优解。
    
    Args:
        x_selected (list of int): 长度 K 的索引列表，表示已选取的 K 帧
        A (torch.Tensor): N x N 的对称得分矩阵
        search_radius (int): 搜索范围 δ，即向前/向后最大移动的步数
    
    Returns:
        list of int: 长度 K 的新索引列表
    """
    N = A.shape[0]
    # x_selected = set(x_selected)  # 转换为集合，加速查找
    # x_selected_new = list(x_selected)
    x_selected_new = x_selected
    A.fill_diagonal_(0)     # 仅包含上三角矩阵，不包括对角线

    for i in range(len(x_selected_new)):
        idx = x_selected_new[i]
        best_idx = idx
        best_value = A[x_selected_new][:, x_selected_new].sum()  # 计算当前目标值 KxK scores

        # 在 idx ± δ 范围内寻找更优索引
        for shift in range(-search_radius, search_radius + 1):
            new_idx = idx + shift
            if new_idx < 0 or new_idx >= N or new_idx in x_selected:
                continue  # 跳过无效索引

            # 替换 idx -> new_idx，计算新得分
            new_x_selected = x_selected_new[:]
            new_x_selected[i] = new_idx  # 替换索引
            new_value = A[new_x_selected][:, new_x_selected].sum()

            if new_value > best_value:  # 只接受更优解
                best_idx = new_idx
                best_value = new_value
        
        # 更新选择
        x_selected_new[i] = best_idx
    return x_selected_new  




import itertools
def max_k_subset_greedy_search(matrix, K, rank, candidate_factor=4, iterations=100):
    """
    在给定的上三角矩阵中找到得分最高的K个节点子集（使用贪心 + 局部搜索）。
    
    :param matrix: PyTorch 上三角矩阵 (n x n)，代表节点之间的得分
    :param K: 选取的节点数量
    :param candidate_factor: 候选池大小（默认 4K）
    :param iterations: 局部搜索的迭代次数
    :return: 得分最高的子集及其分数
    """
    device = matrix.device
    n = matrix.shape[0]  # 获取节点数

    # 计算每个节点的总边权（得分贡献）
    node_scores = matrix.sum(dim=1) + matrix.sum(dim=0)

    # 选择得分前 M = candidate_factor * K 的节点作为候选集
    M = min(candidate_factor * K, n)
    top_candidates = torch.argsort(node_scores, descending=True)[:M]

    # 贪心构造初始解
    best_subset = []
    while len(best_subset) < K:
        max_score = float('-inf')
        best_node = None
        for node in top_candidates:
            if node.item() not in best_subset:
                new_subset = best_subset + [node.item()]
                subset_score = sum(matrix[i, j] for i, j in itertools.combinations(new_subset, 2))
                if subset_score > max_score:
                    max_score = subset_score
                    best_node = node.item()
        best_subset.append(best_node)

    best_score = sum(matrix[i, j] for i, j in itertools.combinations(best_subset, 2))

    # **局部搜索优化**
    for _ in range(iterations):
        for i in range(K):
            for new_node in top_candidates:
                new_node = new_node.item()
                if new_node not in best_subset:
                    temp_subset = best_subset[:]
                    temp_subset[i] = new_node  # 替换一个节点
                    temp_score = sum(matrix[i, j] for i, j in itertools.combinations(temp_subset, 2))
                    if temp_score > best_score:  # 只有更优才替换
                        best_subset = temp_subset
                        best_score = temp_score

    return best_subset


import torch.linalg as linalg
from sklearn.cluster import KMeans
import numpy as np

# 假设你的相似度矩阵是一个n x n的矩阵，使用torch初始化
def spectral_clustering_score_matrix(score_matrix, num_clusters=3):
    device = score_matrix.device 

    # 步骤 1: 计算度矩阵 D
    degree_matrix = torch.sum(score_matrix, dim=1)
    degree_matrix = torch.diag(degree_matrix)

    # 步骤 2: 计算拉普拉斯矩阵 L
    laplacian_matrix = degree_matrix - score_matrix

    # 步骤 3: 计算拉普拉斯矩阵的特征分解
    eigenvalues, eigenvectors = linalg.eigh(laplacian_matrix)

    # 选择前k个最小的特征向量
    k = num_clusters
    eigenvectors_selected = eigenvectors[:, :k]

    # 步骤 4: 对选出的特征向量进行K-means聚类
    eigenvectors_selected_cpu = eigenvectors_selected.cpu().numpy()  # 转换为numpy数组
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(eigenvectors_selected_cpu)

    return labels


def select_center_node(similarity_matrix, labels):
    device = similarity_matrix.device
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    # 获取标签的唯一值（即每个子图）
    unique_labels = torch.unique(torch.tensor(labels, dtype=torch.long, device=device))
    centers = []

    for label in unique_labels:
        # 获取所有属于该子图的节点索引
        indices = torch.nonzero(labels == label).squeeze()
        
        if indices.numel() == 1:
            # 如果子图只有一个节点，那么这个节点即为中心
            center_node_idx = indices
        else:
            # 计算子图内每个节点的相似度之和
            subgraph_similarity = similarity_matrix[indices][:, indices]
            node_similarity_sum = subgraph_similarity.sum(dim=1)

            # 选择与其他节点相似度总和最大的节点作为中心节点
            center_node_idx = indices[torch.argmax(node_similarity_sum)]

        centers.append(center_node_idx.item())  # 将张量转为原始值并添加到列表中
    return centers


# 构建优化后的相似度矩阵
def build_optimized_matrix(similarity_matrix, centers):
    device = similarity_matrix.device
    centers = sorted(centers)
    num_centers = len(centers)
    optimized_matrix = torch.zeros((num_centers, num_centers), dtype=torch.float32, device=device)

    for i in range(num_centers):
        for j in range(i, num_centers):
            # 计算子图之间的相似度，这里使用中心节点之间的相似度
            optimized_matrix[i, j] = similarity_matrix[centers[i], centers[j]]
            optimized_matrix[j, i] = optimized_matrix[i, j]  # 保持对称性
    
    indices_mapping = {i: centers[i] for i in range(len(centers))}
    return optimized_matrix, indices_mapping


def incremental_select_spectral_cluster(matrix, K, rank):
    """
    增量式选择：
    1. 首先计算每个节点的加权度（该节点与所有其他节点的边权之和）。
    2. 选择加权度最大的节点作为起点。
    3. 当选中节点数 < K 时，对于所有未选节点，
       计算它与当前已选节点之间的边权总和，选取贡献最大的节点加入。
    返回选中的节点（按升序排列）。
    """
    device = matrix.device
    n = matrix.shape[0]


    sym_matrix = matrix + matrix.T 
    sym_matrix = sym_matrix.to(device)
    sym_matrix.fill_diagonal_(0)

    if n <= 4*K:
        start = torch.argmax(matrix.diagonal()).item()
        selected = {start}
        indices_mapping = {i: i for i in range(n)}
    else:   
        # spectral clustering
        labels = spectral_clustering_score_matrix(sym_matrix, num_clusters=n//4)
        centers = select_center_node(sym_matrix, labels)
        sparse_matrix, indices_mapping = build_optimized_matrix(sym_matrix, centers)     # should be n//4 x n//4

        start = torch.argmax(sparse_matrix.diagonal()).item()
        selected = {start}
        sym_matrix = sparse_matrix
    

    while len(selected) < K:
        best_increment = -float('inf')
        best_node = None
        for i in range(sym_matrix.shape[0]):
            if i in selected:
                continue
            # 计算 i 与当前 selected 中所有节点之间的边权总和
            inc = 0.0
            for j in selected:
                if i < j:
                    inc += sym_matrix[i, j].item()
                else:
                    inc += sym_matrix[j, i].item()
            if inc > best_increment:
                best_increment = inc
                best_node = i
        selected.add(best_node)

    true_select = [indices_mapping[i] for i in selected]
    return sorted(true_select)