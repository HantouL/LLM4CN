import numpy as np
import networkx as nx
from openai import OpenAI
import re

# KEY = "Your OpenAI API Key"

# Kimi API 配置
KIMI_API_KEY = ""  # 替换为你的 Kimi API 密钥
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
Input_35 = 0.001
Output_35 = 0.002

# 初始化 Kimi 客户端
client = OpenAI(
    api_key=KIMI_API_KEY,
    base_url=KIMI_BASE_URL
)


# 从代码文本中移除函数定义行后的换行符和注释
def extract_function(text):
    pattern = r'(?<=def\s\w+\(.*\):)|#.*'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


# 验证生成的 score_nodes 函数是否可运行并输出合理结果
def check_code(code_str):
    try:
        globals_dict = {}
        exec(code_str, globals_dict)
        edge_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        all_results = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])
        result_dict = globals_dict['score_nodes'](edge_matrix)
        result = sorted_by_value(result_dict)
        flag = result in all_results
        if not flag:
            print("Code output is incorrect")
        return flag
    except Exception as e:
        print("Code can not run")
        return False

# 计算网络的成对连通性
def calculate_pairwise_connectivity(Graph):
    # 获取所有连通分量的大小
    size_of_connected_components = [len(part_graph) for part_graph in nx.connected_components(Graph)]
    element_of_pc = [size * (size - 1) / 2 for size in size_of_connected_components]
    pairwise_connectivity = sum(element_of_pc)
    return pairwise_connectivity

# 基于 pc 计算累积归一化连通性（ANC）
def calculate_anc(adjacency_matrix, nodes_to_remove):
    # 初始化网络并计算原始连通性
    G = nx.from_numpy_array(adjacency_matrix)
    original_pc = calculate_pairwise_connectivity(G)
    accumulated_ratios = 0.0

    # 逐一移除节点，计算每次移除后的连通性比例
    for node in nodes_to_remove:
        G.remove_node(node)
        current_pc = calculate_pairwise_connectivity(G)
        accumulated_ratios += current_pc / original_pc

    # 累加比例并平均。
    anc = accumulated_ratios / len(nodes_to_remove)
    return anc


# 调用kimi生成算法并验证。
def LLM_generate_algorithm(prompt, stochasticity=1):
    while True:
        try:
            completion = client.chat.completions.create(
                model="moonshot-v1-128k",  # 使用 Kimi 的长上下文模型
                messages=[{"role": "user", "content": prompt}],
                temperature=stochasticity,
                top_p=0.9
            )
            algorithm = completion.choices[0].message.content

            # 提取代码部分
            code_start = algorithm.find("```python") + 9
            code_end = algorithm.find("```", code_start)
            if code_start > 8 and code_end > code_start:
                algorithm = algorithm[code_start:code_end].strip()
            else:
                algorithm = algorithm.strip()  # 回退到原始输出

            # 计算成本
            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens
            # 示例定价
            input_price_per_1k = 0.001  # 假设 $0.001/1k tokens
            output_price_per_1k = 0.002  # 假设 $0.002/1k tokens
            all_cost = (input_tokens / 1000 * input_price_per_1k) + (output_tokens / 1000 * output_price_per_1k)

            if check_code(algorithm):
                print("Generated Algorithm Succeeded!")
                break
            else:
                print("Generated Algorithm Failed!")
        except Exception as e:
            print(f"API call failed: {e}")
            algorithm = ""
            all_cost = 0
            continue

    return algorithm, all_cost

# 计算最大连通分量（GCC）的大小
def calculate_size_of_gcc(Graph):
    size_of_connected_components = [len(part_graph) for part_graph in nx.connected_components(Graph)]
    size_of_gcc = max(size_of_connected_components)
    return size_of_gcc


# 计算基于GCC的ANC
def calculate_anc_gcc(adjacency_matrix, nodes_to_remove):
    G = nx.from_numpy_array(adjacency_matrix)
    original = calculate_size_of_gcc(G)
    accumulated_ratios = 0.0

    for node in nodes_to_remove:
        G.remove_node(node)
        current = calculate_size_of_gcc(G)
        accumulated_ratios += current / original

    anc = accumulated_ratios / len(nodes_to_remove)
    return anc


# 按值降序排序字典，返回键列表
def sorted_by_value(my_dict):
    sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_dict.keys())


# 按度分布采样子网络
def sample_by_degree_distribution(adj_matrix, sample_size_ratio):
    # 计算每个节点的度，归一化为概率
    degrees = adj_matrix.sum(axis=1)
    selection_probability = degrees / degrees.sum()
    num_nodes = len(degrees)
    sample_size = int(num_nodes * sample_size_ratio)
    # 使用 np.random.choice 采样节点
    sampled_nodes = np.random.choice(num_nodes, size=sample_size, replace=False, p=selection_probability)
    # 返回采样后的邻接矩阵
    sampled_adj_matrix = adj_matrix[sampled_nodes, :][:, sampled_nodes]
    return sampled_adj_matrix


if __name__ == '__main__':
    print("Utils")
