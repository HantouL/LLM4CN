import random
import multiprocessing
import time
from utils import sample_by_degree_distribution, calculate_anc, calculate_anc_gcc, sorted_by_value
from scipy.sparse import load_npz


# 评估节点评分函数的性能，通过移除关键节点后网络连通性的变化计算得分。
class ScoringModule:

    def __init__(self, file_path, ratio, calculate_type='pc'):
        self.edge_matrix = load_npz(file_path).toarray()
        if len(self.edge_matrix) > 5000:
            self.edge_matrix = sample_by_degree_distribution(self.edge_matrix, 0.1)
        self.number_of_removed_nodes = int(ratio * 0.01 * len(self.edge_matrix))
        self.metric = calculate_type
        print(f'removed number: {self.number_of_removed_nodes}, metric: {self.metric}')

    def _run_algorithm(self, result_queue, algorithm, edge_matrix, number_of_removed_nodes, metric):
        """在子进程中运行算法并计算得分"""
        globals_dict = {}
        try:
            # 使用 exec() 将字符串代码加载到 globals_dict，提取 score_nodes 函数
            exec(algorithm, globals_dict)
            # 调用 score_nodes(self.edge_matrix)，生成节点得分字典 result_dict
            result_dict = globals_dict['score_nodes'](edge_matrix)
            # 使用 sorted_by_value 按得分降序排序，返回 (node_id, score) 列表
            result = sorted_by_value(result_dict)
            if metric == 'pc':
                # 利用 ANC，计算移除节点后所有连通分量的成对连接数比例（比例越小拆的越好）
                score = 1 - calculate_anc(edge_matrix, result[:number_of_removed_nodes])
            elif metric == 'gcc':
                # 利用 GCC，计算移除节点后最大连通分量大小比例（比例越小拆的越好）
                score = 1 - calculate_anc_gcc(edge_matrix, result[:number_of_removed_nodes])
            else:
                score = 0
            result_queue.put(score)
        except Exception as e:
            print("This code can not be evaluated!")
            result_queue.put(0)

    def score_nodes_with_timeout(self, algorithm, timeout=60):
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._run_algorithm,
            args=(result_queue, algorithm, self.edge_matrix, self.number_of_removed_nodes, self.metric)
        )

        process.start()
        process.join(timeout)

        if process.is_alive():
            print("Algorithm execution exceeded timeout. Terminating...")
            process.terminate()
            process.join()
            return 0
        else:
            return result_queue.get()

    # def score_nodes_with_timeout(self, algorithm, timeout=60):
    #     def run_algorithm(result_queue):
    #         try:
    #             # 使用 exec() 将字符串代码加载到 globals_dict，提取 score_nodes 函数
    #             exec(algorithm, globals_dict)
    #             # 调用 score_nodes(self.edge_matrix)，生成节点得分字典 result_dict
    #             result_dict = globals_dict['score_nodes'](self.edge_matrix)
    #             # 使用 sorted_by_value 按得分降序排序，返回 (node_id, score) 列表
    #             result = sorted_by_value(result_dict)
    #             if self.metric == 'pc':
    #                 # 利用ANC，计算移除节点后所有联通分量的成对连接数比例(比例越小拆的越好)
    #                 score = 1 - calculate_anc(self.edge_matrix, result[:self.number_of_removed_nodes])
    #             elif self.metric == 'gcc':
    #                 # 利用GCC，计算移除节点后计算最大连通分量大小比例(比例越小拆的越好)
    #                 score = 1 - calculate_anc_gcc(self.edge_matrix, result[:self.number_of_removed_nodes])
    #             else:
    #                 score = 0
    #             result_queue.put(score)
    #         except Exception as e:
    #             print("This code can not be evaluated!")
    #             result_queue.put(0)
    #
    #     globals_dict = {}
    #     result_queue = multiprocessing.Queue()
    #     process = multiprocessing.Process(target=run_algorithm, args=(result_queue,))
    #
    #     process.start()
    #     process.join(timeout)
    #
    #     if process.is_alive():
    #         print("Algorithm execution exceeded timeout. Terminating...")
    #         process.terminate()
    #         process.join()
    #         return 0
    #     else:
    #         return result_queue.get()

    # 调用 score_nodes_with_timeout 评估算法并返回得分
    def evaluate_algorithm(self, algorithm):
        score = self.score_nodes_with_timeout(algorithm, timeout=60)
        return score


if __name__ == '__main__':
    print('Score Module')
