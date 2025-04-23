import random
import pickle
import csv
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from initialization import InitializationModule
from crossover import CrossoverModule
from mutation import MutationModule
from score import ScoringModule
from pool import AlgorithmPool

if __name__ == '__main__':
    # 通过命令行配置参数运行模式
    '''
        train_type：训练类型（new 表示从头开始，continue 表示加载已有模型继续训练）。
        pool_path：已有算法池的文件路径。
        dataset：数据集名称（如 jazz）。
        ratio：移除节点的百分比（默认20%）。
        calculate：评估指标类型（pc 表示 pairwise connectivity，gcc 表示最大连通分量）。
    '''
    parser = argparse.ArgumentParser(description="LLM for Combinatorial Optimization")
    parser.add_argument("--train_type", type=str, default='new', help="Training type: new or continue")
    parser.add_argument("--pool_path", type=str, default='algorithms_25.pkl', help="File path of gained algorithm pool")
    parser.add_argument("--dataset", type=str, default='jazz', help="Dataset")
    parser.add_argument("--ratio", type=int, default=20, help="Fraction*100 of removed nodes")
    parser.add_argument("--calculate", type=str, default='pc', help="pc of gcc")
    args = parser.parse_args()

    number_population = 10  # 初始群体数量
    pool_size = 10  # 每个群体最大容量
    number_of_init_algorithms = 12  # 初始算法数量
    number_for_crossover = 4  # 交叉时采样算法数量
    target_score = 1  # 目标得分
    target_epoch = 100  # 最大迭代轮数
    prob_mutation = 0.3  # 变异概率
    similarity_lower_threshold = 0.93  # 相似性下限
    similarity_upper_threshold = 0.99  # 相似性上限
    file_path = './results/algorithms_pool.pkl'  # 保存路径
    dataset_path = f"./datasets/{args.dataset}_adj_matrix.npz"  # 数据集路径
    model_name = 'bert-base-uncased'  # 嵌入模型
    Total_Cost = 0  # API调用成本

    # 提示词，引导LLM来生成评分代码
    # 定义问题背景，但未在交叉和变异提示中详细描述网络，以避免干扰LLMs的代码生成。
    task = "Given a edge matrix of a network, you need to find the key nodes in the network. These nodes, when removed from the network, result in a decrease in the size of the largest connected component of the network.\n"
    prompt_crossover = "I have two codes as follows:\n"
    prompt_initial = "Please provide a new algorithm.\n"
    prompt_mutation = "Without changing the input and output of this code, modify this code to make node scoring more reasonable:\n"
    # 明确输入输出格式（邻接矩阵和节点得分字典）。
    prompt_code_request = "Mix the two algorithms above, and create a completely different better Python function called \"score_nodes\" that accepts an \"edge_matrix\" as input and returns \"scored_nodes\" as output. \"edge_matrix\" should be a adjacency matrix in the form of a NumPy array, and \"scored_nodes\" should be a dictionary where the keys are node IDs and the values are node scores."
    extra_prompt = "Provide only one Python function, not any explanation."

    # 算法池
    algorithm_pool = AlgorithmPool(
        number_population=number_population,
        capacity=pool_size,
        model_name=model_name,
        lower_threshold=similarity_lower_threshold,
        upper_threshold=similarity_upper_threshold
    )
    # 初始化模块
    initialization_module = InitializationModule(
        task=task,
        prompt_init=prompt_initial,
        prompt_code_request=prompt_code_request,
        extra_prompt=extra_prompt,
        handmade=True
    )
    # 混合模块
    crossover_module = CrossoverModule(
        task=task,
        prompt_crossover=prompt_crossover,
        prompt_code_request=prompt_code_request,
        extra_prompt=extra_prompt
    )
    # 变异模块
    mutation_module = MutationModule(
        task=task,
        prompt_mutation=prompt_mutation,
        extra_prompt=extra_prompt
    )
    # 评分模块
    scoring_module = ScoringModule(
        file_path=dataset_path,
        ratio=args.ratio,
        calculate_type=args.calculate
    )

    # 如果是新训练
    if args.train_type == 'new':
        initial_algorithms, cost = initialization_module.generate_initial_algorithms(count=number_of_init_algorithms)
        Total_Cost += 0

        print('Initialization!')
        for algorithm in tqdm(initial_algorithms, desc='Scoring Initial Codes'):
            score = scoring_module.evlaluate_algorithm(algorithm)
            algorithm_pool.add_algorithm(algorithm, score, f"Population_{len(algorithm_pool.pool)}", 0)
        print("Evaluate Initial Algorithms")
    # 如果是继续训练
    else:
        algorithm_pool.load_algorithm(args.pool_path)

    average_scores = []
    best_scores = []
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m-%d %H:%M")
    folder_path = f'./results/algorithms_{args.dataset}_{args.ratio}_{args.train_type}_{formatted_time}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    population_statistics = []

    ##################### 主循环 - 进化算法实现 ######################
    for epoch in tqdm(range(target_epoch), desc='Epoch'):
        # 1.混合部分
        print("Begin Crossover!")
        sampled_algorithms = algorithm_pool.sample_algorithms(number_for_crossover)
        best_algorithm = algorithm_pool.get_best_algorithm()
        # 如果新的最佳算法没被记录就记一下
        if best_algorithm not in sampled_algorithms:
            sampled_algorithms.append(best_algorithm)

        # 计算并累计开销
        crossed_algorithms, cost = crossover_module.crossover_algorithms(sampled_algorithms)
        Total_Cost += cost
        # 开始变异
        for algorithm in crossed_algorithms:
            if random.uniform(0, 1) < prob_mutation:
                algorithm, cost = mutation_module.mutate_algorithm(algorithm)
                Total_Cost += cost
            score = scoring_module.evlaluate_algorithm(algorithm)
            population_label, similarity = algorithm_pool.algorithm_classification(algorithm)
            algorithm_pool.add_algorithm(algorithm, score, population_label, similarity)

        # 2. 自我进化阶段
        print("Begin Self-Evolution!")
        epoch_stats = {
            "epoch": epoch,
            "population_stats": []
        }
        for population_label, population in algorithm_pool.pool.items():
            if len(algorithm_pool.pool[population_label]) > 1:
                # 对每个群体内部采样2个算法进行交叉和变异
                sampled_algorithms = algorithm_pool.sample_algorithms_population(population_label, 2)
                # 交叉
                algorithm, cost = crossover_module.crossover_algorithms(sampled_algorithms)
                algorithm = algorithm[0]
                Total_Cost += cost
                # 变异
                if random.uniform(0, 1) < prob_mutation:
                    algorithm, cost = mutation_module.mutate_algorithm(algorithm)
                    Total_Cost += cost
                score = scoring_module.evlaluate_algorithm(algorithm)
                algorithm_pool.add_algorithm(algorithm, score, population_label, 1)

            population_size = len(population)
            ave_score_population = algorithm_pool.calculate_average_score(population_label)
            best_score_population = algorithm_pool.get_highest_score(population_label)

            population_stats = {
                "label": population_label,
                "size": population_size,
                "average_score": ave_score_population,
                "highest_score": best_score_population
            }

            epoch_stats["population_stats"].append(population_stats)

        # 3.统计与保存 种群大小、平均分、最高分
        population_statistics.append(epoch_stats)
        ave_score = algorithm_pool.calculate_average_score()
        best_score = algorithm_pool.get_highest_score()
        average_scores.append(ave_score)
        best_scores.append(best_score)

        print(
            f"Average Score: {ave_score:.4f} Best Score: {best_score:.4f} Population Number: {len(algorithm_pool.pool)}, Pool Size: {algorithm_pool.__len__()} Total Cost: {Total_Cost:.4f}$")

        algorithm_pool.save_algorithms(folder_path + f'/algorithms_{epoch}.pkl') # 保存算法池
        score_to_save = {
            "ave": average_scores,
            "best": best_scores
        }
        with open(folder_path + '/score_data.pkl', 'wb') as file:
            pickle.dump(score_to_save, file)
        with open(folder_path + '/population_data.pkl', 'wb') as file:
            pickle.dump(population_statistics, file)

        epoch += 1
        print('Data saved!')

        if best_score >= target_score:
            print('Success!')
            break

    # 4.成本记录
    with open('./results/API_cost.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([formatted_time, Total_Cost])
