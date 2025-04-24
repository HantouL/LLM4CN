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
    print("解析命令行参数...")
    parser = argparse.ArgumentParser(description="LLM for Combinatorial Optimization")
    parser.add_argument("--train_type", type=str, default='new', help="Training type: new or continue")
    parser.add_argument("--pool_path", type=str, default='algorithms_25.pkl', help="File path of gained algorithm pool")
    parser.add_argument("--dataset", type=str, default='jazz', help="Dataset")
    parser.add_argument("--ratio", type=int, default=20, help="Fraction*100 of removed nodes")
    parser.add_argument("--calculate", type=str, default='pc', help="pc or gcc")
    args = parser.parse_args()
    print(
        f"参数解析完成：train_type={args.train_type}, dataset={args.dataset}, ratio={args.ratio}, calculate={args.calculate}")

    # 参数设置
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
    print("参数设置完成！")

    # 提示词，引导LLM来生成评分代码
    task = "Given an edge matrix of a network, you need to find the key nodes in the network. These nodes, when removed from the network, result in a decrease in the size of the largest connected component of the network.\n"
    prompt_crossover = "I have two codes as follows:\n"
    prompt_initial = "Please provide a new algorithm.\n"
    prompt_mutation = "Without changing the input and output of this code, modify this code to make node scoring more reasonable:\n"
    prompt_code_request = "Mix the two algorithms above, and create a completely different better Python function called \"score_nodes\" that accepts an \"edge_matrix\" as input and returns \"scored_nodes\" as output. \"edge_matrix\" should be a adjacency matrix in the form of a NumPy array, and \"scored_nodes\" should be a dictionary where the keys are node IDs and the values are node scores."
    extra_prompt = "Provide only one Python function, not any explanation."
    print("提示词设置完成！")

    # 初始化模块
    print("初始化 AlgorithmPool...")
    algorithm_pool = AlgorithmPool(
        number_population=number_population,
        capacity=pool_size,
        model_name=model_name,
        lower_threshold=similarity_lower_threshold,
        upper_threshold=similarity_upper_threshold
    )
    print("AlgorithmPool 初始化完成！")

    print("初始化 InitializationModule...")
    initialization_module = InitializationModule(
        task=task,
        prompt_init=prompt_initial,
        prompt_code_request=prompt_code_request,
        extra_prompt=extra_prompt,
        handmade=True
    )
    print("InitializationModule 初始化完成！")

    print("初始化 CrossoverModule...")
    crossover_module = CrossoverModule(
        task=task,
        prompt_crossover=prompt_crossover,
        prompt_code_request=prompt_code_request,
        extra_prompt=extra_prompt
    )
    print("CrossoverModule 初始化完成！")

    print("初始化 MutationModule...")
    mutation_module = MutationModule(
        task=task,
        prompt_mutation=prompt_mutation,
        extra_prompt=extra_prompt
    )
    print("MutationModule 初始化完成！")

    print("初始化 ScoringModule...")
    scoring_module = ScoringModule(
        file_path=dataset_path,
        ratio=args.ratio,
        calculate_type=args.calculate
    )
    print("ScoringModule 初始化完成！")

    # 如果是新训练
    if args.train_type == 'new':
        print(f"开始生成 {number_of_init_algorithms} 个初始算法...")
        initial_algorithms, cost = initialization_module.generate_initial_algorithms(count=number_of_init_algorithms)
        Total_Cost += cost  # 修正：从 0 改为 cost
        print(f"初始算法生成完成，生成 {len(initial_algorithms)} 个算法，成本：{cost}")

        print('初始化算法池！')
        for idx, algorithm in enumerate(tqdm(initial_algorithms, desc='评分初始算法')):
            print(f"评分第 {idx + 1} 个初始算法...")
            score = scoring_module.evaluate_algorithm(algorithm)
            print(f"第 {idx + 1} 个初始算法得分：{score}")
            population_label = f"Population_{len(algorithm_pool.pool)}"
            algorithm_pool.add_algorithm(algorithm, score, population_label, 0)
            print(f"已添加第 {idx + 1} 个初始算法到算法池，群体标签：{population_label}")
        print("初始算法评分和添加完成！")
    # 如果是继续训练
    else:
        print(f"从 {args.pool_path} 加载已有算法池...")
        algorithm_pool.load_algorithm(args.pool_path)
        print("算法池加载完成！")

    average_scores = []
    best_scores = []
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m-%d %H:%M")
    folder_path = f'./results/algorithms_{args.dataset}_{args.ratio}_{args.train_type}_{formatted_time}'
    if not os.path.exists(folder_path):
        print(f"创建结果保存目录：{folder_path}")
        os.makedirs(folder_path)
    population_statistics = []
    print("结果保存目录准备完成！")

    ##################### 主循环 - 进化算法实现 ######################
    print(f"开始主循环，最大轮数：{target_epoch}")
    for epoch in tqdm(range(target_epoch), desc='轮次'):
        print(f"\n=== 第 {epoch + 1} 轮 ===")

        # 1. 混合部分
        print(f"开始交叉，采样 {number_for_crossover} 个算法...")
        sampled_algorithms = algorithm_pool.sample_algorithms(number_for_crossover)
        best_algorithm = algorithm_pool.get_best_algorithm()
        print(f"当前最佳算法得分：{algorithm_pool.get_highest_score()}")
        if best_algorithm not in sampled_algorithms:
            sampled_algorithms.append(best_algorithm)
            print("已添加最佳算法到采样列表")

        print("执行交叉操作...")
        crossed_algorithms, cost = crossover_module.crossover_algorithms(sampled_algorithms)
        Total_Cost += cost
        print(f"交叉完成，生成 {len(crossed_algorithms)} 个新算法，成本：{cost}")

        # 开始变异和评分
        for idx, algorithm in enumerate(crossed_algorithms):
            print(f"处理第 {idx + 1} 个交叉算法...")
            if random.uniform(0, 1) < prob_mutation:
                print("执行变异...")
                algorithm, cost = mutation_module.mutate_algorithm(algorithm)
                Total_Cost += cost
                print(f"变异完成，成本：{cost}")
            print("评分交叉算法...")
            score = scoring_module.evaluate_algorithm(algorithm)
            print(f"第 {idx + 1} 个交叉算法得分：{score}")
            population_label, similarity = algorithm_pool.algorithm_classification(algorithm)
            print(f"交叉算法分类，群体标签：{population_label}，相似度：{similarity}")
            algorithm_pool.add_algorithm(algorithm, score, population_label, similarity)
            print(f"已添加第 {idx + 1} 个交叉算法到算法池")

        # 2. 自我进化阶段
        print("开始自我进化阶段...")
        epoch_stats = {
            "epoch": epoch,
            "population_stats": []
        }
        for population_label, population in algorithm_pool.pool.items():
            if len(algorithm_pool.pool[population_label]) > 1:
                print(f"群体 {population_label} 自我进化，采样 2 个算法...")
                sampled_algorithms = algorithm_pool.sample_algorithms_population(population_label, 2)
                print("执行群体内部交叉...")
                algorithm, cost = crossover_module.crossover_algorithms(sampled_algorithms)
                algorithm = algorithm[0]
                Total_Cost += cost
                print(f"群体内部交叉完成，成本：{cost}")
                if random.uniform(0, 1) < prob_mutation:
                    print("执行群体内部变异...")
                    algorithm, cost = mutation_module.mutate_algorithm(algorithm)
                    Total_Cost += cost
                    print(f"群体内部变异完成，成本：{cost}")
                print("评分群体内部算法...")
                score = scoring_module.evaluate_algorithm(algorithm)
                print(f"群体内部算法得分：{score}")
                print(f"群体内部算法得分：{score}")
                algorithm_pool.add_algorithm(algorithm, score, population_label, 1)
                print(f"已添加群体内部算法到 {population_label}")

            population_size = len(population)
            ave_score_population = algorithm_pool.calculate_average_score(population_label)
            best_score_population = algorithm_pool.get_highest_score(population_label)
            print(
                f"群体 {population_label} 统计：大小={population_size}，平均得分={ave_score_population:.4f}，最高得分={best_score_population:.4f}")

            population_stats = {
                "label": population_label,
                "size": population_size,
                "average_score": ave_score_population,
                "highest_score": best_score_population
            }
            epoch_stats["population_stats"].append(population_stats)

        # 3. 统计与保存 种群大小、平均分、最高分
        print("统计和保存结果...")
        population_statistics.append(epoch_stats)
        ave_score = algorithm_pool.calculate_average_score()
        best_score = algorithm_pool.get_highest_score()
        average_scores.append(ave_score)
        best_scores.append(best_score)

        print(
            f"第 {epoch + 1} 轮统计：平均得分={ave_score:.4f}，最佳得分={best_score:.4f}，群体数量={len(algorithm_pool.pool)}，池大小={algorithm_pool.__len__()}，总成本={Total_Cost:.4f}$")

        print(f"保存算法池到 {folder_path}/algorithms_{epoch}.pkl...")
        algorithm_pool.save_algorithms(folder_path + f'/algorithms_{epoch}.pkl')
        score_to_save = {
            "ave": average_scores,
            "best": best_scores
        }
        with open(folder_path + '/score_data.pkl', 'wb') as file:
            pickle.dump(score_to_save, file)
        with open(folder_path + '/population_data.pkl', 'wb') as file:
            pickle.dump(population_statistics, file)
        print("数据保存完成！")

        if best_score >= target_score:
            print("成功！最佳得分达到目标！")
            break

    # 4. 成本记录
    print("记录 API 成本...")
    with open('./results/API_cost.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([formatted_time, Total_Cost])
    print("API 成本记录完成！")