import random
import pickle # 序列化存储算法池
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity # 计算余弦相似度
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 种群管理
class AlgorithmPool:
    def __init__(self, number_population, capacity, lower_threshold, upper_threshold, model_name=None):
        self.number_population = number_population  # 初始群体数量
        self.capacity = capacity  # 每个群体最大容量
        self.pool = {}
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        if self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

    # 新增算法
    def add_algorithm(self, algorithm, score, population_label, similarity):
        if population_label not in self.pool:
            self.pool[population_label] = []
        population = self.pool[population_label]

        # 如果相似度过高且得分已存在，就直接跳过
        if similarity > self.upper_threshold:
            scores = [algorithm[1] for algorithm in population]
            if score in scores:
                return
        else:
            # 若族群满了，就替换掉得分最低的算法
            if len(population) >= self.capacity:
                population.sort(key=lambda x: x[1])
                if population[0][1] < score:
                    population[0] = (algorithm, score)
            else:
                population.append((algorithm, score))

    # 从文件加载算法池
    def load_algorithm(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                algorithms_data = pickle.load(file)
        except Exception as e:
            print(f"Loading failed: {e}")
            return
        for label, algorithms in algorithms_data.items():
            self.pool[label] = []
            for algorithm in algorithms:
                self.pool[label].append(algorithm)
        print("Loading Succeeded")
    # 将代码转换为嵌入向量
    def calculate_embedding(self, code):
        if self.model and self.tokenizer:
            inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.numpy()[0]
        else:
            return None

    # 对新算法进行分类，决定其族群归属
    def algorithm_classification(self, new_algorithm):
        if not self.pool:
            return None
        new_algorithm_embedding = self.calculate_embedding(new_algorithm)

        max_similarity = -1
        most_similar_population = None

        # 计算新算法的嵌入
        for population_label, population in self.pool.items():
            if not population:
                continue
            # 与每个群体的首个算法比较相似度，找到最相似群体。
            initial_algorithm = population[0][0]
            initial_algorithm_embedding = self.calculate_embedding(initial_algorithm)
            if new_algorithm is not None and initial_algorithm_embedding is not None:
                similarity = cosine_similarity([new_algorithm_embedding], [initial_algorithm_embedding])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_population = population_label
        # 若最大相似度超过 lower_threshold，归入该群体；否则创建新群体
        if max_similarity >= self.lower_threshold:
            return most_similar_population,max_similarity
        else:
            new_population_label = f"Population_{len(self.pool)}"
            return new_population_label, max_similarity

    # 从指定群体采样 number 个算法，得分作为权重。
    def sample_algorithms_population(self, population_label, number):
        if population_label not in self.pool:
            return []
        population = self.pool[population_label]
        
        if number > len(population):
            number = len(population)

        weights = [score for (_, score) in population]
        sampled_algorithms = random.choices(population, weights=weights, k=number)
        return sampled_algorithms

    # 从每个群体采样一个算法
    def sample_one_algorithm_per_population(self):
        sampled_algorithms = []
        for population in self.pool.values():
            if population:
                weights = [score for (_, score) in population]
                random_algorithm = random.choices(population, weights=weights)[0]
                sampled_algorithms.append(random_algorithm)
        return sampled_algorithms

    # 从所有算法中采样 number 个
    def sample_algorithms(self, number):
        all_algorithms = [algorithm for population in self.pool.values() for algorithm in population]
        weights = [score for _, score in all_algorithms]
        
        if number > len(all_algorithms):
            number = len(all_algorithms)
        sampled_algorithms = random.choices(all_algorithms, weights=weights, k= number)
        return sampled_algorithms

    # 返回得分最高的算法及其得分
    def get_best_algorithm(self):
        best_algorithm = None
        best_score = 0

        for population_label, population in self.pool.items():
            for algorithm, score in population:
                if score > best_score:
                    best_score = score
                    best_algorithm = algorithm

        return (best_algorithm, best_score)

    # 计算指定群体或整体平均得分
    def calculate_average_score(self, population_label=None):
        if not self.pool:
            return None

        if population_label:
            if population_label not in self.pool:
                return 0
            population = self.pool[population_label]
            if not population:
                return None
            total_score = sum(score for _, score in population)
            return total_score / len(population)
        else:
            total_score = sum(sum(score for _, score in population) for population in self.pool.values())
            return total_score / self.__len__()

    # 计算所有群体的平均得分
    def calculate_overall_average_score(self):
        if not self.pool:
            return 0.0
        total_score = sum(sum(score for _, score in population) for population in self.pool.values())
        return total_score / self.__len__()

    # 返回指定群体或整体最高得分
    def get_highest_score(self, population_label=None):
        if not self.pool:
            return None

        if population_label:
            if population_label not in self.pool:
                return 0
            population = self.pool[population_label]
            if not population:
                return None
            population.sort(key=lambda x: x[1], reverse=True)
            return population[0][1]
        else:
            highest_score = None
            for population in self.pool.values():
                if population:
                    population.sort(key=lambda x: x[1], reverse=True)
                    highest_in_population = population[0][1]
                    if highest_score is None or highest_in_population > highest_score:
                        highest_score = highest_in_population
            return highest_score

    def __len__(self):
        return sum(len(population) for population in self.pool.values())

    # 将算法池保存到文件
    def save_algorithms(self,file_path):
        with open(file_path,'wb') as file:
            pickle.dump(self.pool,file)
        print("Algorithms Saved!")    

if __name__ == '__main__':
    print('Function Pool')
