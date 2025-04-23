from utils import LLM_generate_algorithm

prompt = """Create a Python function called \"score_nodes\" that accepts an \"edge_matrix\" (NumPy array) as input and returns \"scored_nodes\" (dictionary with node IDs as keys and scores as values)."""
algorithm, cost = LLM_generate_algorithm(prompt, stochasticity=1)
print("生成算法：\n", algorithm)
print("成本：", cost)