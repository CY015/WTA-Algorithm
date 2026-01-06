import numpy as np
import random
import time
from typing import List, Tuple, Dict

# Target和Ammunition类定义如下
class Target:
    def __init__(self, target_id: int, value: float, components: List[Tuple[float, float]]):
        self.id = target_id
        self.value = value
        self.components = components  # (权重, 健康度)列表

        # 1. 提取所有部件的初始重要度权重
        original_importances = [comp[0] for comp in components]
        
        # # 2. 验证初始重要度权重取值合法性（符合建模的0.2/0.5/0.8要求）
        # for imp in original_importances:
        #     if imp not in (0.2, 0.5, 0.8):
        #         raise ValueError(
        #             f"部件初始重要度权重{imp}不合法！仅支持Word文档建模定义的0.2（不重要）、0.5（中等）、0.8（重要）三种取值"
        #         )
        
        # 3. 计算初始重要度权重总和，避免除零错误
        total_original_importance = sum(original_importances)
        if total_original_importance == 0:
            raise ValueError("目标所有部件的初始重要度权重之和不能为0！请至少为一个部件分配有效权重")
        
        # 4. 执行归一化计算：单个部件最终重要度 = 该部件初始权重 / 所有部件初始权重之和（贴合Word文档公式）
        self.normalized_importances = [imp / total_original_importance for imp in original_importances]

    def calculate_damage_effect(self, ammo_damage_prob: List[float]) -> float:
        """计算弹药对该目标的毁伤效能（使用归一化后的重要度权重）"""
        total_effect = 0.0
        # 校验：毁伤概率列表长度需与部件数量一致
        if len(ammo_damage_prob) != len(self.components):
            raise ValueError(f"弹药毁伤概率列表长度（{len(ammo_damage_prob)}）与目标部件数量（{len(self.components)}）不匹配！")
        
        # 核心计算：使用归一化后的重要度权重（替换原始权重），健康度参数仍保留（虽未使用，不破坏原有格式）
        for (_, health), norm_importance, damage_prob in zip(self.components, self.normalized_importances, ammo_damage_prob):
            # 保留health参数位置，如需后续启用动态健康度逻辑，可直接取消注释下方代码
            # total_effect += norm_importance * damage_prob * health
            total_effect += norm_importance * damage_prob
        return total_effect

class Ammunition:
    def __init__(self, ammo_id: int, cost: float, stock: int, damage_profiles: Dict[int, List[float]]):
        self.id = ammo_id
        self.cost = cost
        self.stock = stock
        self.damage_profiles = damage_profiles  # 目标ID到毁伤概率列表的映射

"""静态弹目匹配模型 - 包含自适应交叉变异"""
class AHLNSGAII_Solver:
    """静态弹目匹配模型 - 包含自适应交叉变异"""

    def __init__(self, targets: List[Target], ammunitions: List[Ammunition],
                 adaptability_matrix: np.ndarray, damage_threshold: float):
        self.targets = targets
        self.ammunitions = ammunitions
        self.adaptability_matrix = adaptability_matrix
        self.damage_threshold = damage_threshold
        self.population_size = 30
        self.max_generations = 500

        # 自适应参数
        self.crossover_rate_min = 0.7
        self.crossover_rate_max = 0.9
        self.mutation_rate_min = 0.01
        self.mutation_rate_max = 0.03
        self.local_search_rate_min = 0.2
        self.local_search_rate_max = 0.4

    def calculate_damage_efficiency(self, ammo_id: int, target_id: int) -> float:
        """计算单发弹药的毁伤效能 e_ij"""
        target = self.targets[target_id]
        ammo = self.ammunitions[ammo_id]

        damage_prob = ammo.damage_profiles.get(target.id, [0.1] * len(target.components))
        adaptability = self.adaptability_matrix[ammo_id, target_id]
        weighted_damage = target.calculate_damage_effect(damage_prob)

        return adaptability * weighted_damage

    def calculate_required_rounds(self, ammo_id: int, target_id: int) -> int:
        """计算达到毁伤要求所需弹药数量 r_ij"""
        e_ij = self.calculate_damage_efficiency(ammo_id, target_id)
        if e_ij <= 0:
            return float('inf')

        # 修正：使用文档公式（33）计算最少发射数量
        required = np.ceil(np.log(1 - self.damage_threshold) / np.log(1 - e_ij))
        return max(1, int(required))

    def initialize_population(self) -> List[np.ndarray]:
        """改进的初始化 - 确保满足毁伤要求和库存约束"""
        population = []
        target_count = len(self.targets)
        ammo_count = len(self.ammunitions)

        # 检查边界条件
        if target_count == 0 or ammo_count == 0:
            return [np.zeros((ammo_count, target_count), dtype=int)]

        for _ in range(self.population_size):
            chromosome = np.zeros((ammo_count, target_count), dtype=int)

            # 为每个目标确保达到毁伤要求
            for j in range(target_count):
                # 选择对该目标最有效的弹药
                ammo_effects = []
                for i in range(ammo_count):
                    try:
                        e_ij = self.calculate_damage_efficiency(i, j)
                        ammo_effects.append((i, e_ij))
                    except Exception as e:
                        print(f"计算毁伤效能异常：{e}")
                        ammo_effects.append((i, 0.1))  # 默认值

                # 按毁伤效能排序
                ammo_effects.sort(key=lambda x: x[1], reverse=True)

                # 分配弹药直到满足毁伤要求
                current_damage = 0
                attempts = 0
                while current_damage < self.damage_threshold and attempts < ammo_count:
                    ammo_id, e_ij = ammo_effects[attempts]
                    if e_ij > 0:
                        try:
                            # 计算需要多少发
                            required = self.calculate_required_rounds(ammo_id, j)
                            # 严格的库存检查
                            remaining_stock = self.ammunitions[ammo_id].stock - np.sum(chromosome[ammo_id, :])
                            allocated = min(required, remaining_stock)
                            if allocated > 0:
                                chromosome[ammo_id, j] = allocated
                                # 更新当前毁伤概率
                                current_damage = self.calculate_actual_damage(chromosome, j)
                        except Exception as e:
                            print(f"弹药分配异常：{e}")
                    attempts += 1

            population.append(chromosome)

        return population

    def calculate_actual_damage(self, chromosome: np.ndarray, target_id: int) -> float:
        """计算对特定目标的实际毁伤概率"""
        survival_prob = 1.0

        for i in range(len(self.ammunitions)):
            if chromosome[i, target_id] > 0:
                e_ij = self.calculate_damage_efficiency(i, target_id)
                survival_prob *= (1 - e_ij) ** chromosome[i, target_id]

        return 1 - survival_prob

    def evaluate_objectives(self, chromosome: np.ndarray) -> Tuple[float, float]:
        """评估目标函数（修正费效比定义）"""
        total_effect = 0.0
        total_cost = 0.0

        # 计算总毁伤效能和总成本
        for j in range(len(self.targets)):
            target_damage = self.calculate_actual_damage(chromosome, j)
            # 修正：仅计算达到阈值部分的有效毁伤
            effective_damage = min(target_damage, self.damage_threshold)
            total_effect += self.targets[j].value * effective_damage

            for i in range(len(self.ammunitions)):
                rounds = chromosome[i, j]
                total_cost += self.ammunitions[i].cost * rounds

        # 总时间计算保持不变
        total_time = np.sum(chromosome) * 0.5

        # 修正：费效比定义为有效毁伤与成本的比值
        cost_effectiveness = total_effect / total_cost if total_cost > 0 else 0.001

        return cost_effectiveness, total_time

    def evaluate_objectives_with_penalty(self, chromosome: np.ndarray) -> Tuple[float, float, float]:
        """带惩罚项的目标函数评估（新增过量毁伤惩罚）"""
        total_effect = 0.0
        total_cost = 0.0
        penalty = 0.0

        for j in range(len(self.targets)):
            target_damage = self.calculate_actual_damage(chromosome, j)
            target_value = self.targets[j].value

            # 对未达阈值的惩罚
            if target_damage < self.damage_threshold:
                penalty += (self.damage_threshold - target_damage) * target_value * 10
            # 新增：对过量毁伤的惩罚（超过阈值越多，惩罚越重）
            elif target_damage > self.damage_threshold + 0.1:  # 允许10%的超额容差
                excess = target_damage - (self.damage_threshold + 0.1)
                penalty += excess * target_value * 5  # 过量惩罚（权重可调整）

            # 计算有效毁伤效果（超过阈值的部分不计入收益）
            effective_damage = min(target_damage, self.damage_threshold + 0.1)
            total_effect += target_value * effective_damage

            # 计算总成本
            for i in range(len(self.ammunitions)):
                rounds = chromosome[i, j]
                total_cost += self.ammunitions[i].cost * rounds

        # 总时间计算保持不变
        total_time = np.sum(chromosome) * 0.5

        # 带惩罚的费效比
        # effective_effect = total_effect - penalty
        # cost_effectiveness = effective_effect / total_cost if total_cost > 0 else 0.001

        # return cost_effectiveness, total_time, penalty
        # 确保总成本不为0，避免除0错误
        total_cost = max(total_cost, 1e-6)  # 用极小值代替0
        effective_effect = total_effect - penalty
        cost_effectiveness = effective_effect / total_cost

        return cost_effectiveness, total_time, penalty

    def dominates(self, obj1: Tuple[float, float], obj2: Tuple[float, float]) -> bool:
        """判断解1是否支配解2"""
        # 目标1: 最大化费效比，目标2: 最小化时间
        return (obj1[0] >= obj2[0] and obj1[1] <= obj2[1]) and \
               (obj1[0] > obj2[0] or obj1[1] < obj2[1])

    def fast_non_dominated_sort(self, population: List[np.ndarray]) -> List[List[int]]:
        """非支配排序"""
        if not population:
            return []

        pop_size = len(population)
        S = [[] for _ in range(pop_size)]
        n = [0] * pop_size
        rank = [0] * pop_size
        fronts = [[]]

        # 修正：使用带惩罚项的目标函数评估
        # objectives = [self.evaluate_objectives_with_penalty(ind)[:2] for ind in population]
        objectives = [self.evaluate_objectives_with_penalty(ind) for ind in population]

        for i in range(pop_size):
            for j in range(pop_size):
                if i == j:
                    continue
                if self.dominates(objectives[i], objectives[j]):
                    S[i].append(j)
                elif self.dominates(objectives[j], objectives[i]):
                    n[i] += 1

            if n[i] == 0:
                rank[i] = 0
                fronts[0].append(i)

        i = 0
        while fronts[i]:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            if Q:
                fronts.append(Q)
            else:
                break

        # return fronts
        # 过滤空前沿
        valid_fronts = [front for front in fronts if len(front) > 0]
        return valid_fronts if valid_fronts else []  # 确保不返回空列表或包含空列表的前沿

    # def crowding_distance(self, front_indices: List[int], population: List[np.ndarray]) -> List[float]:
    #     """计算拥挤度距离"""
    #     size = len(front_indices)
    #     if size == 0:
    #         return []

    #     distances = [0.0] * size
    #     # 修正：使用带惩罚项的目标函数评估
    #     objectives = [self.evaluate_objectives_with_penalty(population[i])[:2] for i in front_indices]

    #     # 对每个目标函数
    #     for obj_idx in range(2):
    #         # 创建索引和对应目标值的列表
    #         indexed_objs = list(zip(range(size), [obj[obj_idx] for obj in objectives]))

    #         # 按目标值排序
    #         if obj_idx == 0:  # 费效比，从大到小
    #             indexed_objs.sort(key=lambda x: x[1], reverse=True)
    #         else:  # 时间，从小到大
    #             indexed_objs.sort(key=lambda x: x[1])

    #         # 边界个体距离设为无穷大
    #         if size > 0:
    #             distances[indexed_objs[0][0]] = float('inf')
    #         if size > 1:
    #             distances[indexed_objs[-1][0]] = float('inf')

    #         # 计算中间个体的距离
    #         if size > 2:
    #             min_val = indexed_objs[0][1]
    #             max_val = indexed_objs[-1][1]

    #             if abs(max_val - min_val) > 1e-10:
    #                 for i in range(1, size - 1):
    #                     prev_val = indexed_objs[i-1][1]
    #                     next_val = indexed_objs[i+1][1]
    #                     original_idx = indexed_objs[i][0]
    #                     distances[original_idx] += (next_val - prev_val) / (max_val - min_val)

    #     return distances
    
    def crowding_distance(self, front_indices: List[int], population: List[np.ndarray]) -> List[float]:
        size = len(front_indices)
        if size == 0:
            return []

        distances = [0.0] * size
        objectives = [self.evaluate_objectives_with_penalty(population[i])[:2] for i in front_indices]

        for obj_idx in range(2):
            indexed_objs = list(zip(range(size), [obj[obj_idx] for obj in objectives]))
            if obj_idx == 0:
                indexed_objs.sort(key=lambda x: x[1], reverse=True)  # 费效比：降序
            else:
                indexed_objs.sort(key=lambda x: x[1])  # 时间：升序

            # 边界个体标记为特殊值（用于排序但不影响均值）
            if size > 0:
                distances[indexed_objs[0][0]] = float('inf')  # 保留无穷大用于排序
            if size > 1:
                distances[indexed_objs[-1][0]] = float('inf')

            if size > 2:
                min_val = indexed_objs[0][1]
                max_val = indexed_objs[-1][1]
            # 处理目标值相同的情况（避免除零）
                if abs(max_val - min_val) < 1e-10:
                # 所有目标值相同，中间个体距离设为0
                    for i in range(1, size - 1):
                        original_idx = indexed_objs[i][0]
                        distances[original_idx] += 0.0
                    continue
            
            # 计算中间个体距离
                for i in range(1, size - 1):
                    prev_val = indexed_objs[i-1][1]
                    next_val = indexed_objs[i+1][1]
                    original_idx = indexed_objs[i][0]
                # 确保分子非负（避免数值波动导致的负值）
                    distance_contribution = max(0.0, (next_val - prev_val) / (max_val - min_val))
                    distances[original_idx] += distance_contribution

    # 过滤无穷大值后再计算均值（在调用处使用）
        return distances

    def meets_damage_requirements(self, chromosome: np.ndarray) -> bool:
        """检查是否满足所有目标的毁伤要求和库存约束"""
        # 1. 检查毁伤要求
        for j in range(len(self.targets)):
            damage = self.calculate_actual_damage(chromosome, j)
            if damage < self.damage_threshold - 0.01:  # 容差
                return False
        
        # 2. 新增：检查库存约束
        for i in range(len(self.ammunitions)):
            total_used = np.sum(chromosome[i, :])
            if total_used > self.ammunitions[i].stock:
                return False
                
        return True

    def repair_solution(self, chromosome: np.ndarray) -> np.ndarray:
        """修复解，使其满足毁伤要求和库存约束"""
        repaired = chromosome.copy()

        # 先修复库存约束（确保不超过库存）
        for i in range(len(self.ammunitions)):
            total_used = np.sum(repaired[i, :])
            if total_used > self.ammunitions[i].stock:
                excess = total_used - self.ammunitions[i].stock
                # 从分配量最大的目标开始减少
                target_indices = np.argsort(repaired[i, :])[::-1]  # 降序排列
                for j in target_indices:
                    if excess <= 0:
                        break
                    reduce_amount = min(excess, repaired[i, j])
                    repaired[i, j] -= reduce_amount
                    excess -= reduce_amount

        # 再修复毁伤要求
        for j in range(len(self.targets)):
            current_damage = self.calculate_actual_damage(repaired, j)

            # 如果未达到要求，增加弹药
            while current_damage < self.damage_threshold:
                # 找到对该目标最有效的弹药
                best_ammo = -1
                best_effect = 0

                for i in range(len(self.ammunitions)):
                    e_ij = self.calculate_damage_efficiency(i, j)
                    # 检查库存
                    total_used = np.sum(repaired[i, :])
                    if e_ij > best_effect and total_used < self.ammunitions[i].stock:
                        best_effect = e_ij
                        best_ammo = i

                if best_ammo >= 0:
                    repaired[best_ammo, j] += 1
                    current_damage = self.calculate_actual_damage(repaired, j)
                else:
                    break  # 无法再增加弹药

        return repaired

    def solve(self) -> Tuple[List[Tuple[np.ndarray, Tuple[float, float], int, float]], int]:
        """改进的求解函数 - 增强错误处理"""
        try:
            population = self.initialize_population()
            all_feasible_with_metadata = []

            # 收敛监控参数
            convergence_threshold = 5e-4
            convergence_window = 10
            best_ce_history = []
            convergence_generation = None
            stable_count = 0

            for generation in range(self.max_generations):
                try:
                    current_time = time.time()
                    crossover_rate = self.adaptive_crossover_rate(generation)
                    mutation_rate = self.adaptive_mutation_rate(generation)
                    local_search_rate = self.adaptive_local_search_rate(generation)

                    # 生成子代
                    offspring = []
                    while len(offspring) < self.population_size:
                        try:
                            parent1 = self.tournament_selection(population)
                            parent2 = self.tournament_selection(population)
                            child1, child2 = self.adaptive_crossover(parent1, parent2, crossover_rate)
                            child1 = self.adaptive_mutation(child1, mutation_rate)
                            child2 = self.adaptive_mutation(child2, mutation_rate)
                            if random.random() < local_search_rate:
                                child1 = self.heuristic_local_search(child1)
                                child2 = self.heuristic_local_search(child2)
                            child1 = self.repair_solution(child1)
                            child2 = self.repair_solution(child2)
                            offspring.extend([child1, child2])
                        except Exception as e:
                            print(f"子代生成异常：{e}，跳过当前子代")
                            continue

                    # 合并种群+非支配排序+拥挤度选择
                    combined = population + offspring
                    fronts = self.fast_non_dominated_sort(combined)
                    new_population = []
                    for front in fronts:
                        if len(new_population) + len(front) <= self.population_size:
                            for idx in front:
                                new_population.append(combined[idx])
                        else:
                            crowding_distances = self.crowding_distance(front, combined)
                            sorted_indices = [front[i] for i in np.argsort(crowding_distances)[::-1]]
                            needed = self.population_size - len(new_population)
                            new_population.extend([combined[idx] for idx in sorted_indices[:needed]])
                            break

                    # 收敛检测
                    try:
                        current_best_ce = max([self.evaluate_objectives_with_penalty(ind)[0] for ind in new_population])
                        best_ce_history.append(current_best_ce)

                        if len(best_ce_history) >= convergence_window:
                            recent_best = best_ce_history[-convergence_window:]
                            max_fluctuation = max(recent_best) - min(recent_best)

                            if max_fluctuation < convergence_threshold:
                                stable_count += 1
                                if stable_count >= convergence_window:
                                    convergence_generation = generation
                                    break
                            else:
                                stable_count = 0
                    except Exception as e:
                        print(f"收敛检测异常：{e}")

                    # 筛选可行解
                    feasible_in_new = []
                    for sol in new_population:
                        if self.meets_damage_requirements(sol):
                            # objectives = self.evaluate_objectives_with_penalty(sol)[:2]
                            objectives = self.evaluate_objectives_with_penalty(sol)
                            all_feasible_with_metadata.append((sol, objectives, generation, current_time))

                    population = new_population

                except Exception as e:
                    print(f"第{generation}代迭代异常：{e}")
                    continue

            # 若迭代结束仍未收敛，收敛代数设为最大迭代次数
            if convergence_generation is None:
                convergence_generation = self.max_generations

            # 去重处理
            unique_solutions_with_metadata = self.remove_duplicate_solutions_with_metadata(all_feasible_with_metadata)

            return unique_solutions_with_metadata, convergence_generation

        except Exception as e:
            print(f"求解过程异常：{e}")
            return [], self.max_generations

    def solutions_are_similar(self, sol1: np.ndarray, sol2: np.ndarray, tolerance: float = 0.01) -> bool:
        """判断两个解是否相似（目标函数值相近）"""
        obj1 = self.evaluate_objectives_with_penalty(sol1)[:2]
        obj2 = self.evaluate_objectives_with_penalty(sol2)[:2]

        # 检查费效比和总时间是否相近
        ce_similar = abs(obj1[0] - obj2[0]) / max(abs(obj1[0]), abs(obj2[0]), 1e-10) < tolerance
        time_similar = abs(obj1[1] - obj2[1]) / max(abs(obj1[1]), abs(obj2[1]), 1e-10) < tolerance

        return ce_similar and time_similar

    def remove_duplicate_solutions(self, solutions: List[Tuple[np.ndarray, Tuple[float, float]]]) -> List[Tuple[np.ndarray, Tuple[float, float]]]:
        """移除重复和相似的解"""
        unique_solutions = []
        seen_hashes = set()

        for solution, objectives in solutions:
            # 创建解的哈希值（基于分配矩阵）
            solution_hash = hash(solution.tobytes())

            # 如果哈希值没出现过，且与已有解不相似，则添加
            if solution_hash not in seen_hashes:
                is_duplicate = False
                for existing_sol, _ in unique_solutions:
                    if self.solutions_are_similar(solution, existing_sol):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_solutions.append((solution, objectives))
                    seen_hashes.add(solution_hash)

        return unique_solutions

    def remove_duplicate_solutions_with_metadata(self, solutions: List[Tuple[np.ndarray, Tuple[float, float], int, float]]) -> List[Tuple[np.ndarray, Tuple[float, float], int, float]]:
        """优化去重逻辑：保留更多优质且不相似的解"""
        unique_solutions = []
        seen_patterns = {}
        tolerance = 0.02  # 放宽相似判定阈值

        for solution, objectives, generation, timestamp in solutions:
            # 1. 先按分配模式去重（完全相同的解直接去重）
            pattern_parts = []
            for j in range(len(self.targets)):
                target_allocation = []
                for i in range(len(self.ammunitions)):
                    if solution[i, j] > 0:
                        target_allocation.append(f"{i}_{solution[i, j]}")
                pattern_parts.append(",".join(sorted(target_allocation)))
            pattern = "|".join(pattern_parts)

            if pattern in seen_patterns:
                continue

            # 2. 再按目标函数值去重（相似解只保留最早生成的）
            is_similar = False
            for existing_sol, existing_obj, _, _ in unique_solutions:
                ce_diff = abs(objectives[0] - existing_obj[0]) / max(abs(objectives[0]), abs(existing_obj[0]), 1e-10)
                time_diff = abs(objectives[1] - existing_obj[1]) / max(abs(objectives[1]), abs(existing_obj[1]), 1e-10)
                if ce_diff < tolerance and time_diff < tolerance:
                    is_similar = True
                    break

            if not is_similar:
                unique_solutions.append((solution, objectives, generation, timestamp))
                seen_patterns[pattern] = True

        return unique_solutions


    def adaptive_crossover_rate(self, generation: int) -> float:
        """自适应交叉概率 - 随代数增加逐渐降低"""
        progress = generation / self.max_generations
        return self.crossover_rate_max - (self.crossover_rate_max - self.crossover_rate_min) * progress

    def adaptive_mutation_rate(self, generation: int) -> float:
        """自适应变异概率 - 随代数增加逐渐降低"""
        progress = generation / self.max_generations
        return self.mutation_rate_max - (self.mutation_rate_max - self.mutation_rate_min) * progress

    def adaptive_local_search_rate(self, generation: int) -> float:
        """自适应局部搜索概率 - 后期增加局部搜索"""
        progress = generation / self.max_generations
        return self.local_search_rate_min + (self.local_search_rate_max - self.local_search_rate_min) * progress

    def heuristic_local_search(self, chromosome: np.ndarray) -> np.ndarray:
        """启发式局部搜索算子"""
        improved_chromosome = chromosome.copy()
        target_count = len(self.targets)
        ammo_count = len(self.ammunitions)

        for j in range(target_count):
            current_damage = self.calculate_actual_damage(improved_chromosome, j)

            if current_damage < self.damage_threshold:
                # 毁伤不足：增加最有效弹药的数量
                best_ammo = -1
                best_effect = 0
                for i in range(ammo_count):
                    e_ij = self.calculate_damage_efficiency(i, j)
                    # 检查库存
                    total_used = np.sum(improved_chromosome[i, :])
                    if e_ij > best_effect and total_used < self.ammunitions[i].stock:
                        best_effect = e_ij
                        best_ammo = i

                if best_ammo >= 0:
                    improved_chromosome[best_ammo, j] += 1

            elif current_damage > self.damage_threshold + 0.1:  # 严重过量毁伤
                # 减少效费比最低的弹药数量，但至少保留1发
                non_zero_ammos = []
                for i in range(ammo_count):
                    if improved_chromosome[i, j] > 1:  # 至少2发才能减少
                        cost_effectiveness = self.calculate_damage_efficiency(i, j) / self.ammunitions[i].cost
                        non_zero_ammos.append((i, cost_effectiveness))

                if non_zero_ammos:
                    # 找到效费比最低的弹药
                    worst_ammo = min(non_zero_ammos, key=lambda x: x[1])[0]
                    improved_chromosome[worst_ammo, j] -= 1

        return improved_chromosome

    def tournament_selection(self, population: List[np.ndarray]) -> np.ndarray:
        """锦标赛选择"""
        tournament_size = 3
        tournament = random.sample(population, tournament_size)
        # 修正：使用带惩罚项的目标函数评估
        return max(tournament, key=lambda x: self.evaluate_objectives_with_penalty(x)[0])

    def adaptive_crossover(self, parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float) -> Tuple[
        np.ndarray, np.ndarray]:
        """自适应交叉操作 - 修复版本"""

        # 边界条件处理：目标数<2或交叉概率不满足时直接返回父代
        if len(self.targets) < 2 or random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()

        # 初始化子代
        child1 = parent1.copy()
        child2 = parent2.copy()

        try:
            # 多种交叉策略
            crossover_type = random.choice(['single_point', 'uniform', 'target_based'])

            if crossover_type == 'single_point':
                # 确保有足够的交叉点
                if len(self.targets) >= 2:
                    crossover_point = random.randint(1, len(self.targets) - 1)
                    for j in range(crossover_point, len(self.targets)):
                        child1[:, j] = parent2[:, j]
                        child2[:, j] = parent1[:, j]

            elif crossover_type == 'uniform':
                # 均匀交叉 - 对每个目标独立决定
                for j in range(len(self.targets)):
                    if random.random() < 0.5:
                        child1[:, j] = parent2[:, j]
                        child2[:, j] = parent1[:, j]

            elif crossover_type == 'target_based':
                # 基于目标的交叉
                for j in range(len(self.targets)):
                    if random.random() < 0.5:
                        child1[:, j] = parent1[:, j]
                        child2[:, j] = parent2[:, j]
                    else:
                        child1[:, j] = parent2[:, j]
                        child2[:, j] = parent1[:, j]

        except Exception as e:
            # 如果交叉过程中出现错误，返回父代复制
            print(f"交叉操作异常：{e}，返回父代")
            return parent1.copy(), parent2.copy()

        return child1, child2

    def adaptive_mutation(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """自适应变异操作 - 增强错误处理"""
        try:
            mutated = chromosome.copy()
            target_count = len(self.targets)
            ammo_count = len(self.ammunitions)

            # 检查边界条件
            if target_count == 0 or ammo_count == 0:
                return mutated

            # 对每个弹药-目标对进行变异
            for i in range(ammo_count):
                for j in range(target_count):
                    if random.random() < mutation_rate:
                        # 多种变异策略
                        mutation_type = random.choice(['perturb', 'swap', 'redistribute'])

                        if mutation_type == 'perturb':
                            # 扰动变异
                            change = random.choice([-2, -1, 1, 2])
                            new_value = max(0, mutated[i, j] + change)

                            # 检查库存约束
                            total_used = np.sum(mutated[i, :]) - mutated[i, j] + new_value
                            if total_used <= self.ammunitions[i].stock:
                                mutated[i, j] = new_value

                        elif mutation_type == 'swap':
                            # 交换变异 - 在两个目标间交换弹药分配
                            if mutated[i, j] > 0 and target_count > 1:
                                other_j = random.randint(0, target_count - 1)
                                if other_j != j:
                                    # 交换分配量
                                    temp = mutated[i, j]
                                    mutated[i, j] = mutated[i, other_j]
                                    mutated[i, other_j] = temp

                        elif mutation_type == 'redistribute':
                            # 重新分配变异 - 确保有足够弹药可分配
                            if mutated[i, j] >= 2 and target_count > 1:  # 至少需要2发才能分配
                                # 将部分弹药重新分配到其他目标
                                max_transfer = mutated[i, j] // 2
                                if max_transfer >= 1:
                                    transfer_amount = random.randint(1, max_transfer)
                                    other_j = random.randint(0, target_count - 1)

                                    mutated[i, j] -= transfer_amount
                                    mutated[i, other_j] += transfer_amount

            return mutated

        except Exception as e:
            print(f"变异操作异常：{e}，返回原染色体")
            return chromosome.copy()

# 示例使用
def main():
    # 测试数据
    # targets = [
    #     Target(1, 100.0, [(0.3, 1.0), (0.4, 1.0), (0.3, 1.0)]),
    #     Target(2, 80.0, [(0.5, 1.0), (0.5, 1.0)]),
    #     Target(3, 120.0, [(0.2, 1.0), (0.3, 1.0), (0.3, 1.0), (0.2, 1.0)]),
    #     Target(4, 100.0, [(0.3, 1.0), (0.4, 1.0), (0.3, 1.0)]),
    #     Target(5, 80.0, [(0.5, 1.0), (0.5, 1.0)]),
    #     Target(6, 120.0, [(0.2, 1.0), (0.3, 1.0), (0.3, 1.0), (0.2, 1.0)])
    # ]

    # ammunitions = [
    #     Ammunition(1, 10.0, 30, {
    #         1: [0.6, 0.4, 0.5], 2: [0.7, 0.5], 3: [0.5, 0.6, 0.4, 0.3],
    #         4: [0.6, 0.4, 0.5], 5: [0.7, 0.5], 6: [0.5, 0.6, 0.4, 0.3]
    #     }),
    #     Ammunition(2, 15.0, 30, {
    #         1: [0.8, 0.6, 0.7], 2: [0.6, 0.7], 3: [0.7, 0.5, 0.6, 0.5],
    #         4: [0.6, 0.4, 0.5], 5: [0.7, 0.5], 6: [0.5, 0.6, 0.4, 0.3]
    #     }),
    #     Ammunition(3, 25.0, 30, {
    #         1: [0.9, 0.8, 0.85], 2: [0.85, 0.9], 3: [0.8, 0.7, 0.75, 0.8],
    #         4: [0.6, 0.4, 0.5], 5: [0.7, 0.5], 6: [0.5, 0.6, 0.4, 0.3]
    #     }),
    #     Ammunition(4, 20.0, 30, {
    #         1: [0.8, 0.5, 0.7], 2: [0.6, 0.7], 3: [0.7, 0.5, 0.6, 0.5],
    #         4: [0.6, 0.4, 0.5], 5: [0.7, 0.5], 6: [0.5, 0.6, 0.7, 0.3]
    #     }),
    #     Ammunition(5, 5.0, 30, {
    #         1: [0.9, 0.8, 0.85], 2: [0.85, 0.9], 3: [0.8, 0.7, 0.55, 0.8],
    #         4: [0.6, 0.2, 0.4], 5: [0.8, 0.5], 6: [0.5, 0.6, 0.4, 0.3]
    #     })
    # ]

    # adaptability_matrix = np.array([
    #     [0.8, 0.7, 0.6, 0.8, 0.7, 0.6],
    #     [0.9, 0.8, 0.7, 0.9, 0.8, 0.7],
    #     [0.7, 0.9, 0.8, 0.7, 0.9, 0.8],
    #     [0.8, 0.6, 0.5, 0.4, 0.7, 0.6],
    #     [0.9, 0.9, 0.8, 0.5, 0.6, 0.6]
    # ])

    # 真实数据
    # 测试参数(5个目标)
    # targets = [
    #     Target(1, 80.0, [(0.5, 1.0)]),# 人员集群
    #     Target(2, 120.0, [(0.2, 1.0), (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), 
    #                       (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.8, 1.0)]),# 地下指挥所
    #     Target(3, 100.0, [(0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), 
    #                       (0.8, 1.0)]),# 陆基雷达站
    #     Target(4, 110.0, [(0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.2, 1.0),
    #                       (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.5, 1.0),
    #                       (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.2, 1.0), (0.8, 1.0), 
    #                       (0.5, 1.0)]),# 机场
    #     Target(5, 105.0, [(0.2, 1.0), (0.2, 1.0), (0.2, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0),
    #                       (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), 
    #                       (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), 
    #                       (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),                           
    #                       (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), 
    #                       (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.8, 1.0), (0.8, 1.0) ]),# 阵地(雷达车15个，电源车13个，导弹发射车18个，指挥控制车17个)
    # ]

    # ammunitions = [
    #     Ammunition(1, 13.0, 10, {
    #         1: [0.8], 
    #         2: [0.8, 0.8, 0.8, 0.6, 0.6,
    #             0.5, 0.8, 0.7, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.8, 0.1], 
    #         3: [0.5, 0.2, 0.6, 0.7, 0.8,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2],
    #         4: [0.2, 0.2, 0.8, 0.7, 0.7,
    #             0.7, 0.7, 0.2, 0.2, 0.3,
    #             0.2, 0.2, 0.2, 0.2, 0.2, 
    #             0.6], 
    #         5: [0.7, 0.7, 0.7, 0.8, 0.8,
    #             0.8, 0.8, 0.6, 0.7, 0.7,
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.8, 0.7, 0.7, 0.6, 0.6,
    #             0.7, 0.5, 0.7, 0.7, 0.7,
    #             0.5, 0.6, 0.5, 
    #             0.5, 0.7, 0.5, 0.5, 0.5, 
    #             0.5, 0.3, 0.7, 0.5, 0.4,
    #             0.4, 0.4, 0.5, 0.5, 0.5,
    #             0.5, 0.5, 0.5,                           
    #             0.4, 0.5, 0.4, 0.5, 0.5, 
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.5, 0.5], 
    #     }),# 杀爆1(当量大)
    #     Ammunition(2, 10.0, 10, {
    #         1: [0.7], 
    #         2: [0.7, 0.7, 0.7, 0.5, 0.5,
    #             0.4, 0.7, 0.6, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.7, 0.1], 
    #         3: [0.4, 0.2, 0.5, 0.6, 0.7,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2],
    #         4: [0.2, 0.2, 0.7, 0.6, 0.6,
    #             0.6, 0.6, 0.2, 0.2, 0.2,
    #             0.2, 0.2, 0.2, 0.2, 0.2, 
    #             0.5], 
    #         5: [0.6, 0.6, 0.6, 0.7, 0.7,
    #             0.7, 0.7, 0.5, 0.6, 0.6,
    #             0.6, 0.6, 0.6, 0.6, 0.6,
    #             0.7, 0.6, 0.6, 0.5, 0.5,
    #             0.6, 0.4, 0.6, 0.6, 0.6,
    #             0.4, 0.5, 0.4, 
    #             0.4, 0.6, 0.4, 0.4, 0.4, 
    #             0.4, 0.2, 0.6, 0.4, 0.3,
    #             0.3, 0.3, 0.4, 0.4, 0.4,
    #             0.4, 0.4, 0.4,                           
    #             0.3, 0.4, 0.3, 0.4, 0.4, 
    #             0.3, 0.4, 0.3, 0.4, 0.4,
    #             0.3, 0.4, 0.3, 0.4, 0.4,
    #             0.4, 0.4], 
    #     }),# 杀爆2(当量小)
    #     Ammunition(3, 20.0, 10, {
    #         1: [0.1], 
    #         2: [0.6, 0.6, 0.6, 0.7, 0.7,
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.7, 0.7, 0.7, 0.7, 0.7, 
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.5, 0.7], 
    #         3: [0.7, 0.7, 0.3, 0.3, 0.5,
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.7],
    #         4: [0.7, 0.7, 0.4, 0.4, 0.4,
    #             0.4, 0.4, 0.7, 0.7, 0.7,
    #             0.7, 0.7, 0.7, 0.7, 0.6, 
    #             0.7], 
    #         5: [0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1,                            
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1], 
    #     }),# 侵爆1(1.8m)
    #     Ammunition(4, 25.0, 10, {
    #         1: [0.1], 
    #         2: [0.65, 0.65, 0.65, 0.75, 0.75,
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.75, 0.75, 0.75, 0.75, 0.75, 
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.55, 0.75], 
    #         3: [0.75, 0.75, 0.35, 0.35, 0.55,
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.75],
    #         4: [0.75, 0.75, 0.45, 0.45, 0.45,
    #             0.45, 0.45, 0.75, 0.75, 0.75,
    #             0.75, 0.75, 0.75, 0.75, 0.65, 
    #             0.75], 
    #         5: [0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1,                            
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1], 
    #     }),# 侵爆2(6m)
    #     Ammunition(5, 40.0, 10, {
    #         1: [0.2], 
    #         2: [0.7, 0.7, 0.7, 0.8, 0.8,
    #             0.8, 0.8, 0.8, 0.8, 0.8,
    #             0.8, 0.8, 0.8, 0.8, 0.8, 
    #             0.8, 0.8, 0.8, 0.8, 0.8,
    #             0.6, 0.8], 
    #         3: [0.8, 0.8, 0.4, 0.4, 0.6,
    #             0.8, 0.8, 0.8, 0.8, 0.8,
    #             0.8, 0.8, 0.8, 0.8, 0.8,
    #             0.8],
    #         4: [0.8, 0.8, 0.4, 0.4, 0.4,
    #             0.4, 0.4, 0.8, 0.8, 0.8,
    #             0.8, 0.8, 0.8, 0.8, 0.7, 
    #             0.8], 
    #         5: [0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15,                            
    #             0.15, 0.15, 0.15, 0.15, 0.15, 
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15], 
    #     }),# 侵爆3(61m)
    #     Ammunition(6, 27.0, 10, {
    #         1: [0.9], 
    #         2: [0.85, 0.85, 0.85, 0.65, 0.65,
    #             0.55, 0.85, 0.75, 0.2, 0.2,
    #             0.2, 0.2, 0.2, 0.2, 0.2, 
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.85, 0.2], 
    #         3: [0.55, 0.25, 0.65, 0.75, 0.85,
    #             0.25, 0.25, 0.25, 0.25, 0.25,
    #             0.25, 0.25, 0.25, 0.25, 0.25,
    #             0.25],
    #         4: [0.3, 0.3, 0.8, 0.75, 0.7,
    #             0.7, 0.75, 0.3, 0.3, 0.35,
    #             0.3, 0.3, 0.3, 0.3, 0.3, 
    #             0.6], 
    #         5: [0.75, 0.75, 0.75, 0.85, 0.85,
    #             0.85, 0.85, 0.65, 0.75, 0.75,
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.85, 0.75, 0.75, 0.65, 0.65,
    #             0.75, 0.55, 0.75, 0.75, 0.75,
    #             0.55, 0.65, 0.55, 
    #             0.55, 0.75, 0.55, 0.55, 0.55, 
    #             0.55, 0.35, 0.75, 0.55, 0.45,
    #             0.45, 0.45, 0.55, 0.55, 0.55,
    #             0.55, 0.55, 0.55,                           
    #             0.45, 0.55, 0.45, 0.55, 0.55, 
    #             0.45, 0.55, 0.45, 0.55, 0.55,
    #             0.45, 0.55, 0.45, 0.55, 0.55,
    #             0.55, 0.55], 
    #     }),# 子母1
    #     Ammunition(7, 20.0, 10, {
    #         1: [0.1], 
    #         2: [0.6, 0.6, 0.6, 0.6, 0.6,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1], 
    #         3: [0.1, 0.1, 0.1, 0.3, 0.3,
    #             0.4, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1],
    #         4: [0.1, 0.1, 0.4, 0.2, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1], 
    #         5: [0.6, 0.6, 0.6, 0.7, 0.7,
    #             0.7, 0.7, 0.5, 0.6, 0.6,
    #             0.6, 0.6, 0.6, 0.6, 0.6,
    #             0.7, 0.6, 0.6, 0.5, 0.5,
    #             0.6, 0.4, 0.6, 0.6, 0.6,
    #             0.4, 0.5, 0.4, 
    #             0.4, 0.6, 0.4, 0.4, 0.4, 
    #             0.4, 0.2, 0.6, 0.4, 0.3,
    #             0.3, 0.3, 0.4, 0.4, 0.4,
    #             0.4, 0.4, 0.4,                           
    #             0.4, 0.5, 0.4, 0.5, 0.5, 
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.5, 0.5], 
    #     }),# 聚能1(1.3m破甲)
    #     Ammunition(8, 18.0, 10, {
    #         1: [0.1], 
    #         2: [0.5, 0.5, 0.5, 0.5, 0.5,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1], 
    #         3: [0.1, 0.1, 0.1, 0.2, 0.2,
    #             0.3, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1],
    #         4: [0.1, 0.1, 0.3, 0.15, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1], 
    #         5: [0.6, 0.6, 0.6, 0.7, 0.7,
    #             0.7, 0.7, 0.5, 0.6, 0.6,
    #             0.6, 0.6, 0.6, 0.6, 0.6,
    #             0.7, 0.6, 0.6, 0.5, 0.5,
    #             0.6, 0.4, 0.6, 0.6, 0.6,
    #             0.4, 0.5, 0.4, 
    #             0.4, 0.6, 0.4, 0.4, 0.4, 
    #             0.4, 0.2, 0.6, 0.4, 0.3,
    #             0.3, 0.3, 0.4, 0.4, 0.4,
    #             0.4, 0.4, 0.4,                           
    #             0.4, 0.5, 0.4, 0.5, 0.5, 
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.5, 0.5], 
    #     })# 聚能2(1.1m破甲)
    # ]
    # adaptability_matrix = np.array([
    #     [0.9, 0.25, 0.45, 0.25, 0.8],   # 杀爆1
    #     [0.8, 0.2, 0.4, 0.2, 0.7],      # 杀爆2
    #     [0.1, 0.7, 0.75, 0.75, 0.1],    # 侵爆1
    #     [0.1, 0.8, 0.8, 0.8, 0.1],      # 侵爆2
    #     [0.1, 0.9, 0.85, 0.85, 0.1],    # 侵爆3
    #     [0.9, 0.3, 0.5, 0.6, 0.9],      # 子母1
    #     [0.1, 0.2, 0.1, 0.1, 0.7],      # 聚能1
    #     [0.1, 0.15, 0.1, 0.1, 0.7],     # 聚能2
    # ])

    targets = [
        Target(1, 100.0, [(0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), 
                          (0.8, 1.0)]),# 陆基雷达站
        Target(2, 100.0, [(0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), 
                          (0.8, 1.0)]),# 陆基雷达站
        Target(3, 120.0, [(0.2, 1.0), (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), 
                          (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.8, 1.0)]),# 地下指挥所
        Target(4, 120.0, [(0.2, 1.0), (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), 
                          (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.8, 1.0)]),# 地下指挥所
        Target(5, 105.0, [(0.2, 1.0), (0.2, 1.0), (0.2, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0),
                          (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), 
                          (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), 
                          (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),                           
                          (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), 
                          (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.8, 1.0), (0.8, 1.0) ]),# 阵地(雷达车15个，电源车13个，导弹发射车18个，指挥控制车17个)
        Target(6, 105.0, [(0.2, 1.0), (0.2, 1.0), (0.2, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0),
                          (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), 
                          (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), 
                          (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),                           
                          (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), 
                          (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.8, 1.0), (0.8, 1.0) ]),# 阵地(雷达车15个，电源车13个，导弹发射车18个，指挥控制车17个)
    ]

    ammunitions = [
        Ammunition(1, 6.0, 10, {
            1: [0.5, 0.2, 0.6, 0.7, 0.8,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.2],
            2: [0.5, 0.2, 0.6, 0.7, 0.8,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.2],
            3: [0.8, 0.8, 0.8, 0.6, 0.6,
                0.5, 0.8, 0.7, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.8, 0.1], 
            4: [0.8, 0.8, 0.8, 0.6, 0.6,
                0.5, 0.8, 0.7, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.8, 0.1], 
            5: [0.7, 0.7, 0.7, 0.8, 0.8,
                0.8, 0.8, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.8, 0.7, 0.7, 0.6, 0.6,
                0.7, 0.5, 0.7, 0.7, 0.7,
                0.5, 0.6, 0.5, 
                0.5, 0.7, 0.5, 0.5, 0.5, 
                0.5, 0.3, 0.7, 0.5, 0.4,
                0.4, 0.4, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5,                           
                0.4, 0.5, 0.4, 0.5, 0.5, 
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.5, 0.5], 
            6: [0.7, 0.7, 0.7, 0.8, 0.8,
                0.8, 0.8, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.8, 0.7, 0.7, 0.6, 0.6,
                0.7, 0.5, 0.7, 0.7, 0.7,
                0.5, 0.6, 0.5, 
                0.5, 0.7, 0.5, 0.5, 0.5, 
                0.5, 0.3, 0.7, 0.5, 0.4,
                0.4, 0.4, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5,                           
                0.4, 0.5, 0.4, 0.5, 0.5, 
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.5, 0.5], 
        }),# 杀爆1(当量大)
        Ammunition(2, 4.0, 10, {
            1: [0.4, 0.2, 0.5, 0.6, 0.7,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.2],
            2: [0.4, 0.2, 0.5, 0.6, 0.7,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.2],
            3: [0.7, 0.7, 0.7, 0.5, 0.5,
                0.4, 0.7, 0.6, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.7, 0.1], 
            4: [0.7, 0.7, 0.7, 0.5, 0.5,
                0.4, 0.7, 0.6, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.7, 0.1], 
            5: [0.6, 0.6, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.5, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6,
                0.7, 0.6, 0.6, 0.5, 0.5,
                0.6, 0.4, 0.6, 0.6, 0.6,
                0.4, 0.5, 0.4, 
                0.4, 0.6, 0.4, 0.4, 0.4, 
                0.4, 0.2, 0.6, 0.4, 0.3,
                0.3, 0.3, 0.4, 0.4, 0.4,
                0.4, 0.4, 0.4,                           
                0.3, 0.4, 0.3, 0.4, 0.4, 
                0.3, 0.4, 0.3, 0.4, 0.4,
                0.3, 0.4, 0.3, 0.4, 0.4,
                0.4, 0.4], 
            6: [0.6, 0.6, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.5, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6,
                0.7, 0.6, 0.6, 0.5, 0.5,
                0.6, 0.4, 0.6, 0.6, 0.6,
                0.4, 0.5, 0.4, 
                0.4, 0.6, 0.4, 0.4, 0.4, 
                0.4, 0.2, 0.6, 0.4, 0.3,
                0.3, 0.3, 0.4, 0.4, 0.4,
                0.4, 0.4, 0.4,                           
                0.3, 0.4, 0.3, 0.4, 0.4, 
                0.3, 0.4, 0.3, 0.4, 0.4,
                0.3, 0.4, 0.3, 0.4, 0.4,
                0.4, 0.4], 
        }),# 杀爆2(当量小)
        Ammunition(3, 5.0, 10, {
            1: [0.7, 0.7, 0.3, 0.3, 0.5,
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.7],
            2: [0.7, 0.7, 0.3, 0.3, 0.5,
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.7],
            3: [0.6, 0.6, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.7, 0.7, 0.7, 0.7, 0.7, 
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.5, 0.7], 
            4: [0.6, 0.6, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.7, 0.7, 0.7, 0.7, 0.7, 
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.5, 0.7], 
            5: [0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1,                            
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1],
            6: [0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1,                            
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1],  
        }),# 侵爆1(1.8m)
        Ammunition(4, 7.0, 10, {
            1: [0.75, 0.75, 0.35, 0.35, 0.55,
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.75],
            2: [0.75, 0.75, 0.35, 0.35, 0.55,
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.75],
            3: [0.65, 0.65, 0.65, 0.75, 0.75,
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.75, 0.75, 0.75, 0.75, 0.75, 
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.55, 0.75], 
            4: [0.65, 0.65, 0.65, 0.75, 0.75,
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.75, 0.75, 0.75, 0.75, 0.75, 
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.55, 0.75], 
            5: [0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1,                            
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1], 
            6: [0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1,                            
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1],  
        }),# 侵爆2(6m)
        Ammunition(5, 10.0, 10, {
            1: [0.8, 0.8, 0.4, 0.4, 0.6,
                0.8, 0.8, 0.8, 0.8, 0.8,
                0.8, 0.8, 0.8, 0.8, 0.8,
                0.8],
            2: [0.8, 0.8, 0.4, 0.4, 0.6,
                0.8, 0.8, 0.8, 0.8, 0.8,
                0.8, 0.8, 0.8, 0.8, 0.8,
                0.8],
            3: [0.7, 0.7, 0.7, 0.8, 0.8,
                0.8, 0.8, 0.8, 0.8, 0.8,
                0.8, 0.8, 0.8, 0.8, 0.8, 
                0.8, 0.8, 0.8, 0.8, 0.8,
                0.6, 0.8], 
            4: [0.7, 0.7, 0.7, 0.8, 0.8,
                0.8, 0.8, 0.8, 0.8, 0.8,
                0.8, 0.8, 0.8, 0.8, 0.8, 
                0.8, 0.8, 0.8, 0.8, 0.8,
                0.6, 0.8], 
            5: [0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15,                            
                0.15, 0.15, 0.15, 0.15, 0.15, 
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15], 
            6: [0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15,                            
                0.15, 0.15, 0.15, 0.15, 0.15, 
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15], 
        }),# 侵爆3(61m)
        Ammunition(6, 7.0, 10, {
            1: [0.55, 0.25, 0.65, 0.75, 0.85,
                0.25, 0.25, 0.25, 0.25, 0.25,
                0.25, 0.25, 0.25, 0.25, 0.25,
                0.25],
            2: [0.55, 0.25, 0.65, 0.75, 0.85,
                0.25, 0.25, 0.25, 0.25, 0.25,
                0.25, 0.25, 0.25, 0.25, 0.25,
                0.25],
            3: [0.85, 0.85, 0.85, 0.65, 0.65,
                0.55, 0.85, 0.75, 0.2, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2, 
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.85, 0.2], 
            4: [0.85, 0.85, 0.85, 0.65, 0.65,
                0.55, 0.85, 0.75, 0.2, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2, 
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.85, 0.2],
            5: [0.75, 0.75, 0.75, 0.85, 0.85,
                0.85, 0.85, 0.65, 0.75, 0.75,
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.85, 0.75, 0.75, 0.65, 0.65,
                0.75, 0.55, 0.75, 0.75, 0.75,
                0.55, 0.65, 0.55, 
                0.55, 0.75, 0.55, 0.55, 0.55, 
                0.55, 0.35, 0.75, 0.55, 0.45,
                0.45, 0.45, 0.55, 0.55, 0.55,
                0.55, 0.55, 0.55,                           
                0.45, 0.55, 0.45, 0.55, 0.55, 
                0.45, 0.55, 0.45, 0.55, 0.55,
                0.45, 0.55, 0.45, 0.55, 0.55,
                0.55, 0.55], 
            6: [0.75, 0.75, 0.75, 0.85, 0.85,
                0.85, 0.85, 0.65, 0.75, 0.75,
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.85, 0.75, 0.75, 0.65, 0.65,
                0.75, 0.55, 0.75, 0.75, 0.75,
                0.55, 0.65, 0.55, 
                0.55, 0.75, 0.55, 0.55, 0.55, 
                0.55, 0.35, 0.75, 0.55, 0.45,
                0.45, 0.45, 0.55, 0.55, 0.55,
                0.55, 0.55, 0.55,                           
                0.45, 0.55, 0.45, 0.55, 0.55, 
                0.45, 0.55, 0.45, 0.55, 0.55,
                0.45, 0.55, 0.45, 0.55, 0.55,
                0.55, 0.55],
        }),# 子母1
        Ammunition(7, 3.0, 10, {
            1: [0.1, 0.1, 0.1, 0.3, 0.3,
                0.4, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1],
            2: [0.1, 0.1, 0.1, 0.3, 0.3,
                0.4, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1],
            3: [0.6, 0.6, 0.6, 0.6, 0.6,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1], 
            4: [0.6, 0.6, 0.6, 0.6, 0.6,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1], 
            5: [0.6, 0.6, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.5, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6,
                0.7, 0.6, 0.6, 0.5, 0.5,
                0.6, 0.4, 0.6, 0.6, 0.6,
                0.4, 0.5, 0.4, 
                0.4, 0.6, 0.4, 0.4, 0.4, 
                0.4, 0.2, 0.6, 0.4, 0.3,
                0.3, 0.3, 0.4, 0.4, 0.4,
                0.4, 0.4, 0.4,                           
                0.4, 0.5, 0.4, 0.5, 0.5, 
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.5, 0.5], 
            6: [0.6, 0.6, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.5, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6,
                0.7, 0.6, 0.6, 0.5, 0.5,
                0.6, 0.4, 0.6, 0.6, 0.6,
                0.4, 0.5, 0.4, 
                0.4, 0.6, 0.4, 0.4, 0.4, 
                0.4, 0.2, 0.6, 0.4, 0.3,
                0.3, 0.3, 0.4, 0.4, 0.4,
                0.4, 0.4, 0.4,                           
                0.4, 0.5, 0.4, 0.5, 0.5, 
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.5, 0.5], 
        }),# 聚能1(1.3m破甲)
        Ammunition(8, 2.0, 10, {
            1: [0.1, 0.1, 0.1, 0.2, 0.2,
                0.3, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1],
            2: [0.1, 0.1, 0.1, 0.2, 0.2,
                0.3, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1],
            3: [0.5, 0.5, 0.5, 0.5, 0.5,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1], 
            4: [0.5, 0.5, 0.5, 0.5, 0.5,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1], 
            5: [0.6, 0.6, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.5, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6,
                0.7, 0.6, 0.6, 0.5, 0.5,
                0.6, 0.4, 0.6, 0.6, 0.6,
                0.4, 0.5, 0.4, 
                0.4, 0.6, 0.4, 0.4, 0.4, 
                0.4, 0.2, 0.6, 0.4, 0.3,
                0.3, 0.3, 0.4, 0.4, 0.4,
                0.4, 0.4, 0.4,                           
                0.4, 0.5, 0.4, 0.5, 0.5, 
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.5, 0.5], 
            6: [0.6, 0.6, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.5, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6,
                0.7, 0.6, 0.6, 0.5, 0.5,
                0.6, 0.4, 0.6, 0.6, 0.6,
                0.4, 0.5, 0.4, 
                0.4, 0.6, 0.4, 0.4, 0.4, 
                0.4, 0.2, 0.6, 0.4, 0.3,
                0.3, 0.3, 0.4, 0.4, 0.4,
                0.4, 0.4, 0.4,                           
                0.4, 0.5, 0.4, 0.5, 0.5, 
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.5, 0.5], 
        })# 聚能2(1.1m破甲)
    ]
    # 2-指挥所 3-雷达 5-阵地
    adaptability_matrix = np.array([
        [0.85, 0.85, 0.25, 0.25, 0.8, 0.8],   # 杀爆1
        [0.8, 0.8, 0.2, 0.2, 0.7, 0.7],      # 杀爆2
        [0.75, 0.75, 0.7, 0.7, 0.1, 0.1],    # 侵爆1
        [0.8, 0.8, 0.8, 0.8, 0.1, 0.1],      # 侵爆2
        [0.85, 0.85, 0.9, 0.9, 0.1, 0.1],    # 侵爆3
        [0.5, 0.5, 0.3, 0.3, 0.9, 0.9],      # 子母1
        [0.1, 0.1, 0.2, 0.2, 0.7, 0.7],      # 聚能1
        [0.1, 0.1, 0.15, 0.15, 0.7, 0.7],     # 聚能2
    ])

    solver = AHLNSGAII_Solver(targets, ammunitions, adaptability_matrix, 0.85)

    print("开始求解静态弹目匹配问题（AHL-NSGA-II优化版）...")
    start_time = time.time()
    # 接收帕累托解和收敛代数
    pareto_solutions, convergence_gen = solver.solve()
    total_runtime = time.time() - start_time

    # 输出核心对比信息
    print(f"\n优化算法结果:")
    print(f"   收敛代数: {convergence_gen}")
    print(f"   整体求解总时间: {total_runtime:.3f} 秒")
    print(f"   唯一帕累托解数量: {len(pareto_solutions)}")

    print(f"\n找到 {len(pareto_solutions)} 个唯一且满足毁伤要求的帕累托最优解:")
    for i, (solution, objectives, gen, sol_timestamp) in enumerate(pareto_solutions):
        ce, time_cost = objectives[:2]
        # 计算解的相对生成时间（相对于整体开始时间）
        relative_time = sol_timestamp - start_time
        print(f"\n解 {i+1}:")
        print(f"  费效比 = {ce:.4f}, 任务时间 = {time_cost:.2f}")

        print(f"  生成代数 = {gen}, 生成时间 = {relative_time:.3f} 秒")
        print("  分配详情:")

        for j, target in enumerate(targets):
            target_rounds = 0
            ammo_details = []
            for i_ammo, ammo in enumerate(ammunitions):
                rounds = solution[i_ammo, j]
                if rounds > 0:
                    target_rounds += rounds
                    ammo_details.append(f"弹药{ammo.id}:{rounds}发")
            actual_damage = solver.calculate_actual_damage(solution, j)
            status = "✓" if actual_damage >= 0.8 else "✗"
            print(f"    目标{target.id}: 总计{target_rounds}发, 毁伤概率: {actual_damage:.3f} {status}")
            if ammo_details:
                print(f"      ({', '.join(ammo_details)})")

    print(f"\n整体求解总时间: {total_runtime:.3f} 秒")

# if __name__ == "__main__":
#     main()

def run_multiple_tests(test_times: int = 15):
    # 初始化统计容器
    stats = {
        "best_ce_list": [],  # 每次测试的最优费效比
        "avg_ce_list": [],  # 每次测试的平均费效比（去重后）
        "best_task_time": [],  # 新增：最优任务时间
        "avg_task_time": [],    # 新增：平均任务时间
        "feasible_ratio_list": [],  # 每次测试的达标解比例
        "convergence_gen_list": [],  # 每次测试的收敛代数
        "total_time_list": [],  # 每次测试的总求解时间
        "unique_solutions_num_list": [],  # 每次测试去重后解的数量
        "crowding_distance_avg_list": []  # 每次测试的拥挤度均值
    }

    # 2. 新增：全局最优解跟踪变量
    global_best_ce = -float('inf')
    global_best_mt = -float('inf')  # 最优任务时间
    global_best_solution = None  # 最优解的分配矩阵
    global_best_metadata = None  # (目标值, 生成代数, 测试序号, 相对时间)

# 测试参数(8*5个目标)
    targets = [
        Target(1, 80.0, [(0.5, 1.0)]),# 人员集群
        Target(2, 120.0, [(0.2, 1.0), (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), 
                          (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.8, 1.0)]),# 地下指挥所
        Target(3, 100.0, [(0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), 
                          (0.8, 1.0)]),# 陆基雷达站
        Target(4, 110.0, [(0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.2, 1.0),
                          (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.5, 1.0),
                          (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.2, 1.0), (0.8, 1.0), 
                          (0.5, 1.0)]),# 机场
        Target(5, 105.0, [(0.2, 1.0), (0.2, 1.0), (0.2, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
                          (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0),
                          (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), 
                          (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), 
                          (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),                           
                          (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), 
                          (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0),
                          (0.8, 1.0), (0.8, 1.0) ]),# 阵地(雷达车15个，电源车13个，导弹发射车18个，指挥控制车17个)
    ]

    ammunitions = [
        Ammunition(1, 13.0, 10, {
            1: [0.8], 
            2: [0.8, 0.8, 0.8, 0.6, 0.6,
                0.5, 0.8, 0.7, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.8, 0.1], 
            3: [0.5, 0.2, 0.6, 0.7, 0.8,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.2],
            4: [0.2, 0.2, 0.8, 0.7, 0.7,
                0.7, 0.7, 0.2, 0.2, 0.3,
                0.2, 0.2, 0.2, 0.2, 0.2, 
                0.6], 
            5: [0.7, 0.7, 0.7, 0.8, 0.8,
                0.8, 0.8, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.8, 0.7, 0.7, 0.6, 0.6,
                0.7, 0.5, 0.7, 0.7, 0.7,
                0.5, 0.6, 0.5, 
                0.5, 0.7, 0.5, 0.5, 0.5, 
                0.5, 0.3, 0.7, 0.5, 0.4,
                0.4, 0.4, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5,                           
                0.4, 0.5, 0.4, 0.5, 0.5, 
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.5, 0.5], 
        }),# 杀爆1(当量大)
        Ammunition(2, 10.0, 10, {
            1: [0.7], 
            2: [0.7, 0.7, 0.7, 0.5, 0.5,
                0.4, 0.7, 0.6, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.7, 0.1], 
            3: [0.4, 0.2, 0.5, 0.6, 0.7,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.2],
            4: [0.2, 0.2, 0.7, 0.6, 0.6,
                0.6, 0.6, 0.2, 0.2, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2, 
                0.5], 
            5: [0.6, 0.6, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.5, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6,
                0.7, 0.6, 0.6, 0.5, 0.5,
                0.6, 0.4, 0.6, 0.6, 0.6,
                0.4, 0.5, 0.4, 
                0.4, 0.6, 0.4, 0.4, 0.4, 
                0.4, 0.2, 0.6, 0.4, 0.3,
                0.3, 0.3, 0.4, 0.4, 0.4,
                0.4, 0.4, 0.4,                           
                0.3, 0.4, 0.3, 0.4, 0.4, 
                0.3, 0.4, 0.3, 0.4, 0.4,
                0.3, 0.4, 0.3, 0.4, 0.4,
                0.4, 0.4], 
        }),# 杀爆2(当量小)
        Ammunition(3, 20.0, 10, {
            1: [0.1], 
            2: [0.6, 0.6, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.7, 0.7, 0.7, 0.7, 0.7, 
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.5, 0.7], 
            3: [0.7, 0.7, 0.3, 0.3, 0.5,
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.7, 0.7, 0.7, 0.7, 0.7,
                0.7],
            4: [0.7, 0.7, 0.4, 0.4, 0.4,
                0.4, 0.4, 0.7, 0.7, 0.7,
                0.7, 0.7, 0.7, 0.7, 0.6, 
                0.7], 
            5: [0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1,                            
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1], 
        }),# 侵爆1(1.8m)
        Ammunition(4, 25.0, 10, {
            1: [0.1], 
            2: [0.65, 0.65, 0.65, 0.75, 0.75,
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.75, 0.75, 0.75, 0.75, 0.75, 
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.55, 0.75], 
            3: [0.75, 0.75, 0.35, 0.35, 0.55,
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.75],
            4: [0.75, 0.75, 0.45, 0.45, 0.45,
                0.45, 0.45, 0.75, 0.75, 0.75,
                0.75, 0.75, 0.75, 0.75, 0.65, 
                0.75], 
            5: [0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1,                            
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1], 
        }),# 侵爆2(6m)
        Ammunition(5, 40.0, 10, {
            1: [0.2], 
            2: [0.7, 0.7, 0.7, 0.8, 0.8,
                0.8, 0.8, 0.8, 0.8, 0.8,
                0.8, 0.8, 0.8, 0.8, 0.8, 
                0.8, 0.8, 0.8, 0.8, 0.8,
                0.6, 0.8], 
            3: [0.8, 0.8, 0.4, 0.4, 0.6,
                0.8, 0.8, 0.8, 0.8, 0.8,
                0.8, 0.8, 0.8, 0.8, 0.8,
                0.8],
            4: [0.8, 0.8, 0.4, 0.4, 0.4,
                0.4, 0.4, 0.8, 0.8, 0.8,
                0.8, 0.8, 0.8, 0.8, 0.7, 
                0.8], 
            5: [0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15,                            
                0.15, 0.15, 0.15, 0.15, 0.15, 
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15], 
        }),# 侵爆3(61m)
        Ammunition(6, 27.0, 10, {
            1: [0.9], 
            2: [0.85, 0.85, 0.85, 0.65, 0.65,
                0.55, 0.85, 0.75, 0.2, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2, 
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.85, 0.2], 
            3: [0.55, 0.25, 0.65, 0.75, 0.85,
                0.25, 0.25, 0.25, 0.25, 0.25,
                0.25, 0.25, 0.25, 0.25, 0.25,
                0.25],
            4: [0.3, 0.3, 0.8, 0.75, 0.7,
                0.7, 0.75, 0.3, 0.3, 0.35,
                0.3, 0.3, 0.3, 0.3, 0.3, 
                0.6], 
            5: [0.75, 0.75, 0.75, 0.85, 0.85,
                0.85, 0.85, 0.65, 0.75, 0.75,
                0.75, 0.75, 0.75, 0.75, 0.75,
                0.85, 0.75, 0.75, 0.65, 0.65,
                0.75, 0.55, 0.75, 0.75, 0.75,
                0.55, 0.65, 0.55, 
                0.55, 0.75, 0.55, 0.55, 0.55, 
                0.55, 0.35, 0.75, 0.55, 0.45,
                0.45, 0.45, 0.55, 0.55, 0.55,
                0.55, 0.55, 0.55,                           
                0.45, 0.55, 0.45, 0.55, 0.55, 
                0.45, 0.55, 0.45, 0.55, 0.55,
                0.45, 0.55, 0.45, 0.55, 0.55,
                0.55, 0.55], 
        }),# 子母1
        Ammunition(7, 20.0, 10, {
            1: [0.1], 
            2: [0.6, 0.6, 0.6, 0.6, 0.6,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1], 
            3: [0.1, 0.1, 0.1, 0.3, 0.3,
                0.4, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1],
            4: [0.1, 0.1, 0.4, 0.2, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1], 
            5: [0.6, 0.6, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.5, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6,
                0.7, 0.6, 0.6, 0.5, 0.5,
                0.6, 0.4, 0.6, 0.6, 0.6,
                0.4, 0.5, 0.4, 
                0.4, 0.6, 0.4, 0.4, 0.4, 
                0.4, 0.2, 0.6, 0.4, 0.3,
                0.3, 0.3, 0.4, 0.4, 0.4,
                0.4, 0.4, 0.4,                           
                0.4, 0.5, 0.4, 0.5, 0.5, 
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.5, 0.5], 
        }),# 聚能1(1.3m破甲)
        Ammunition(8, 18.0, 10, {
            1: [0.1], 
            2: [0.5, 0.5, 0.5, 0.5, 0.5,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1], 
            3: [0.1, 0.1, 0.1, 0.2, 0.2,
                0.3, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1],
            4: [0.1, 0.1, 0.3, 0.15, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 
                0.1], 
            5: [0.6, 0.6, 0.6, 0.7, 0.7,
                0.7, 0.7, 0.5, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6,
                0.7, 0.6, 0.6, 0.5, 0.5,
                0.6, 0.4, 0.6, 0.6, 0.6,
                0.4, 0.5, 0.4, 
                0.4, 0.6, 0.4, 0.4, 0.4, 
                0.4, 0.2, 0.6, 0.4, 0.3,
                0.3, 0.3, 0.4, 0.4, 0.4,
                0.4, 0.4, 0.4,                           
                0.4, 0.5, 0.4, 0.5, 0.5, 
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.4, 0.5, 0.4, 0.5, 0.5,
                0.5, 0.5], 
        })# 聚能2(1.1m破甲)
    ]

    adaptability_matrix = np.array([
        [0.9, 0.25, 0.45, 0.25, 0.8],   # 杀爆1
        [0.8, 0.2, 0.4, 0.2, 0.7],      # 杀爆2
        [0.1, 0.7, 0.75, 0.75, 0.1],    # 侵爆1
        [0.1, 0.8, 0.8, 0.8, 0.1],      # 侵爆2
        [0.1, 0.9, 0.85, 0.85, 0.1],    # 侵爆3
        [0.9, 0.3, 0.5, 0.6, 0.9],      # 子母1
        [0.1, 0.2, 0.1, 0.1, 0.7],      # 聚能1
        [0.1, 0.15, 0.1, 0.1, 0.7],     # 聚能2
    ])



    # 真实测试参数(8*9)
    # targets = [
    #     Target(1, 80.0, [(0.5, 1.0)]),# 人员集群
    #     Target(2, 120.0, [(0.2, 1.0), (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), 
    #                       (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.8, 1.0)]),# 地下指挥所
    #     Target(3, 100.0, [(0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), 
    #                       (0.8, 1.0)]),# 陆基雷达站
    #     Target(4, 110.0, [(0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.2, 1.0),
    #                       (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.5, 1.0),
    #                       (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.2, 1.0), (0.8, 1.0), 
    #                       (0.5, 1.0)]),# 机场
    #     Target(5, 105.0, [(0.2, 1.0), (0.2, 1.0), (0.2, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0),
    #                       (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), 
    #                       (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), 
    #                       (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),                           
    #                       (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), 
    #                       (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.8, 1.0), (0.8, 1.0) ]),# 阵地(雷达车15个，电源车13个，导弹发射车18个，指挥控制车17个)
    #     Target(6, 80.0, [(0.5, 1.0)]),# 人员集群
    #     Target(7, 100.0, [(0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), 
    #                       (0.8, 1.0)]),# 陆基雷达站
    #     Target(8, 100.0, [(0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), 
    #                       (0.8, 1.0)]),# 陆基雷达站
    #     Target(9, 105.0, [(0.2, 1.0), (0.2, 1.0), (0.2, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
    #                       (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0),
    #                       (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), 
    #                       (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), 
    #                       (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),                           
    #                       (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), 
    #                       (0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0),
    #                       (0.8, 1.0), (0.8, 1.0) ]),# 阵地(雷达车15个，电源车13个，导弹发射车18个，指挥控制车17个)
    # ]

    # ammunitions = [
    #     Ammunition(1, 13.0, 10, {
    #         1: [0.8], 
    #         2: [0.8, 0.8, 0.8, 0.6, 0.6,
    #             0.5, 0.8, 0.7, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.8, 0.1], 
    #         3: [0.5, 0.2, 0.6, 0.7, 0.8,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2],
    #         4: [0.2, 0.2, 0.8, 0.7, 0.7,
    #             0.7, 0.7, 0.2, 0.2, 0.3,
    #             0.2, 0.2, 0.2, 0.2, 0.2, 
    #             0.6], 
    #         5: [0.7, 0.7, 0.7, 0.8, 0.8,
    #             0.8, 0.8, 0.6, 0.7, 0.7,
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.8, 0.7, 0.7, 0.6, 0.6,
    #             0.7, 0.5, 0.7, 0.7, 0.7,
    #             0.5, 0.6, 0.5, 
    #             0.5, 0.7, 0.5, 0.5, 0.5, 
    #             0.5, 0.3, 0.7, 0.5, 0.4,
    #             0.4, 0.4, 0.5, 0.5, 0.5,
    #             0.5, 0.5, 0.5,                           
    #             0.4, 0.5, 0.4, 0.5, 0.5, 
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.5, 0.5], 
    #         6: [0.8], 
    #         7: [0.5, 0.2, 0.6, 0.7, 0.8,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2],
    #         8: [0.5, 0.2, 0.6, 0.7, 0.8,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2],
    #         9: [0.7, 0.7, 0.7, 0.8, 0.8,
    #             0.8, 0.8, 0.6, 0.7, 0.7,
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.8, 0.7, 0.7, 0.6, 0.6,
    #             0.7, 0.5, 0.7, 0.7, 0.7,
    #             0.5, 0.6, 0.5, 
    #             0.5, 0.7, 0.5, 0.5, 0.5, 
    #             0.5, 0.3, 0.7, 0.5, 0.4,
    #             0.4, 0.4, 0.5, 0.5, 0.5,
    #             0.5, 0.5, 0.5,                           
    #             0.4, 0.5, 0.4, 0.5, 0.5, 
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.5, 0.5], 
    #     }),# 杀爆1(当量大)
    #     Ammunition(2, 10.0, 10, {
    #         1: [0.7], 
    #         2: [0.7, 0.7, 0.7, 0.5, 0.5,
    #             0.4, 0.7, 0.6, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.7, 0.1], 
    #         3: [0.4, 0.2, 0.5, 0.6, 0.7,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2],
    #         4: [0.2, 0.2, 0.7, 0.6, 0.6,
    #             0.6, 0.6, 0.2, 0.2, 0.2,
    #             0.2, 0.2, 0.2, 0.2, 0.2, 
    #             0.5], 
    #         5: [0.6, 0.6, 0.6, 0.7, 0.7,
    #             0.7, 0.7, 0.5, 0.6, 0.6,
    #             0.6, 0.6, 0.6, 0.6, 0.6,
    #             0.7, 0.6, 0.6, 0.5, 0.5,
    #             0.6, 0.4, 0.6, 0.6, 0.6,
    #             0.4, 0.5, 0.4, 
    #             0.4, 0.6, 0.4, 0.4, 0.4, 
    #             0.4, 0.2, 0.6, 0.4, 0.3,
    #             0.3, 0.3, 0.4, 0.4, 0.4,
    #             0.4, 0.4, 0.4,                           
    #             0.3, 0.4, 0.3, 0.4, 0.4, 
    #             0.3, 0.4, 0.3, 0.4, 0.4,
    #             0.3, 0.4, 0.3, 0.4, 0.4,
    #             0.4, 0.4], 
    #         6: [0.7], 
    #         7: [0.4, 0.2, 0.5, 0.6, 0.7,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2],
    #         8: [0.4, 0.2, 0.5, 0.6, 0.7,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.2],
    #         9: [0.6, 0.6, 0.6, 0.7, 0.7,
    #             0.7, 0.7, 0.5, 0.6, 0.6,
    #             0.6, 0.6, 0.6, 0.6, 0.6,
    #             0.7, 0.6, 0.6, 0.5, 0.5,
    #             0.6, 0.4, 0.6, 0.6, 0.6,
    #             0.4, 0.5, 0.4, 
    #             0.4, 0.6, 0.4, 0.4, 0.4, 
    #             0.4, 0.2, 0.6, 0.4, 0.3,
    #             0.3, 0.3, 0.4, 0.4, 0.4,
    #             0.4, 0.4, 0.4,                           
    #             0.3, 0.4, 0.3, 0.4, 0.4, 
    #             0.3, 0.4, 0.3, 0.4, 0.4,
    #             0.3, 0.4, 0.3, 0.4, 0.4,
    #             0.4, 0.4], 
    #     }),# 杀爆2(当量小)
    #     Ammunition(3, 20.0, 10, {
    #         1: [0.1], 
    #         2: [0.6, 0.6, 0.6, 0.7, 0.7,
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.7, 0.7, 0.7, 0.7, 0.7, 
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.5, 0.7], 
    #         3: [0.7, 0.7, 0.3, 0.3, 0.5,
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.7],
    #         4: [0.7, 0.7, 0.4, 0.4, 0.4,
    #             0.4, 0.4, 0.7, 0.7, 0.7,
    #             0.7, 0.7, 0.7, 0.7, 0.6, 
    #             0.7], 
    #         5: [0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1,                            
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1], 
    #         6: [0.1], 
    #         7: [0.7, 0.7, 0.3, 0.3, 0.5,
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.7],
    #         8: [0.7, 0.7, 0.3, 0.3, 0.5,
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.7, 0.7, 0.7, 0.7, 0.7,
    #             0.7],
    #         9: [0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1,                            
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1], 
    #     }),# 侵爆1(1.8m)
    #     Ammunition(4, 25.0, 10, {
    #         1: [0.1], 
    #         2: [0.65, 0.65, 0.65, 0.75, 0.75,
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.75, 0.75, 0.75, 0.75, 0.75, 
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.55, 0.75], 
    #         3: [0.75, 0.75, 0.35, 0.35, 0.55,
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.75],
    #         4: [0.75, 0.75, 0.45, 0.45, 0.45,
    #             0.45, 0.45, 0.75, 0.75, 0.75,
    #             0.75, 0.75, 0.75, 0.75, 0.65, 
    #             0.75], 
    #         5: [0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1,                            
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1], 
    #         6: [0.1], 
    #         7: [0.75, 0.75, 0.35, 0.35, 0.55,
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.75],
    #         8: [0.75, 0.75, 0.35, 0.35, 0.55,
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.75],
    #         9: [0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1,                            
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1],    
    #     }),# 侵爆2(6m)
    #     Ammunition(5, 40.0, 10, {
    #         1: [0.2], 
    #         2: [0.7, 0.7, 0.7, 0.8, 0.8,
    #             0.8, 0.8, 0.8, 0.8, 0.8,
    #             0.8, 0.8, 0.8, 0.8, 0.8, 
    #             0.8, 0.8, 0.8, 0.8, 0.8,
    #             0.6, 0.8], 
    #         3: [0.8, 0.8, 0.4, 0.4, 0.6,
    #             0.8, 0.8, 0.8, 0.8, 0.8,
    #             0.8, 0.8, 0.8, 0.8, 0.8,
    #             0.8],
    #         4: [0.8, 0.8, 0.4, 0.4, 0.4,
    #             0.4, 0.4, 0.8, 0.8, 0.8,
    #             0.8, 0.8, 0.8, 0.8, 0.7, 
    #             0.8], 
    #         5: [0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15,                            
    #             0.15, 0.15, 0.15, 0.15, 0.15, 
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15], 
    #         6: [0.2],
    #         7: [0.8, 0.8, 0.4, 0.4, 0.6,
    #             0.8, 0.8, 0.8, 0.8, 0.8,
    #             0.8, 0.8, 0.8, 0.8, 0.8,
    #             0.8],
    #         8: [0.8, 0.8, 0.4, 0.4, 0.6,
    #             0.8, 0.8, 0.8, 0.8, 0.8,
    #             0.8, 0.8, 0.8, 0.8, 0.8,
    #             0.8],
    #         9: [0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15,                            
    #             0.15, 0.15, 0.15, 0.15, 0.15, 
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15, 0.15, 0.15, 0.15,
    #             0.15, 0.15],
    #     }),# 侵爆3(61m)
    #     Ammunition(6, 27.0, 10, {
    #         1: [0.9], 
    #         2: [0.85, 0.85, 0.85, 0.65, 0.65,
    #             0.55, 0.85, 0.75, 0.2, 0.2,
    #             0.2, 0.2, 0.2, 0.2, 0.2, 
    #             0.2, 0.2, 0.2, 0.2, 0.2,
    #             0.85, 0.2], 
    #         3: [0.55, 0.25, 0.65, 0.75, 0.85,
    #             0.25, 0.25, 0.25, 0.25, 0.25,
    #             0.25, 0.25, 0.25, 0.25, 0.25,
    #             0.25],
    #         4: [0.3, 0.3, 0.8, 0.75, 0.7,
    #             0.7, 0.75, 0.3, 0.3, 0.35,
    #             0.3, 0.3, 0.3, 0.3, 0.3, 
    #             0.6], 
    #         5: [0.75, 0.75, 0.75, 0.85, 0.85,
    #             0.85, 0.85, 0.65, 0.75, 0.75,
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.85, 0.75, 0.75, 0.65, 0.65,
    #             0.75, 0.55, 0.75, 0.75, 0.75,
    #             0.55, 0.65, 0.55, 
    #             0.55, 0.75, 0.55, 0.55, 0.55, 
    #             0.55, 0.35, 0.75, 0.55, 0.45,
    #             0.45, 0.45, 0.55, 0.55, 0.55,
    #             0.55, 0.55, 0.55,                           
    #             0.45, 0.55, 0.45, 0.55, 0.55, 
    #             0.45, 0.55, 0.45, 0.55, 0.55,
    #             0.45, 0.55, 0.45, 0.55, 0.55,
    #             0.55, 0.55], 
    #         6: [0.9],
    #         7: [0.55, 0.25, 0.65, 0.75, 0.85,
    #             0.25, 0.25, 0.25, 0.25, 0.25,
    #             0.25, 0.25, 0.25, 0.25, 0.25,
    #             0.25],
    #         8: [0.55, 0.25, 0.65, 0.75, 0.85,
    #             0.25, 0.25, 0.25, 0.25, 0.25,
    #             0.25, 0.25, 0.25, 0.25, 0.25,
    #             0.25],
    #         9: [0.75, 0.75, 0.75, 0.85, 0.85,
    #             0.85, 0.85, 0.65, 0.75, 0.75,
    #             0.75, 0.75, 0.75, 0.75, 0.75,
    #             0.85, 0.75, 0.75, 0.65, 0.65,
    #             0.75, 0.55, 0.75, 0.75, 0.75,
    #             0.55, 0.65, 0.55, 
    #             0.55, 0.75, 0.55, 0.55, 0.55, 
    #             0.55, 0.35, 0.75, 0.55, 0.45,
    #             0.45, 0.45, 0.55, 0.55, 0.55,
    #             0.55, 0.55, 0.55,                           
    #             0.45, 0.55, 0.45, 0.55, 0.55, 
    #             0.45, 0.55, 0.45, 0.55, 0.55,
    #             0.45, 0.55, 0.45, 0.55, 0.55,
    #             0.55, 0.55],
    #     }),# 子母1
    #     Ammunition(7, 20.0, 10, {
    #         1: [0.1], 
    #         2: [0.6, 0.6, 0.6, 0.6, 0.6,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1], 
    #         3: [0.1, 0.1, 0.1, 0.3, 0.3,
    #             0.4, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1],
    #         4: [0.1, 0.1, 0.4, 0.2, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1], 
    #         5: [0.6, 0.6, 0.6, 0.7, 0.7,
    #             0.7, 0.7, 0.5, 0.6, 0.6,
    #             0.6, 0.6, 0.6, 0.6, 0.6,
    #             0.7, 0.6, 0.6, 0.5, 0.5,
    #             0.6, 0.4, 0.6, 0.6, 0.6,
    #             0.4, 0.5, 0.4, 
    #             0.4, 0.6, 0.4, 0.4, 0.4, 
    #             0.4, 0.2, 0.6, 0.4, 0.3,
    #             0.3, 0.3, 0.4, 0.4, 0.4,
    #             0.4, 0.4, 0.4,                           
    #             0.4, 0.5, 0.4, 0.5, 0.5, 
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.5, 0.5], 
    #         6: [0.1],
    #         7: [0.1, 0.1, 0.1, 0.3, 0.3,
    #             0.4, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1],
    #         8: [0.1, 0.1, 0.1, 0.3, 0.3,
    #             0.4, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1],
    #         9: [0.6, 0.6, 0.6, 0.7, 0.7,
    #             0.7, 0.7, 0.5, 0.6, 0.6,
    #             0.6, 0.6, 0.6, 0.6, 0.6,
    #             0.7, 0.6, 0.6, 0.5, 0.5,
    #             0.6, 0.4, 0.6, 0.6, 0.6,
    #             0.4, 0.5, 0.4, 
    #             0.4, 0.6, 0.4, 0.4, 0.4, 
    #             0.4, 0.2, 0.6, 0.4, 0.3,
    #             0.3, 0.3, 0.4, 0.4, 0.4,
    #             0.4, 0.4, 0.4,                           
    #             0.4, 0.5, 0.4, 0.5, 0.5, 
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.5, 0.5],
    #     }),# 聚能1(1.3m破甲)
    #     Ammunition(8, 18.0, 10, {
    #         1: [0.1], 
    #         2: [0.5, 0.5, 0.5, 0.5, 0.5,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1], 
    #         3: [0.1, 0.1, 0.1, 0.2, 0.2,
    #             0.3, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1],
    #         4: [0.1, 0.1, 0.3, 0.15, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1, 
    #             0.1], 
    #         5: [0.6, 0.6, 0.6, 0.7, 0.7,
    #             0.7, 0.7, 0.5, 0.6, 0.6,
    #             0.6, 0.6, 0.6, 0.6, 0.6,
    #             0.7, 0.6, 0.6, 0.5, 0.5,
    #             0.6, 0.4, 0.6, 0.6, 0.6,
    #             0.4, 0.5, 0.4, 
    #             0.4, 0.6, 0.4, 0.4, 0.4, 
    #             0.4, 0.2, 0.6, 0.4, 0.3,
    #             0.3, 0.3, 0.4, 0.4, 0.4,
    #             0.4, 0.4, 0.4,                           
    #             0.4, 0.5, 0.4, 0.5, 0.5, 
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.5, 0.5], 
    #         6: [0.1],
    #         7: [0.1, 0.1, 0.1, 0.2, 0.2,
    #             0.3, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1],
    #         8: [0.1, 0.1, 0.1, 0.2, 0.2,
    #             0.3, 0.1, 0.1, 0.1, 0.1,
    #             0.1, 0.1, 0.1, 0.1, 0.1,
    #             0.1],
    #         9: [0.6, 0.6, 0.6, 0.7, 0.7,
    #             0.7, 0.7, 0.5, 0.6, 0.6,
    #             0.6, 0.6, 0.6, 0.6, 0.6,
    #             0.7, 0.6, 0.6, 0.5, 0.5,
    #             0.6, 0.4, 0.6, 0.6, 0.6,
    #             0.4, 0.5, 0.4, 
    #             0.4, 0.6, 0.4, 0.4, 0.4, 
    #             0.4, 0.2, 0.6, 0.4, 0.3,
    #             0.3, 0.3, 0.4, 0.4, 0.4,
    #             0.4, 0.4, 0.4,                           
    #             0.4, 0.5, 0.4, 0.5, 0.5, 
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.4, 0.5, 0.4, 0.5, 0.5,
    #             0.5, 0.5],
    #     })# 聚能2(1.1m破甲)
    # ]

    # adaptability_matrix = np.array([
    #     [0.9, 0.25, 0.45, 0.25, 0.8, 0.9, 0.45, 0.45, 0.8],   # 杀爆1
    #     [0.8, 0.2, 0.4, 0.2, 0.7, 0.8, 0.4, 0.4, 0.7],      # 杀爆2
    #     [0.1, 0.7, 0.75, 0.75, 0.1, 0.1, 0.75, 0.75, 0.1],    # 侵爆1
    #     [0.1, 0.8, 0.8, 0.8, 0.1, 0.1, 0.8, 0.8, 0.1],      # 侵爆2
    #     [0.1, 0.9, 0.85, 0.85, 0.1, 0.1, 0.85, 0.85, 0.1],    # 侵爆3
    #     [0.9, 0.3, 0.5, 0.6, 0.9, 0.9, 0.5, 0.5, 0.9],      # 子母1
    #     [0.1, 0.2, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.7],      # 聚能1
    #     [0.1, 0.15, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.7],     # 聚能2
    # ])

    solver = AHLNSGAII_Solver(targets, ammunitions, adaptability_matrix, 0.8)
    print(f"开始 {test_times} 次独立测试（含全局最优解跟踪）...")
    for test_idx in range(test_times):
        print(f"\n=== 第 {test_idx + 1}/{test_times} 次测试 ===")
        # 固定随机种子（与原始测试一致）
        random.seed(test_idx)
        np.random.seed(test_idx)

        # 初始化求解器（使用你的原始求解器类，如StaticWeaponTargetAssignment或StaticWeaponTargetAssignmentOriginal）
        # solver = StaticWeaponTargetAssignment(targets, ammunitions, adaptability_matrix, 0.8)

        # 执行求解
        start_time = time.time()
        pareto_solutions, convergence_gen = solver.solve()
        total_time = time.time() - start_time

        # 跳过无效测试
        if not pareto_solutions:
            print("本次测试无有效解，跳过")
            continue

        # 4. 原始统计逻辑：计算当前测试的各项指标
        current_best_ce = max([obj[0] for sol, obj, _, _ in pareto_solutions])
        avg_ce = np.mean([obj[0] for sol, obj, _, _ in pareto_solutions])

        # 新增：统计任务时间（目标2：最小化时间）
        current_task_times = [obj[1] for sol, obj, _, _ in pareto_solutions]
        current_best_task_time = min(current_task_times)
        current_avg_task_time = np.mean(current_task_times)

        feasible_ratio = 1.0
        unique_num = len(pareto_solutions)

        # 计算拥挤度均值
        front_indices = list(range(len(pareto_solutions)))
        crowding_distances = solver.crowding_distance(front_indices, [sol for sol, _, _, _ in pareto_solutions])
        # 过滤无穷大值，只计算有效距离的均值
        valid_distances = [d for d in crowding_distances if d != float('inf')]
        if valid_distances:
            crowding_avg = np.mean(valid_distances)
        else:
            crowding_avg = 0.0  # 或根据实际需求设置默认值
        # crowding_avg = np.mean([d for d in crowding_distances if not np.isinf(d)])

        # 更新统计容器
        stats["best_ce_list"].append(current_best_ce)
        stats["avg_ce_list"].append(avg_ce)
        stats["best_task_time"].append(current_best_task_time)
        stats["avg_task_time"].append(current_avg_task_time)
        stats["feasible_ratio_list"].append(feasible_ratio)
        stats["convergence_gen_list"].append(convergence_gen)
        stats["total_time_list"].append(total_time)
        stats["unique_solutions_num_list"].append(unique_num)
        stats["crowding_distance_avg_list"].append(crowding_avg)

        # 5. 新增：跟踪全局最优解
        # 找到当前测试中费效比最高的解
        current_best_idx = np.argmax([obj[0] for sol, obj, _, _ in pareto_solutions])
        current_best_sol, current_best_obj, current_gen, current_timestamp = pareto_solutions[current_best_idx]

        # 与全局最优对比并更新
        if current_best_obj[0] > global_best_ce:
            global_best_ce = current_best_obj[0]
            global_best_solution = current_best_sol
            global_best_metadata = (current_best_obj, current_gen, test_idx, current_timestamp - start_time)
            print(f"第 {test_idx + 1} 次测试刷新全局最优费效比: {global_best_ce:.3f}")
        else:
            print(f"本次测试最优费效比: {current_best_obj[0]:.4f}（未超过全局最优）")

        # 输出当前测试统计
        print(f"第 {test_idx + 1} 次测试统计：")
        print(f"  最优费效比: {current_best_ce:.3f}, 平均费效比: {avg_ce:.3f}")
        print(f"  收敛代数: {convergence_gen}, 总时间: {total_time:.3f} 秒")

    # 6. 输出原始统计报告
    print("\n" + "=" * 50)
    print("多次独立测试统计报告")
    print("=" * 50)
    final_stats = {}
    for key, values in stats.items():
        if not values:
            final_stats[key] = "无有效数据"
            continue
        mean_val = np.mean(values)
        std_val = np.std(values)
        final_stats[key] = f"{mean_val:.3f} ± {std_val:.3f}"

    # 统计所有测试中出现的最优解
    all_best_ces = stats["best_ce_list"]
    global_max_ce = max(all_best_ces) if all_best_ces else None
    print(f"所有测试中的全局最大费效比: {global_max_ce:.3f}")

    print(f"测试次数: {len(stats['best_ce_list'])} 次（有效测试）")
    print(f"最优费效比: {final_stats['best_ce_list']}")
    print(f"平均费效比: {final_stats['avg_ce_list']}")
    print(f"最优任务时间: {final_stats['best_task_time']}")
    print(f"平均任务时间: {final_stats['avg_task_time']}")
    print(f"收敛代数: {final_stats['convergence_gen_list']}")
    print(f"平均求解时间: {final_stats['total_time_list']} 秒")
    print(f"去重后解数量: {final_stats['unique_solutions_num_list']}")
    print(f"拥挤度均值: {final_stats['crowding_distance_avg_list']}")

    # 7. 新增：输出全局最优解详情
    print("\n" + "=" * 60)
    print(f"全局最优费效比解（最高值: {global_best_ce:.4f}）")
    print("=" * 60)
    if global_best_solution is None:
        print("未找到有效解")
        return

    # 解析最优解元数据
    best_obj, best_gen, test_idx, relative_time = global_best_metadata
    best_time = best_obj[1]

    print(f"测试来源: 第 {test_idx + 1} 次测试")
    print(f"生成代数: 第 {best_gen} 代")
    print(f"生成时间: {relative_time:.3f} 秒")
    print(f"费效比: {global_best_ce:.4f}，任务时间: {best_time:.2f}")
    print("\n详细分配方案:")

    # 输出每个目标的分配详情
    for j, target in enumerate(targets):
        target_total = 0
        ammo_details = []
        for i, ammo in enumerate(ammunitions):
            rounds = global_best_solution[i, j]
            if rounds > 0:
                target_total += rounds
                ammo_details.append(f"弹药{ammo.id}: {rounds}发（成本: {ammo.cost * rounds:.2f}）")

        # 计算毁伤概率
        damage = solver.calculate_actual_damage(global_best_solution, j)
        status = "✓ 达标" if damage >= 0.8 else "✗ 未达标"
        print(f"目标{target.id}:")
        print(f"  总弹药: {target_total}发，毁伤概率: {damage:.4f}（{status}）")
        if ammo_details:
            print(f"  分配详情: {', '.join(ammo_details)}")

    # 全局消耗统计
    total_cost = 0.0
    total_damage_value = 0.0
    for j, target in enumerate(targets):
        total_damage_value += target.value * solver.calculate_actual_damage(global_best_solution, j)
        for i, ammo in enumerate(ammunitions):
            total_cost += ammo.cost * global_best_solution[i, j]

    print("\n全局统计:")
    print(f"总弹药消耗: {np.sum(global_best_solution)}发")
    print(f"总成本: {total_cost:.2f}，总毁伤价值: {total_damage_value:.2f}")
    print(f"费效比验证: {total_damage_value / total_cost:.4f}")


# 运行15次独立测试
if __name__ == "__main__":
    main()
    # run_multiple_tests(test_times=15)