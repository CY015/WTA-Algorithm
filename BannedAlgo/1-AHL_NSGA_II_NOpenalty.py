import numpy as np
import random
import time
from typing import List, Tuple, Dict


# 目标类定义
class Target:
    def __init__(self, target_id: int, value: float, components: List[Tuple[float, float]]):
        self.id = target_id
        self.value = value
        self.components = components  # (权重, 健康度)列表

    def calculate_damage_effect(self, ammo_damage_prob: List[float]) -> float:
        """计算弹药对该目标的毁伤效能"""
        total_effect = 0.0
        for (weight, health), damage_prob in zip(self.components, ammo_damage_prob):
            total_effect += weight * damage_prob * health
        return total_effect


# 弹药类定义
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
        self.population_size = 20
        self.max_generations = 500

        # 自适应参数
        self.crossover_rate_min = 0.7
        self.crossover_rate_max = 0.9
        self.mutation_rate_min = 0.1
        self.mutation_rate_max = 0.3
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

        # 使用公式计算最少发射数量
        required = np.ceil(np.log(1 - self.damage_threshold) / np.log(1 - e_ij))
        return max(1, int(required))

    def initialize_population(self) -> List[np.ndarray]:
        """改进的初始化 - 确保满足毁伤要求和库存约束"""
        population = []
        target_count = len(self.targets)
        ammo_count = len(self.ammunitions)

        for _ in range(self.population_size):
            chromosome = np.zeros((ammo_count, target_count), dtype=int)

            # 为每个目标确保达到毁伤要求
            for j in range(target_count):
                # 选择对该目标最有效的弹药
                ammo_effects = []
                for i in range(ammo_count):
                    e_ij = self.calculate_damage_efficiency(i, j)
                    ammo_effects.append((i, e_ij))

                # 按毁伤效能排序
                ammo_effects.sort(key=lambda x: x[1], reverse=True)

                # 分配弹药直到满足毁伤要求
                current_damage = 0
                attempts = 0
                while current_damage < self.damage_threshold and attempts < ammo_count:
                    ammo_id, e_ij = ammo_effects[attempts]
                    if e_ij > 0:
                        # 计算需要多少发
                        required = self.calculate_required_rounds(ammo_id, j)
                        # 严格的库存检查
                        remaining_stock = self.ammunitions[ammo_id].stock - np.sum(chromosome[ammo_id, :])
                        allocated = min(required, remaining_stock)
                        if allocated > 0:
                            chromosome[ammo_id, j] = allocated
                            # 更新当前毁伤概率
                            current_damage = self.calculate_actual_damage(chromosome, j)
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
        """评估目标函数（费效比和总时间）"""
        total_effect = 0.0
        total_cost = 0.0

        # 计算总毁伤效能和总成本
        for j in range(len(self.targets)):
            target_damage = self.calculate_actual_damage(chromosome, j)
            # 仅计算达到阈值部分的有效毁伤
            effective_damage = min(target_damage, self.damage_threshold)
            total_effect += self.targets[j].value * effective_damage

            for i in range(len(self.ammunitions)):
                rounds = chromosome[i, j]
                total_cost += self.ammunitions[i].cost * rounds

        # 总时间计算
        total_time = np.sum(chromosome) * 0.5

        # 费效比定义为有效毁伤与成本的比值
        cost_effectiveness = total_effect / total_cost if total_cost > 0 else 0.001

        return cost_effectiveness, total_time

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

        # 使用常规目标函数评估
        objectives = [self.evaluate_objectives(ind) for ind in population]

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

        return fronts

    def crowding_distance(self, front_indices: List[int], population: List[np.ndarray]) -> List[float]:
        """计算拥挤度距离"""
        size = len(front_indices)
        if size == 0:
            return []

        distances = [0.0] * size
        # 使用常规目标函数评估
        objectives = [self.evaluate_objectives(population[i]) for i in front_indices]

        # 对每个目标函数
        for obj_idx in range(2):
            # 创建索引和对应目标值的列表
            indexed_objs = list(zip(range(size), [obj[obj_idx] for obj in objectives]))

            # 按目标值排序
            if obj_idx == 0:  # 费效比，从大到小
                indexed_objs.sort(key=lambda x: x[1], reverse=True)
            else:  # 时间，从小到大
                indexed_objs.sort(key=lambda x: x[1])

            # 边界个体距离设为无穷大
            if size > 0:
                distances[indexed_objs[0][0]] = float('inf')
            if size > 1:
                distances[indexed_objs[-1][0]] = float('inf')

            # 计算中间个体的距离
            if size > 2:
                min_val = indexed_objs[0][1]
                max_val = indexed_objs[-1][1]

                if abs(max_val - min_val) > 1e-10:
                    for i in range(1, size - 1):
                        prev_val = indexed_objs[i - 1][1]
                        next_val = indexed_objs[i + 1][1]
                        original_idx = indexed_objs[i][0]
                        distances[original_idx] += (next_val - prev_val) / (max_val - min_val)

        return distances

    def meets_damage_requirements(self, chromosome: np.ndarray) -> bool:
        """检查是否满足所有目标的毁伤要求和库存约束"""
        # 1. 检查毁伤要求
        for j in range(len(self.targets)):
            damage = self.calculate_actual_damage(chromosome, j)
            if damage < self.damage_threshold - 0.01:  # 容差
                return False

        # 2. 检查库存约束
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

    def adaptive_crossover_rate(self, generation: int) -> float:
        """自适应交叉率（随代数线性变化）"""
        progress = generation / self.max_generations
        return self.crossover_rate_max - (self.crossover_rate_max - self.crossover_rate_min) * progress

    def adaptive_mutation_rate(self, generation: int) -> float:
        """自适应变异率（随代数线性变化）"""
        progress = generation / self.max_generations
        return self.mutation_rate_max - (self.mutation_rate_max - self.mutation_rate_min) * progress

    def adaptive_local_search_rate(self, generation: int) -> float:
        """自适应局部搜索率（随代数线性变化）"""
        progress = generation / self.max_generations
        return self.local_search_rate_max - (self.local_search_rate_max - self.local_search_rate_min) * progress

    def tournament_selection(self, population: List[np.ndarray]) -> np.ndarray:
        """锦标赛选择"""
        tournament_size = 3
        contestants = random.sample(population, tournament_size)
        # 选择锦标赛中目标更优的个体
        return max(contestants, key=lambda ind: self.evaluate_objectives(ind)[0])

    def adaptive_crossover(self, parent1: np.ndarray, parent2: np.ndarray, rate: float) -> Tuple[
        np.ndarray, np.ndarray]:
        """自适应交叉操作"""
        if random.random() < rate:
            # 执行均匀交叉
            mask = np.random.rand(*parent1.shape) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2
        return parent1.copy(), parent2.copy()

    def adaptive_mutation(self, chromosome: np.ndarray, rate: float) -> np.ndarray:
        """自适应变异操作"""
        mutated = chromosome.copy()
        for i in range(mutated.shape[0]):
            for j in range(mutated.shape[1]):
                if random.random() < rate:
                    # 小幅度调整弹药数量
                    change = random.choice([-1, 1])
                    new_value = max(0, mutated[i, j] + change)
                    mutated[i, j] = new_value
        return mutated

    def heuristic_local_search(self, chromosome: np.ndarray) -> np.ndarray:
        """启发式局部搜索"""
        improved = chromosome.copy()
        # 尝试优化每个目标的弹药分配
        for j in range(improved.shape[1]):
            current_damage = self.calculate_actual_damage(improved, j)
            if current_damage >= self.damage_threshold:
                # 尝试减少低效弹药
                for i in range(improved.shape[0]):
                    if improved[i, j] > 0:
                        # 临时减少1发
                        improved[i, j] -= 1
                        new_damage = self.calculate_actual_damage(improved, j)
                        if new_damage < self.damage_threshold - 0.01:
                            # 减少后不满足要求，恢复
                            improved[i, j] += 1
        return improved

    def solve(self) -> Tuple[List[Tuple[np.ndarray, Tuple[float, float], int, float]], int]:
        """求解函数 - 补全NSGA-II拥挤度选择逻辑，新增收敛代数记录"""
        population = self.initialize_population()
        all_feasible_with_metadata = []

        # 收敛监控参数
        convergence_threshold = 5e-4  # 费效比变化阈值
        convergence_window = 10  # 连续稳定代数
        best_ce_history = []  # 记录每代最优费效比
        convergence_generation = None  # 最终收敛代数
        stable_count = 0  # 当前连续稳定代数计数器

        for generation in range(self.max_generations):
            current_time = time.time()
            crossover_rate = self.adaptive_crossover_rate(generation)
            mutation_rate = self.adaptive_mutation_rate(generation)
            local_search_rate = self.adaptive_local_search_rate(generation)

            # 生成子代
            offspring = []
            while len(offspring) < self.population_size:
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

            # 计算当前代最优费效比（使用常规评估）
            current_best_ce = max([self.evaluate_objectives(ind)[0] for ind in new_population])
            best_ce_history.append(current_best_ce)

            # 收敛检测逻辑
            if len(best_ce_history) >= convergence_window:
                # 取最近N代的最优费效比
                recent_best = best_ce_history[-convergence_window:]
                # 计算最近N代的最大波动幅度
                max_fluctuation = max(recent_best) - min(recent_best)

                if max_fluctuation < convergence_threshold:
                    stable_count += 1
                    # 连续N代稳定，判定为收敛
                    if stable_count >= convergence_window:
                        convergence_generation = generation
                        print(
                            f"\n 算法在第 {generation} 代收敛（连续 {convergence_window} 代费效比波动小于 {convergence_threshold}）")
                        break  # 可提前停止迭代
                else:
                    stable_count = 0  # 波动超过阈值，重置稳定计数器

            # 筛选可行解并记录元数据
            feasible_in_new = []
            for sol in new_population:
                if self.meets_damage_requirements(sol):
                    feasible_in_new.append(sol)
                    objectives = self.evaluate_objectives(sol)
                    all_feasible_with_metadata.append((sol, objectives, generation, current_time))

            population = new_population

        # 若迭代结束仍未收敛，收敛代数设为最大迭代次数
        if convergence_generation is None:
            convergence_generation = self.max_generations
            print(f"\n  算法在最大迭代次数（{self.max_generations}代）内未完全收敛")

        # 去重处理
        unique_solutions_with_metadata = self.remove_duplicate_solutions_with_metadata(all_feasible_with_metadata)
        print(f"\n去重前: {len(all_feasible_with_metadata)} 个解")
        print(f"去重后: {len(unique_solutions_with_metadata)} 个唯一解")

        # 返回帕累托解和收敛代数
        return unique_solutions_with_metadata, convergence_generation

    def solutions_are_similar(self, sol1: np.ndarray, sol2: np.ndarray, tolerance: float = 0.01) -> bool:
        """判断两个解是否相似（目标函数值相近）"""
        obj1 = self.evaluate_objectives(sol1)
        obj2 = self.evaluate_objectives(sol2)

        # 检查费效比和总时间是否相近
        ce_similar = abs(obj1[0] - obj2[0]) / max(abs(obj1[0]), abs(obj2[0]), 1e-10) < tolerance
        time_similar = abs(obj1[1] - obj2[1]) / max(abs(obj1[1]), abs(obj2[1]), 1e-10) < tolerance

        return ce_similar and time_similar

    def remove_duplicate_solutions_with_metadata(self,
                                                 solutions: List[Tuple[np.ndarray, Tuple[float, float], int, float]]) -> \
    List[Tuple[np.ndarray, Tuple[float, float], int, float]]:
        """移除重复和相似的解（带元数据）"""
        unique_solutions = []
        seen_hashes = set()

        for solution, objectives, generation, timestamp in solutions:
            # 创建解的哈希值（基于分配矩阵）
            solution_hash = hash(solution.tobytes())

            # 如果哈希值没出现过，且与已有解不相似，则添加
            if solution_hash not in seen_hashes:
                is_duplicate = False
                for existing_sol, _, _, _ in unique_solutions:
                    if self.solutions_are_similar(solution, existing_sol):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_solutions.append((solution, objectives, generation, timestamp))
                    seen_hashes.add(solution_hash)

        return unique_solutions


def run_multiple_tests(test_times: int = 15):
    # 初始化统计容器
    stats = {
        "best_ce_list": [],  # 每次测试的最优费效比
        "avg_ce_list": [],  # 每次测试的平均费效比（去重后）
        "feasible_ratio_list": [],  # 每次测试的达标解比例
        "convergence_gen_list": [],  # 每次测试的收敛代数
        "total_time_list": [],  # 每次测试的总求解时间
        "unique_solutions_num_list": [],  # 每次测试去重后解的数量
        "crowding_distance_avg_list": []  # 每次测试的拥挤度均值
    }

    # 2. 新增：全局最优解跟踪变量
    global_best_ce = -float('inf')
    global_best_solution = None  # 最优解的分配矩阵
    global_best_metadata = None  # (目标值, 生成代数, 测试序号, 相对时间)

    # 测试参数
    targets = [
        Target(1, 100.0, [(0.3, 1.0), (0.4, 1.0), (0.3, 1.0)]),
        Target(2, 80.0, [(0.5, 1.0), (0.5, 1.0)]),
        Target(3, 120.0, [(0.2, 1.0), (0.3, 1.0), (0.3, 1.0), (0.2, 1.0)]),
        Target(4, 100.0, [(0.3, 1.0), (0.4, 1.0), (0.3, 1.0)]),
        Target(5, 80.0, [(0.5, 1.0), (0.5, 1.0)]),
        Target(6, 120.0, [(0.2, 1.0), (0.3, 1.0), (0.3, 1.0), (0.2, 1.0)])
    ]

    ammunitions = [
        Ammunition(1, 10.0, 30, {
            1: [0.6, 0.4, 0.5], 2: [0.7, 0.5], 3: [0.5, 0.6, 0.4, 0.3],
            4: [0.6, 0.4, 0.5], 5: [0.7, 0.5], 6: [0.5, 0.6, 0.4, 0.3]
        }),
        Ammunition(2, 15.0, 30, {
            1: [0.8, 0.6, 0.7], 2: [0.6, 0.7], 3: [0.7, 0.5, 0.6, 0.5],
            4: [0.6, 0.4, 0.5], 5: [0.7, 0.5], 6: [0.5, 0.6, 0.4, 0.3]
        }),
        Ammunition(3, 25.0, 30, {
            1: [0.9, 0.8, 0.85], 2: [0.85, 0.9], 3: [0.8, 0.7, 0.75, 0.8],
            4: [0.6, 0.4, 0.5], 5: [0.7, 0.5], 6: [0.5, 0.6, 0.4, 0.3]
        }),
        Ammunition(4, 20.0, 30, {
            1: [0.8, 0.5, 0.7], 2: [0.6, 0.7], 3: [0.7, 0.5, 0.6, 0.5],
            4: [0.6, 0.4, 0.5], 5: [0.7, 0.5], 6: [0.5, 0.6, 0.7, 0.3]
        }),
        Ammunition(5, 5.0, 30, {
            1: [0.9, 0.8, 0.85], 2: [0.85, 0.9], 3: [0.8, 0.7, 0.55, 0.8],
            4: [0.6, 0.2, 0.4], 5: [0.8, 0.5], 6: [0.5, 0.6, 0.4, 0.3]
        })
    ]

    adaptability_matrix = np.array([
        [0.8, 0.7, 0.6, 0.8, 0.7, 0.6],
        [0.9, 0.8, 0.7, 0.9, 0.8, 0.7],
        [0.7, 0.9, 0.8, 0.7, 0.9, 0.8],
        [0.8, 0.6, 0.5, 0.4, 0.7, 0.6],
        [0.9, 0.9, 0.8, 0.5, 0.6, 0.6]
    ])
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
        feasible_ratio = 1.0
        unique_num = len(pareto_solutions)

        # 计算拥挤度均值
        front_indices = list(range(len(pareto_solutions)))
        crowding_distances = solver.crowding_distance(front_indices, [sol for sol, _, _, _ in pareto_solutions])
        crowding_avg = np.mean([d for d in crowding_distances if not np.isinf(d)])

        # 更新统计容器
        stats["best_ce_list"].append(current_best_ce)
        stats["avg_ce_list"].append(avg_ce)
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
            print(f"第 {test_idx + 1} 次测试刷新全局最优费效比: {global_best_ce:.4f}")
        else:
            print(f"本次测试最优费效比: {current_best_obj[0]:.4f}（未超过全局最优）")

        # 输出当前测试统计
        print(f"第 {test_idx + 1} 次测试统计：")
        print(f"  最优费效比: {current_best_ce:.4f}, 平均费效比: {avg_ce:.4f}")
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
        final_stats[key] = f"{mean_val:.4f} ± {std_val:.4f}"

    # 统计所有测试中出现的最优解
    all_best_ces = stats["best_ce_list"]
    global_max_ce = max(all_best_ces) if all_best_ces else None
    print(f"所有测试中的全局最大费效比: {global_max_ce:.4f}")

    print(f"测试次数: {len(stats['best_ce_list'])} 次（有效测试）")
    print(f"最优费效比: {final_stats['best_ce_list']}")
    print(f"平均费效比: {final_stats['avg_ce_list']}")
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
    # main()
    run_multiple_tests(test_times=15)