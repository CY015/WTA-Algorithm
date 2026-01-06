#
# AL-NSGA-II.py
# 有自适应、没有启发式算子
#
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random, time


class Target:
    """目标类"""

    def __init__(self, target_id: int, value: float, components: List[Tuple[float, float]]):
        self.id = target_id
        self.value = value
        self.components = components

    def calculate_damage_effect(self, ammo_damage_prob: List[float]) -> float:
        """计算弹药对该目标的毁伤效能"""
        total_effect = 0.0
        for (weight, health), damage_prob in zip(self.components, ammo_damage_prob):
            total_effect += weight * damage_prob * health
        return total_effect


class Ammunition:
    """弹药类"""

    def __init__(self, ammo_id: int, cost: float, stock: int, damage_profiles: Dict[int, List[float]]):
        self.id = ammo_id
        self.cost = cost
        self.stock = stock
        self.damage_profiles = damage_profiles


class StaticWeaponTargetAssignment:
    """静态弹目匹配模型 - 含自适应交叉变异（无启发式局部搜索，用于对比实验）"""

    def __init__(self, targets: List[Target], ammunitions: List[Ammunition],
                 adaptability_matrix: np.ndarray, damage_threshold: float):
        self.targets = targets
        self.ammunitions = ammunitions
        self.adaptability_matrix = adaptability_matrix
        self.damage_threshold = damage_threshold
        self.population_size = 20
        self.max_generations = 100

        # 自适应参数
        self.crossover_rate_min = 0.6
        self.crossover_rate_max = 0.9
        self.mutation_rate_min = 0.05
        self.mutation_rate_max = 0.15
        # 移除局部搜索相关参数（不再使用）

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

        required = np.ceil(np.log(1 - self.damage_threshold) / np.log(1 - e_ij))
        return max(1, int(required))

    def initialize_population(self) -> List[np.ndarray]:
        """改进的初始化 - 确保满足毁伤要求"""
        population = []
        target_count = len(self.targets)
        ammo_count = len(self.ammunitions)

        for _ in range(self.population_size):
            chromosome = np.zeros((ammo_count, target_count), dtype=int)

            # 为每个目标确保达到毁伤要求
            for j in range(target_count):
                ammo_effects = []
                for i in range(ammo_count):
                    e_ij = self.calculate_damage_efficiency(i, j)
                    ammo_effects.append((i, e_ij))

                ammo_effects.sort(key=lambda x: x[1], reverse=True)

                current_damage = 0
                attempts = 0
                while current_damage < self.damage_threshold and attempts < ammo_count:
                    ammo_id, e_ij = ammo_effects[attempts]
                    if e_ij > 0:
                        required = self.calculate_required_rounds(ammo_id, j)
                        max_possible = min(required, self.ammunitions[ammo_id].stock // target_count)
                        allocated = max(1, max_possible)
                        chromosome[ammo_id, j] = allocated
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
        """评估目标函数"""
        ce, time, _ = self.evaluate_objectives_with_penalty(chromosome)
        return ce, time

    def evaluate_objectives_with_penalty(self, chromosome: np.ndarray) -> Tuple[float, float, float]:
        """带惩罚项的目标函数评估（保留过量毁伤惩罚）"""
        total_effect = 0.0
        total_cost = 0.0
        penalty = 0.0

        for j in range(len(self.targets)):
            target_damage = self.calculate_actual_damage(chromosome, j)
            target_value = self.targets[j].value

            # 未达阈值惩罚
            if target_damage < self.damage_threshold:
                penalty += (self.damage_threshold - target_damage) * target_value * 10
            # 过量毁伤惩罚
            elif target_damage > self.damage_threshold + 0.1:
                excess = target_damage - (self.damage_threshold + 0.1)
                penalty += excess * target_value * 5

            # 有效毁伤（超过阈值部分不计入）
            effective_damage = min(target_damage, self.damage_threshold + 0.1)
            total_effect += target_value * effective_damage

            # 总成本
            for i in range(len(self.ammunitions)):
                rounds = chromosome[i, j]
                total_cost += self.ammunitions[i].cost * rounds

        total_time = np.sum(chromosome) * 0.5
        effective_effect = total_effect - penalty
        cost_effectiveness = effective_effect / total_cost if total_cost > 0 else 0.001

        return cost_effectiveness, total_time, penalty

    def dominates(self, obj1: Tuple[float, float], obj2: Tuple[float, float]) -> bool:
        """判断解1是否支配解2"""
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
        objectives = [self.evaluate_objectives(population[i]) for i in front_indices]

        for obj_idx in range(2):
            indexed_objs = list(zip(range(size), [obj[obj_idx] for obj in objectives]))

            if obj_idx == 0:
                indexed_objs.sort(key=lambda x: x[1], reverse=True)
            else:
                indexed_objs.sort(key=lambda x: x[1])

            if size > 0:
                distances[indexed_objs[0][0]] = float('inf')
            if size > 1:
                distances[indexed_objs[-1][0]] = float('inf')

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
        """检查是否满足所有目标的毁伤要求"""
        for j in range(len(self.targets)):
            damage = self.calculate_actual_damage(chromosome, j)
            if damage < self.damage_threshold - 0.01:
                return False
        return True

    def repair_solution(self, chromosome: np.ndarray) -> np.ndarray:
        """修复解（仅补充至刚好满足阈值）"""
        repaired = chromosome.copy()

        for j in range(len(self.targets)):
            current_damage = self.calculate_actual_damage(repaired, j)

            while current_damage < self.damage_threshold:
                best_ammo = -1
                best_effect = 0
                for i in range(len(self.ammunitions)):
                    e_ij = self.calculate_damage_efficiency(i, j)
                    total_used = np.sum(repaired[i, :])
                    if e_ij > best_effect and total_used < self.ammunitions[i].stock:
                        best_effect = e_ij
                        best_ammo = i

                if best_ammo >= 0:
                    repaired[best_ammo, j] += 1
                    current_damage = self.calculate_actual_damage(repaired, j)
                else:
                    break

        return repaired

    def solve(self) -> List[Tuple[np.ndarray, Tuple[float, float], int, float]]:
        """求解函数 - 保留自适应交叉变异，移除启发式局部搜索"""
        population = self.initialize_population()
        all_feasible_with_metadata = []

        for generation in range(self.max_generations):
            current_time = time.time()
            crossover_rate = self.adaptive_crossover_rate(generation)
            mutation_rate = self.adaptive_mutation_rate(generation)

            # 生成子代（移除局部搜索步骤）
            offspring = []
            while len(offspring) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)

                # 自适应交叉（保留）
                child1, child2 = self.adaptive_crossover(parent1, parent2, crossover_rate)

                # 自适应变异（保留）
                child1 = self.adaptive_mutation(child1, mutation_rate)
                child2 = self.adaptive_mutation(child2, mutation_rate)

                # 【移除】启发式局部搜索步骤

                # 修复解（保留）
                child1 = self.repair_solution(child1)
                child2 = self.repair_solution(child2)

                offspring.extend([child1, child2])

            # 合并种群
            combined = population + offspring
            fronts = self.fast_non_dominated_sort(combined)

            # 按NSGA-II规则选择新种群（保留）
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

            # 记录可行解元数据
            feasible_in_new = []
            for sol in new_population:
                if self.meets_damage_requirements(sol):
                    feasible_in_new.append(sol)
                    objectives = self.evaluate_objectives(sol)
                    all_feasible_with_metadata.append((sol, objectives, generation, current_time))

            # 输出进度
            if generation % 20 == 0:
                best_ce = max([self.evaluate_objectives(ind)[0] for ind in new_population])
                feasible_count = len(feasible_in_new)
                print(f"Generation {generation}: ")
                print(f"  Best CE = {best_ce:.4f}, Feasible = {feasible_count}/{len(new_population)}")
                print(f"  Adaptive Rates: Crossover={crossover_rate:.3f}, Mutation={mutation_rate:.3f}")

            population = new_population

        # 去重处理
        unique_solutions_with_metadata = self.remove_duplicate_solutions_with_metadata(all_feasible_with_metadata)

        print(f"\n去重前: {len(all_feasible_with_metadata)} 个解")
        print(f"去重后: {len(unique_solutions_with_metadata)} 个唯一解")

        return unique_solutions_with_metadata

    def solutions_are_similar(self, sol1: np.ndarray, sol2: np.ndarray, tolerance: float = 0.01) -> bool:
        """判断两个解是否相似"""
        obj1 = self.evaluate_objectives(sol1)
        obj2 = self.evaluate_objectives(sol2)

        ce_similar = abs(obj1[0] - obj2[0]) / max(abs(obj1[0]), abs(obj2[0]), 1e-10) < tolerance
        time_similar = abs(obj1[1] - obj2[1]) / max(abs(obj1[1]), abs(obj2[1]), 1e-10) < tolerance

        return ce_similar and time_similar

    def remove_duplicate_solutions_with_metadata(self,
                                                 solutions: List[Tuple[np.ndarray, Tuple[float, float], int, float]]) -> \
    List[Tuple[np.ndarray, Tuple[float, float], int, float]]:
        """严格去重 - 保留元数据"""
        seen_patterns = {}

        for solution, objectives, generation, timestamp in solutions:
            pattern_parts = []
            for j in range(len(self.targets)):
                target_allocation = []
                for i in range(len(self.ammunitions)):
                    if solution[i, j] > 0:
                        target_allocation.append(f"{i}_{solution[i, j]}")
                pattern_parts.append(",".join(sorted(target_allocation)))
            pattern = "|".join(pattern_parts)

            if pattern not in seen_patterns or generation < seen_patterns[pattern][2]:
                seen_patterns[pattern] = (solution, objectives, generation, timestamp)

        return list(seen_patterns.values())

    def adaptive_crossover_rate(self, generation: int) -> float:
        """自适应交叉概率"""
        progress = generation / self.max_generations
        return self.crossover_rate_max - (self.crossover_rate_max - self.crossover_rate_min) * progress

    def adaptive_mutation_rate(self, generation: int) -> float:
        """自适应变异概率"""
        progress = generation / self.max_generations
        return self.mutation_rate_max - (self.mutation_rate_max - self.mutation_rate_min) * progress

    def tournament_selection(self, population: List[np.ndarray]) -> np.ndarray:
        """锦标赛选择"""
        tournament_size = 3
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: self.evaluate_objectives(x)[0])

    def adaptive_crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                           crossover_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """自适应交叉操作（保留）"""
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        crossover_type = random.choice(['single_point', 'uniform', 'target_based'])

        if crossover_type == 'single_point':
            crossover_point = random.randint(1, len(self.targets) - 1)
            for j in range(crossover_point, len(self.targets)):
                child1[:, j] = parent2[:, j]
                child2[:, j] = parent1[:, j]

        elif crossover_type == 'uniform':
            for j in range(len(self.targets)):
                if random.random() < 0.5:
                    child1[:, j] = parent2[:, j]
                    child2[:, j] = parent1[:, j]

        elif crossover_type == 'target_based':
            for j in range(len(self.targets)):
                if random.random() < 0.5:
                    child1[:, j] = parent1[:, j]
                    child2[:, j] = parent2[:, j]
                else:
                    child1[:, j] = parent2[:, j]
                    child2[:, j] = parent1[:, j]

        return child1, child2

    def adaptive_mutation(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """自适应变异操作（保留）"""
        mutated = chromosome.copy()
        target_count = len(self.targets)
        ammo_count = len(self.ammunitions)

        for i in range(ammo_count):
            for j in range(target_count):
                if random.random() < mutation_rate:
                    mutation_type = random.choice(['perturb', 'swap', 'redistribute'])

                    if mutation_type == 'perturb':
                        change = random.choice([-2, -1, 1, 2])
                        new_value = max(0, mutated[i, j] + change)
                        total_used = np.sum(mutated[i, :]) - mutated[i, j] + new_value
                        if total_used <= self.ammunitions[i].stock:
                            mutated[i, j] = new_value

                    elif mutation_type == 'swap':
                        if mutated[i, j] > 0:
                            other_j = random.randint(0, target_count - 1)
                            if other_j != j:
                                temp = mutated[i, j]
                                mutated[i, j] = mutated[i, other_j]
                                mutated[i, other_j] = temp

                    elif mutation_type == 'redistribute':
                        if mutated[i, j] >= 2:
                            max_transfer = mutated[i, j] // 2
                            if max_transfer >= 1:
                                transfer_amount = random.randint(1, max_transfer)
                                other_j = random.randint(0, target_count - 1)
                                mutated[i, j] -= transfer_amount
                                mutated[i, other_j] += transfer_amount

        return mutated


# 示例使用
def main():
    # 测试数据
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
            1: [0.6, 0.4, 0.5],
            2: [0.7, 0.5],
            3: [0.5, 0.6, 0.4, 0.3],
            4: [0.6, 0.4, 0.5],
            5: [0.7, 0.5],
            6: [0.5, 0.6, 0.4, 0.3]
        }),
        Ammunition(2, 15.0, 30, {
            1: [0.8, 0.6, 0.7],
            2: [0.6, 0.7],
            3: [0.7, 0.5, 0.6, 0.5],
            4: [0.6, 0.4, 0.5],
            5: [0.7, 0.5],
            6: [0.5, 0.6, 0.4, 0.3]
        }),
        Ammunition(3, 25.0, 30, {
            1: [0.9, 0.8, 0.85],
            2: [0.85, 0.9],
            3: [0.8, 0.7, 0.75, 0.8],
            4: [0.6, 0.4, 0.5],
            5: [0.7, 0.5],
            6: [0.5, 0.6, 0.4, 0.3]
        }),
        Ammunition(4, 20.0, 30, {
            1: [0.8, 0.5, 0.7],
            2: [0.6, 0.7],
            3: [0.7, 0.5, 0.6, 0.5],
            4: [0.6, 0.4, 0.5],
            5: [0.7, 0.5],
            6: [0.5, 0.6, 0.7, 0.3]
        }),
        Ammunition(5, 5.0, 30, {
            1: [0.9, 0.8, 0.85],
            2: [0.85, 0.9],
            3: [0.8, 0.7, 0.55, 0.8],
            4: [0.6, 0.2, 0.4],
            5: [0.8, 0.5],
            6: [0.5, 0.6, 0.4, 0.3]
        })
    ]

    adaptability_matrix = np.array([
        [0.8, 0.7, 0.6, 0.8, 0.7, 0.6],
        [0.9, 0.8, 0.7, 0.9, 0.8, 0.7],
        [0.7, 0.9, 0.8, 0.7, 0.9, 0.8],
        [0.8, 0.6, 0.5, 0.4, 0.7, 0.6],
        [0.9, 0.9, 0.8, 0.5, 0.6, 0.6]
    ])

    solver = StaticWeaponTargetAssignment(targets, ammunitions, adaptability_matrix, 0.8)

    print("开始求解静态弹目匹配问题（无启发式局部搜索）...")
    start_time = time.time()
    pareto_solutions = solver.solve()
    total_runtime = time.time() - start_time

    print(f"\n找到 {len(pareto_solutions)} 个唯一且满足毁伤要求的帕累托最优解:")
    for i, (solution, objectives, gen, sol_timestamp) in enumerate(pareto_solutions):
        ce, time_cost = objectives
        relative_time = sol_timestamp - start_time
        print(f"\n解 {i + 1}:")
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


if __name__ == "__main__":
    main()