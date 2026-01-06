import numpy as np
import random
import time
from typing import List, Tuple, Dict

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
        """
        damage_profiles: dict mapping target.id -> list of damage probabilities per component
        """
        self.id = ammo_id
        self.cost = cost
        self.stock = stock
        self.damage_profiles = damage_profiles

class MOPSO_Solver:
    def __init__(self, targets: List[Target], ammunitions: List[Ammunition],
                 adaptability_matrix: np.ndarray, damage_threshold: float):
        self.targets = targets
        self.ammunitions = ammunitions
        self.adaptability_matrix = np.array(adaptability_matrix)
        self.damage_threshold = damage_threshold

        # PSO parameters (tunable)
        self.swarm_size = 30  # 粒子群规模（种群大小）
        self.max_iters = 500  # 最大迭代次数
        self.w = 0.7          # 惯性权重
        self.c1 = 1.5         # 认知系数（个体学习因子）
        self.c2 = 1.5         # 社会系数（群体学习因子）

        self.ammo_count = len(ammunitions)
        self.target_count = len(targets)
        self.max_alloc_per_pair = max([ammo.stock for ammo in ammunitions]) if ammunitions else 10  # 最大分配量限制

        # External archive storing (solution_matrix(int), objectives_tuple)
        self.archive: List[Tuple[np.ndarray, Tuple[float, float]]] = []
        self.archive_max_size = 100 # 外部存档最大容量

        # deterministic seeding for reproducibility (can be turned off)
        random.seed(0)
        np.random.seed(0)

    # ----------------- Problem-specific helper functions (compatible with AHL-NSGA-II) -----------------
    def calculate_damage_efficiency(self, ammo_id: int, target_id: int) -> float:
        target = self.targets[target_id]
        ammo = self.ammunitions[ammo_id]
        damage_prob = ammo.damage_profiles.get(target.id, [0.1] * len(target.components))
        adaptability = float(self.adaptability_matrix[ammo_id, target_id])
        weighted_damage = target.calculate_damage_effect(damage_prob)
        return adaptability * weighted_damage

    def calculate_actual_damage(self, chromosome: np.ndarray, target_id: int) -> float:
        """Compute combined damage probability on target j given allocation matrix."""
        survival_prob = 1.0
        for i in range(self.ammo_count):
            rounds = int(round(chromosome[i, target_id]))
            if rounds > 0:
                e_ij = self.calculate_damage_efficiency(i, target_id)
                # clamp e_ij to [0,1)
                e_ij = max(0.0, min(e_ij, 0.999999))
                survival_prob *= (1.0 - e_ij) ** rounds
        return 1.0 - survival_prob

    def evaluate_objectives_with_penalty(self, chromosome: np.ndarray) -> Tuple[float, float, float]:
        """
        Returns (cost_effectiveness, total_time, penalty)
        cost_effectiveness is primary (higher better), total_time is secondary (lower better).
        This mirrors the AHL-NSGA-II objective semantics for easier comparison.
        """
        total_effect = 0.0
        total_cost = 0.0
        penalty = 0.0
        for j in range(self.target_count):
            target_damage = self.calculate_actual_damage(chromosome, j)
            target_value = self.targets[j].value
            if target_damage < self.damage_threshold:
                penalty += (self.damage_threshold - target_damage) * target_value * 10.0
            elif target_damage > self.damage_threshold + 0.1:
                excess = target_damage - (self.damage_threshold + 0.1)
                penalty += excess * target_value * 5.0
            effective_damage = min(target_damage, self.damage_threshold + 0.1)
            total_effect += target_value * effective_damage
            for i in range(self.ammo_count):
                rounds = int(round(chromosome[i, j]))
                total_cost += self.ammunitions[i].cost * rounds
        total_time = float(np.sum(np.round(chromosome))) * 0.5  # arbitrary time per round
        effective_effect = total_effect - penalty
        cost_effectiveness = effective_effect / total_cost if total_cost > 0 else 0.001
        return cost_effectiveness, total_time, penalty

    def dominates(self, obj1: Tuple[float, float], obj2: Tuple[float, float]) -> bool:
        """Return True if obj1 dominates obj2 (obj1 better or equal in all, strictly better in at least one).
        obj tuple: (cost_effectiveness (higher better), time (lower better))
        """
        return (obj1[0] >= obj2[0] and obj1[1] <= obj2[1]) and (obj1[0] > obj2[0] or obj1[1] < obj2[1])

    # ----------------- Constraints repair and heuristics -----------------
    def repair_solution(self, matrix: np.ndarray) -> np.ndarray:
        """Repair allocation matrix to satisfy inventory constraints and attempt to meet damage thresholds greedily."""
        repaired = matrix.copy().astype(int)
        # 1) Inventory constraints (per ammunition)
        for i in range(self.ammo_count):
            total_used = int(np.sum(repaired[i, :]))
            if total_used > self.ammunitions[i].stock:
                excess = int(total_used - self.ammunitions[i].stock)
                # reduce from largest allocations first
                order = np.argsort(repaired[i, :])[::-1]
                for j in order:
                    if excess <= 0:
                        break
                    reduc = min(excess, repaired[i, j])
                    repaired[i, j] -= reduc
                    excess -= reduc
        # 2) Try to increase allocations greedily to meet damage threshold (subject to stocks)
        for j in range(self.target_count):
            curr = self.calculate_actual_damage(repaired, j)
            attempts = 0
            while curr < self.damage_threshold and attempts < 100:
                best_ammo = -1
                best_effect = 0.0
                for i in range(self.ammo_count):
                    e = self.calculate_damage_efficiency(i, j)
                    used = int(np.sum(repaired[i, :]))
                    if e > best_effect and used < self.ammunitions[i].stock:
                        best_effect = e
                        best_ammo = i
                if best_ammo >= 0:
                    repaired[best_ammo, j] += 1
                    curr = self.calculate_actual_damage(repaired, j)
                else:
                    break
                attempts += 1
        repaired[repaired < 0] = 0
        return repaired

    # ----------------- Swarm initialization and archive maintenance -----------------
    def initialize_swarm(self):
        swarm = []
        velocities = []
        pbest = []
        pbest_obj = []
        for s in range(self.swarm_size):
            mat = np.zeros((self.ammo_count, self.target_count), dtype=int)
            for j in range(self.target_count):
                n_choices = random.randint(1, min(3, max(1, self.ammo_count)))
                choices = random.sample(range(self.ammo_count), n_choices)
                for i in choices:
                    qty = random.randint(0, max(1, self.ammunitions[i].stock // max(1, self.target_count)))
                    mat[i, j] = qty
            mat = self.repair_solution(mat)
            swarm.append(mat.astype(float))
            velocities.append(np.random.uniform(-1.0, 1.0, size=(self.ammo_count, self.target_count)))
            pbest.append(mat.copy().astype(float))
            pbest_obj.append(self.evaluate_objectives_with_penalty(mat)[:2])
            self.try_add_archive(mat)
        return np.array(swarm), np.array(velocities), np.array(pbest), pbest_obj

    def try_add_archive(self, solution_matrix: np.ndarray):
        sol = solution_matrix.astype(int)
        obj = self.evaluate_objectives_with_penalty(sol)[:2]
        non_dominated = True
        to_remove = []
        for k, (arch_sol, arch_obj) in enumerate(self.archive):
            if self.dominates(arch_obj, obj):
                non_dominated = False
                break
            if self.dominates(obj, arch_obj):
                to_remove.append(k)
        if not non_dominated:
            return
        for idx in sorted(to_remove, reverse=True):
            self.archive.pop(idx)
        self.archive.append((sol.copy(), obj))
        if len(self.archive) > self.archive_max_size:
            self.crowding_trim_archive()

    def crowding_trim_archive(self):
        size = len(self.archive)
        if size <= self.archive_max_size:
            return
        objs = [entry[1] for entry in self.archive]
        ce_list = [o[0] for o in objs]
        time_list = [o[1] for o in objs]
        distances = [0.0] * size
        idxs = list(range(size))
        sorted_ce = sorted(idxs, key=lambda k: ce_list[k], reverse=True)
        sorted_time = sorted(idxs, key=lambda k: time_list[k])
        distances[sorted_ce[0]] = float('inf')
        distances[sorted_ce[-1]] = float('inf')
        distances[sorted_time[0]] = float('inf')
        distances[sorted_time[-1]] = float('inf')

        def accumulate(sorted_list, values):
            minv = values[sorted_list[0]]
            maxv = values[sorted_list[-1]]
            if abs(maxv - minv) < 1e-10:
                return
            for i in range(1, len(sorted_list) - 1):
                prevv = values[sorted_list[i - 1]]
                nextv = values[sorted_list[i + 1]]
                distances[sorted_list[i]] += (nextv - prevv) / (maxv - minv)

        accumulate(sorted_ce, ce_list)
        accumulate(sorted_time, time_list)
        order = sorted(range(size), key=lambda k: distances[k])
        remove_count = size - self.archive_max_size
        for idx in sorted(order[:remove_count], reverse=True):
            self.archive.pop(idx)

    def select_leader(self):
        if not self.archive:
            return np.zeros((self.ammo_count, self.target_count)).astype(float)
        objs = [entry[1] for entry in self.archive]
        ce_list = np.array([o[0] for o in objs])
        time_list = np.array([o[1] for o in objs])
        ce_norm = (ce_list - ce_list.min()) / (ce_list.max() - ce_list.min() + 1e-10)
        time_norm = (time_list - time_list.min()) / (time_list.max() - time_list.min() + 1e-10)
        sparsity = 1.0 / (1.0 + ce_norm + time_norm)
        probs = sparsity / sparsity.sum()
        idx = np.random.choice(range(len(self.archive)), p=probs)
        leader = self.archive[idx][0]
        return leader.astype(float)

    def mutate_particle(self, position: np.ndarray, iter_idx: int) -> np.ndarray:
        pos = position.copy()
        scale = max(1.0, (1.0 - iter_idx / max(1, self.max_iters)) * 3.0)
        i = random.randrange(self.ammo_count)
        j = random.randrange(self.target_count)
        change = random.randint(-2, 2)
        pos[i, j] = max(0, int(round(pos[i, j] + change * scale)))
        return pos

    # ----------------- Core solver -----------------
    def solve(self):
        """
        Enhanced MOPSO solve() – compatible with MOPSO_Grok-style logging.
        --------------------------
        Returns:
            archive: [
                (solution_matrix, (ce, time), generation, timestamp)
            ]
            total_iterations
        """
        swarm, velocities, pbest, pbest_obj = self.initialize_swarm()

        # Internal archive stored as: (matrix, objectives, generation, timestamp)
        new_archive = []

        start_time = time.time()

        for it in range(self.max_iters):
            for s in range(self.swarm_size):

                # choose leader
                leader = self.select_leader()

                r1 = np.random.rand(self.ammo_count, self.target_count)
                r2 = np.random.rand(self.ammo_count, self.target_count)

                cognitive = self.c1 * r1 * (pbest[s] - swarm[s])
                social = self.c2 * r2 * (leader - swarm[s])

                velocities[s] = self.w * velocities[s] + cognitive + social
                velocities[s] = np.clip(velocities[s], -self.max_alloc_per_pair, self.max_alloc_per_pair)

                swarm[s] = swarm[s] + velocities[s]
                swarm[s] = np.clip(np.round(swarm[s]), 0, self.max_alloc_per_pair).astype(float)

                # mutation
                if random.random() < 0.05:
                    swarm[s] = self.mutate_particle(swarm[s], it)

                repaired = self.repair_solution(np.round(swarm[s]).astype(int))
                swarm[s] = repaired.astype(float)

                # evaluate
                cur_obj = self.evaluate_objectives_with_penalty(repaired)[:2]
                prev = pbest_obj[s]

                if self.dominates(cur_obj, prev) or (cur_obj[0] > prev[0]):
                    pbest[s] = repaired.astype(float)
                    pbest_obj[s] = cur_obj

                # add to enhanced archive
                timestamp = time.time() - start_time
                new_archive.append((
                    repaired.copy(),
                    cur_obj,
                    it,
                    timestamp
                ))

                # also maintain legacy archive (for swarm leader selection)
                self.try_add_archive(repaired)

        # remove duplicates in MOPSO-style archive
        filtered = []
        seen = set()
        for sol, obj, gen, ts in new_archive:
            key = sol.tobytes()
            if key not in seen:
                seen.add(key)
                filtered.append((sol, obj, gen, ts))

        return filtered, self.max_iters

# ----------------- Multi-run testing function -----------------
# def run_multiple_tests(num_runs: int,
#                        targets: List[Target],
#                        ammunitions: List[Ammunition],
#                        adaptability_matrix: np.ndarray,
#                        damage_threshold: float,
#                        verbose: bool = True):
#     pareto_counts = []
#     best_ce_list = []
#     best_time_list = []
#     converge_iters = []
#
#     for run in range(num_runs):
#         if verbose:
#             print(f"--- MOPSO run {run+1}/{num_runs} ---")
#         solver = MOPSO_Solver(targets, ammunitions, adaptability_matrix, damage_threshold)
#         pareto_front, iters = solver.solve()
#         converge_iters.append(iters)
#         pareto_counts.append(len(pareto_front))
#         if pareto_front:
#             objs = [obj for (_, obj) in pareto_front]
#             ce_values = [o[0] for o in objs]
#             time_values = [o[1] for o in objs]
#             best_idx = int(np.argmax(ce_values))
#             best_ce_list.append(float(ce_values[best_idx]))
#             best_time_list.append(float(time_values[best_idx]))
#         else:
#             best_ce_list.append(0.0)
#             best_time_list.append(0.0)
#
#     import numpy as _np
#     result = {
#         "num_runs": num_runs,
#         "avg_best_ce": float(_np.mean(best_ce_list)),
#         "avg_best_time": float(_np.mean(best_time_list)),
#         "avg_pareto_size": float(_np.mean(pareto_counts)),
#         "avg_iters": float(_np.mean(converge_iters)),
#         "details": {
#             "best_ce_list": best_ce_list,
#             "best_time_list": best_time_list,
#             "pareto_counts": pareto_counts,
#             "iters_list": converge_iters
#         }
#     }
#     return result

# ----------------- Example main (safe default) -----------------
def main():
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


    # 测试参数(8*9)
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
    
    damage_threshold = 0.8

    print("Demo: Running 15 short MOPSO runs for quick verification (this is lightweight).")
    res = run_multiple_tests(num_runs=15, targets=targets, ammunitions=ammunitions,
                             adaptability_matrix=adaptability_matrix, damage_threshold=damage_threshold, verbose=True)
    print("\\nMulti-run summary:")
    for k, v in res.items():
        if k != "details":
            print(f"{k}: {v}")


def run_multiple_tests(num_runs,
                       targets,
                       ammunitions,
                       adaptability_matrix,
                       damage_threshold,
                       verbose=True):

    all_best_ce = []
    all_avg_ce = []
    all_converge_gens = []
    all_times = []
    all_unique_counts = []
    # 新增这两行（与上面保持对齐）
    all_best_task_time = []    # 每次测试中 Pareto 解里最小的任务时间
    all_avg_task_time = []     # 每次测试中 Pareto 解的任务时间平均值

    global_best_ce = -1e9
    global_best_solution = None
    global_best_meta = None  # (obj, gen, ts, run_idx)

    print(f"\n开始 {num_runs} 次独立测试（含全局最优解跟踪）...\n")

    for run in range(1, num_runs + 1):
        print(f"=== 第 {run}/{num_runs} 次测试 ===\n")

        solver = MOPSO_Solver(targets, ammunitions, adaptability_matrix, damage_threshold)

        start_time = time.time()
        archive, converge_gen = solver.solve()
        total_time = time.time() - start_time

        # 记录统计
        raw_count = len(archive)
        unique_archive = []
        seen = set()
        for sol, obj, gen, ts in archive:
            key = sol.tobytes()
            if key not in seen:
                seen.add(key)
                unique_archive.append((sol, obj, gen, ts))

        unique_count = len(unique_archive)

        print(f"算法在第 {converge_gen} 代收敛")
        print(f"去重前: {raw_count} 个解")
        print(f"去重后: {unique_count} 个唯一解")

        if unique_count == 0:
            print(f"第 {run} 次测试无有效解（跳过）\n")
            continue

        # 提取 CE 列表
        ce_list = [obj[0] for (_, obj, _, _) in unique_archive]
        avg_ce = float(np.mean(ce_list))
        best_idx = int(np.argmax(ce_list))

        best_sol, best_obj, best_gen, best_ts = unique_archive[best_idx]
        best_ce = best_obj[0]
        best_time = best_obj[1]

        # 新增：统计任务时间（目标2：最小化）
        task_time_list = [obj[1] for (_, obj, _, _) in unique_archive]
        current_best_task_time = float(np.min(task_time_list))
        current_avg_task_time = float(np.mean(task_time_list))

        # 记录到全局统计列表
        all_best_task_time.append(current_best_task_time)
        all_avg_task_time.append(current_avg_task_time)

        # 是否刷新全局最优
        if best_ce > global_best_ce:
            print(f"第 {run} 次测试刷新全局最优费效比: {best_ce:.4f}")
            global_best_ce = best_ce
            global_best_solution = best_sol.copy()
            global_best_meta = (best_obj, best_gen, best_ts, run)
        else:
            print(f"本次测试最优费效比: {best_ce:.4f}（未超过全局最优）")

        print(f"第 {run} 次测试统计：")
        print(f"  最优费效比: {best_ce:.4f}, 平均费效比: {avg_ce:.4f}")
        print(f"  收敛代数: {converge_gen}, 总时间: {total_time:.3f} 秒\n")

        # 记录
        all_best_ce.append(best_ce)
        all_avg_ce.append(avg_ce)
        all_converge_gens.append(converge_gen)
        all_times.append(total_time)
        all_unique_counts.append(unique_count)

    # ========== 全局统计 ==========
    print("==================================================")
    print("多次独立测试统计报告")
    print("==================================================")

    if len(all_best_ce) == 0:
        print("无有效测试结果。")
        return

    ce_mean = float(np.mean(all_best_ce))
    ce_std = float(np.std(all_best_ce))
    avg_ce_mean = float(np.mean(all_avg_ce))
    avg_ce_std = float(np.std(all_avg_ce))
    gen_mean = float(np.mean(all_converge_gens))
    gen_std = float(np.std(all_converge_gens))
    time_mean = float(np.mean(all_times))
    time_std = float(np.std(all_times))
    uniq_mean = float(np.mean(all_unique_counts))
    uniq_std = float(np.std(all_unique_counts))
    best_task_mean = float(np.mean(all_best_task_time))
    best_task_std = float(np.std(all_best_task_time))
    avg_task_mean = float(np.mean(all_avg_task_time))
    avg_task_std = float(np.std(all_avg_task_time))

    print(f"测试次数: {len(all_best_ce)} 次（有效测试）")
    print(f"最优费效比: {ce_mean:.4f} ± {ce_std:.4f}")
    print(f"平均费效比: {avg_ce_mean:.4f} ± {avg_ce_std:.4f}")
    print(f"最优任务时间: {best_task_mean:.2f} ± {best_task_std:.2f}")
    print(f"平均任务时间: {avg_task_mean:.2f} ± {avg_task_std:.2f}")
    print(f"收敛代数: {gen_mean:.4f} ± {gen_std:.4f}")
    print(f"平均求解时间: {time_mean:.4f} ± {time_std:.4f} 秒")
    print(f"去重后解数量: {uniq_mean:.2f} ± {uniq_std:.2f}\n")

    # ========== 全局最优解输出 ==========
    print("============================================================")
    print(f"全局最优费效比解（最高值: {global_best_ce:.4f}）")
    print("============================================================")

    best_obj, best_gen, best_ts, best_run = global_best_meta
    print(f"测试来源: 第 {best_run} 次测试")
    print(f"生成代数: 第 {best_gen} 代")
    print(f"生成时间: {best_ts:.3f} 秒")
    print(f"费效比: {best_obj[0]:.4f}，任务时间: {best_obj[1]:.2f}\n")

    print("详细分配方案:")
    solution = global_best_solution
    total_rounds = 0
    total_cost = 0.0
    total_effect_value = 0.0

    for j in range(len(targets)):
        col = solution[:, j]
        rounds = int(np.sum(col))
        dmg = solver.calculate_actual_damage(solution, j)
        value = targets[j].value * min(dmg, damage_threshold + 0.1)

        total_rounds += rounds
        total_effect_value += value

        print(f"目标{j+1}:")
        print(f"  总弹药: {rounds}发，毁伤概率: {dmg:.4f}（{'✓ 达标' if dmg>=damage_threshold else '× 未达标'}）")

        # 逐弹药输出
        details = []
        for i in range(len(ammunitions)):
            qty = int(solution[i, j])
            if qty > 0:
                cost = qty * ammunitions[i].cost
                total_cost += cost
                details.append(f"弹药{i+1}: {qty}发（成本: {cost:.2f}）")

        if details:
            print("  分配详情: " + ", ".join(details))
        else:
            print("  分配详情: 未分配弹药")
        print()

    print("全局统计:")
    print(f"总弹药消耗: {total_rounds}发")
    print(f"总成本: {total_cost:.2f}，总毁伤价值: {total_effect_value:.2f}")
    print(f"费效比验证: {total_effect_value / total_cost:.4f}")

    print("\n============================================================\n")

    return {
        "best_ce_list": all_best_ce,
        "avg_ce_list": all_avg_ce,
        "gen_list": all_converge_gens,
        "time_list": all_times,
        "uniq_list": all_unique_counts,
        "global_best": {
            "solution": global_best_solution,
            "meta": global_best_meta
        }
    }


if __name__ == "__main__":
    main()
