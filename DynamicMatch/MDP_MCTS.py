# -*- coding: utf-8 -*-

import numpy as np
import math
import random
import time
from typing import List, Tuple, Dict, Any, Optional
from StaticMatch.AHL_NSGA_II import AHLNSGAII_Solver, Target, Ammunition


# ==================== 先定义两个会在内部使用的类（避免前向引用）====================
class MCTSState:
    def __init__(self, target_damage: List[float], ammo_stocks: List[int], time_step: int):
        self.target_damage = target_damage      # 活跃目标当前毁伤度
        self.ammo_stocks = ammo_stocks          # 所有弹药库存
        self.time_step = time_step              # 时间步

    def is_terminal(self) -> bool:
        return all(d >= 0.8 for d in self.target_damage)


class MCTSNode:
    def __init__(self, state: MCTSState, parent=None, action=None, outer=None):
        self.state = state
        self.parent = parent
        self.action = action                    # type: Optional[np.ndarray]
        self.children: List['MCTSNode'] = []
        self.visit_count = 0
        self.total_reward = 0.0
        self.untried_actions: List[np.ndarray] = []
        self.outer = outer                      # 持有外部 OfflineDynamicTargetWeaponAssignment 实例

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c: float = 1.41) -> 'MCTSNode':
        # UCT
        choices = []
        for child in self.children:
            if child.visit_count == 0:
                choices.append(float('inf'))
            else:
                exploit = child.total_reward / child.visit_count
                explore = c * math.sqrt(math.log(self.visit_count) / child.visit_count)
                choices.append(exploit + explore)
        return self.children[np.argmax(choices)]

    def expand(self, action: np.ndarray, next_state: MCTSState) -> 'MCTSNode':
        child = MCTSNode(next_state, self, action, self.outer)
        self.children.append(child)
        # 移除已扩展的动作
        self.untried_actions = [a for a in self.untried_actions if not np.array_equal(a, action)]
        return child


# ==================== 主类 ====================
class OfflineDynamicTargetWeaponAssignment:
    def __init__(self, initial_targets: List[Target], initial_ammos: List[Ammunition],
                 adaptability_matrix: np.ndarray, target_add_sequence: Dict[float, List[Target]],
                 ammo_supply_sequence: Dict[float, List[Tuple[int, int]]] = None,
                 damage_threshold: float = 0.8, max_decision_time: float = 5.0,
                 discount_factor: float = 0.95,
                 lambda_: float = 15.0, mu_: float = 0.08, eta_: float = 0.8):

        self.initial_timestamp = time.time()
        self.current_timestamp = self.initial_timestamp

        # 重要：把所有序列的 key 都转成绝对时间（而不是相对时间）
        self.simulation_time = 0.0  # 新增：仿真时钟
        self.decision_interval = 5.0  # 每轮决策间隔（原来MCTS思考5秒）

        # 直接使用相对时间，不再加 time.time()
        self.target_add_sequence = sorted(target_add_sequence.items())  # key 是相对时间
        self.ammo_supply_sequence = sorted((ammo_supply_sequence or {}).items())
        self.next_target_idx = self.next_ammo_idx = 0

        # self.target_add_sequence = sorted(target_add_sequence.items())
        # self.ammo_supply_sequence = sorted((ammo_supply_sequence or {}).items())
        # self.next_target_idx = self.next_ammo_idx = 0

        self.targets = {t.id: t for t in initial_targets}
        self.target_damage = {tid: 0.0 for tid in self.targets}

        self.ammos = {a.id: a for a in initial_ammos}
        self.ammo_stock = {aid: a.stock for aid, a in self.ammos.items()}

        # --- [新增] 用于记录每个目标消耗的弹药 {target_id: {ammo_id: count}} ---
        self.history_consumption = {}

        self.adaptability_matrix = adaptability_matrix
        self.damage_threshold = damage_threshold
        self.max_decision_time = max_decision_time
        self.discount_factor = discount_factor
        self.lambda_ = lambda_
        self.mu_ = mu_
        self.eta_ = eta_
        self.exploration_constant = 1.41 # 根号2

        self.decision_times = []

    # ==================== 2. 打印详细分配信息（满足需求1）====================
    def _print_action_detail(self, action: np.ndarray):
        """打印本轮详细弹药分配情况"""
        active_targets = self._get_active_targets()
        ammo_list = self._get_ammo_list()

        print("  本轮弹药分配详情：")
        total_rounds = 0
        for i, ammo in enumerate(ammo_list):
            for j, target in enumerate(active_targets):
                cnt = int(action[i, j])
                if cnt > 0:
                    print(f"    → 弹药{ammo.id} ({ammo.cost}万元/发) × {cnt} → 目标{target.id} ({target.value}价值)")
                    total_rounds += cnt
        if total_rounds == 0:
            print("    （本轮无分配）")
        else:
            print(f"    本轮共发射 {total_rounds} 发")

    # ------------------ 辅助 ------------------
    def _get_active_targets(self) -> List[Target]:
        return [self.targets[tid] for tid, d in self.target_damage.items() if d < self.damage_threshold]

    def _get_ammo_list(self) -> List[Ammunition]:
        return [self.ammos[aid] for aid in sorted(self.ammos.keys())]

    def _single_hit_prob(self, ammo: Ammunition, target: Target, target_id: int) -> float:
        col = target_id - 1
        adapt = (self.adaptability_matrix[ammo.id-1, col]
                 if col < self.adaptability_matrix.shape[1] else 0.5)
        profile = ammo.damage_profiles.get(target_id, [0.1] * len(target.components))
        total_w = sum(w for w, _ in target.components)
        p = sum((w/total_w) * pc for (w, _), pc in zip(target.components, profile))
        return adapt * p

    # 正态分布
    def _sample_damage(self, rounds: int) -> float:
        mean = 0.45 * rounds
        std = max(0.05, 0.12 * rounds)
        dmg = np.random.normal(mean, std)
        return max(0.05, min(1.0, dmg))
    
    # 根据弹药数量动态调节 Beta 参数，实现“多打多准”的饱和效应
    # def _sample_damage(self, rounds: int) -> float:
    #     if rounds == 1:
    #         a, b = 2.0, 5.0    # 期望 0.286
    #     elif rounds <= 3:
    #         a, b = 2.8, 4.5    # 期望 0.38
    #     elif rounds <= 6:
    #         a, b = 3.8, 4.0    # 期望 0.487
    #     elif rounds <= 10:
    #         a, b = 5.0, 3.5    # 期望 0.588
    #     else:
    #         a, b = 6.0, 2.5    # 期望 0.706，趋于饱和       
    #     return np.random.beta(a, b)

    # ------------------ 随机转移 ------------------
    def _transition(self, state: MCTSState, action: np.ndarray) -> MCTSState:
        active_ids = [tid for tid, d in self.target_damage.items() if d < self.damage_threshold]
        ammo_list = self._get_ammo_list()

        new_damage = state.target_damage.copy()
        new_stock = state.ammo_stocks.copy()

        # 消耗弹药
        for i, ammo in enumerate(ammo_list):
            used = int(action[i].sum())
            new_stock[i] = max(0, new_stock[i] - used)

        # 每个目标随机毁伤
        for j, tid in enumerate(active_ids):
            rounds = sum(int(action[i, j]) for i in range(len(ammo_list)))
            if rounds == 0: continue

            hit_prob = 0.0
            for i, ammo in enumerate(ammo_list):
                r = int(action[i, j])
                if r == 0: continue
                p = self._single_hit_prob(ammo, self.targets[tid], tid)
                hit_prob += 1 - (1 - p)**r
            hit_prob = min(1.0, hit_prob)

            delta = self._sample_damage(rounds) if random.random() < hit_prob else 0.0
            new_damage[j] = min(1.0, new_damage[j] + delta)

        return MCTSState(new_damage, new_stock, state.time_step + 1)

    # ------------------ 完整奖励函数 ------------------
    def _reward(self, state: MCTSState, action: np.ndarray, next_state: MCTSState) -> float:
        active_ids = [tid for tid, d in self.target_damage.items() if d < self.damage_threshold]
        ammo_list = self._get_ammo_list()

        # 毁伤奖励
        R_dmg = 0.0
        for j, tid in enumerate(active_ids):
            delta = next_state.target_damage[j] - state.target_damage[j]
            if delta > 0:
                remaining = max(1e-6, self.damage_threshold - state.target_damage[j])
                R_dmg += self.targets[tid].value * (delta / remaining)

        # 成本惩罚
        R_cost = sum(int(action[i].sum()) * ammo_list[i].cost for i in range(len(ammo_list)))

        # 时间惩罚
        R_time = 1.0

        return self.lambda_ * R_dmg - self.mu_ * R_cost - self.eta_ * R_time

    # ------------------ 费效比 rollout ------------------
    def _rollout_policy(self, state: MCTSState) -> np.ndarray:
        n_ammo = len(self._get_ammo_list())
        n_target = len(state.target_damage)
        action = np.zeros((n_ammo, n_target), dtype=int)
        stock = state.ammo_stocks.copy()
        active_targets = self._get_active_targets()

        values = [self.targets[t.id].value * max(0, 0.8 - state.target_damage[i])
                  for i, t in enumerate(active_targets)]

        for j in np.argsort(values)[::-1]:
            if values[j] <= 0: continue
            target = active_targets[j]
            best_ratio = best_i = -1
            for i, ammo in enumerate(self._get_ammo_list()):
                if stock[i] <= 0: continue
                p = self._single_hit_prob(ammo, target, target.id)
                ratio = p / max(1.0, ammo.cost)
                if ratio > best_ratio:
                    best_ratio, best_i = ratio, i
            if best_i >= 0:
                amt = min(2, stock[best_i])
                action[best_i, j] = amt
                stock[best_i] -= amt
        return action

    def _simulate(self, node: MCTSNode) -> float:
        state = node.state
        total = 0.0
        depth = 0
        max_depth = 60
        while not state.is_terminal() and depth < max_depth and any(s > 0 for s in state.ammo_stocks):
            action = self._rollout_policy(state)
            next_state = self._transition(state, action)
            r = self._reward(state, action, next_state)
            total += (self.discount_factor ** depth) * r
            state = next_state
            depth += 1
        return total

    # ------------------ MCTS 四大步骤 ------------------
    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.exploration_constant)
        return node

    def _backpropagate(self, node: MCTSNode, reward: float):
        while node:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    def _generate_candidate_actions(self) -> List[np.ndarray]:
        active = self._get_active_targets()
        ammo = self._get_ammo_list()
        if not active:
            return [np.zeros((len(ammo), 0), dtype=int)]

        # 更新 ammo.stock 为当前真实库存
        for a in ammo:
            a.stock = self.ammo_stock[a.id]

        ids = [t.id for t in active]
        cols = [max(0, tid-1) for tid in ids]
        sub_mat = self.adaptability_matrix[:, cols]

        try:
            solver = AHLNSGAII_Solver(active, ammo, sub_mat, self.damage_threshold)
            pareto, _ = solver.solve()
            actions = [sol[0] for sol in pareto if sol[0].shape == (len(ammo), len(active))]
            return actions if actions else self._random_actions(10)
        except Exception as e:
            print("AHL-NSGA-II 失败，改用随机动作:", e)
            return self._random_actions(10)

    def _random_actions(self, n: int = 10) -> List[np.ndarray]:
        active = self._get_active_targets()
        ammo = self._get_ammo_list()
        res = []
        for _ in range(n):
            act = np.zeros((len(ammo), len(active)), dtype=int)
            for j in range(len(active)):
                avail = [i for i, a in enumerate(ammo) if self.ammo_stock[a.id] > 0]
                if avail:
                    i = random.choice(avail)
                    amt = random.randint(1, min(2, self.ammo_stock[ammo[i].id]))
                    act[i, j] = amt
            res.append(act)
        return res

    def _create_initial_state(self) -> MCTSState:
        active_ids = sorted([tid for tid, d in self.target_damage.items() if d < self.damage_threshold])
        dmg = [self.target_damage[tid] for tid in active_ids]
        stock = [self.ammo_stock[aid] for aid in sorted(self.ammos.keys())]
        step = int(time.time() - self.initial_timestamp)
        return MCTSState(dmg, stock, step)

    # ------------------ 主搜索 ------------------
    def mcts_search(self) -> np.ndarray:
        start = time.time()
        root_state = self._create_initial_state()
        root = MCTSNode(root_state, outer=self)

        root.untried_actions = self._generate_candidate_actions()

        iter_count = 0
        while time.time() - start < self.max_decision_time:
            iter_count += 1
            node = self._select(root)

            if not node.is_fully_expanded() and not node.state.is_terminal():
                act = random.choice(node.untried_actions)
                next_s = self._transition(node.state, act)
                node = node.expand(act, next_s)

            reward = self._simulate(node)
            self._backpropagate(node, reward)

        # 选访问次数最多的孩子
        if not root.children:
            best_action = np.zeros((len(self._get_ammo_list()), len(self._get_active_targets())), dtype=int)
        else:
            best_child = max(root.children, key=lambda c: c.visit_count)
            best_action = best_child.action

        elapsed = time.time() - start
        self.decision_times.append(elapsed)
        print(f"MCTS决策完成 | 迭代 {iter_count} 次 | 耗时 {elapsed:.2f}s | 活跃目标 {len(self._get_active_targets())}")
        return best_action

    # ------------------ 态势更新 ------------------
    def _update_situation(self) -> bool:
        changed = False
        now = self.simulation_time

        while (self.next_target_idx < len(self.target_add_sequence) and
               now >= self.target_add_sequence[self.next_target_idx][0]):
            _, new_targets = self.target_add_sequence[self.next_target_idx]
            for t in new_targets:
                if t.id not in self.targets:
                    self.targets[t.id] = t
                    self.target_damage[t.id] = 0.0
                    print(f"[{now:.0f}s] 新目标出现 → 目标{t.id} ({t.value}价值)")
                    changed = True
            self.next_target_idx += 1

        while (self.next_ammo_idx < len(self.ammo_supply_sequence) and
               now >= self.ammo_supply_sequence[self.next_ammo_idx][0]):
            _, supplies = self.ammo_supply_sequence[self.next_ammo_idx]
            for aid, cnt in supplies:
                self.ammo_stock[aid] += cnt
                ammo = self.ammos[aid]
                print(f"[{now:.0f}s] 弹药补充 → 弹药{aid} +{cnt}发 (现库存{self.ammo_stock[aid]})")
                changed = True
            self.next_ammo_idx += 1

        return changed

    # ------------------ 主循环 ------------------
    def run_offline_simulation(self, max_time: float = 400.0):
        print("MDP-MCTS 动态弹目匹配完整版启动".center(80, "="))

        while self.simulation_time < max_time:
            print(f"\n{'=' * 20} [仿真时间 {self.simulation_time:.0f}s] {'=' * 20}")

            # 1. 态势更新
            changed = self._update_situation()
            if changed:
                active = self._get_active_targets()
                print(f"   → 当前活跃目标: {[t.id for t in active]}")

            # 2. MCTS决策
            action = self.mcts_search()

            # 3. 执行分配 + 输出
            if action.size > 0 and action.shape[1] > 0:
                self._print_action_detail(action)
                self._execute_action(action)
                print(f"[{self.simulation_time:.0f}s] 火力打击完成，总发射 {action.sum()} 发")
            else:
                print(f"[{self.simulation_time:.0f}s] 无需打击")

            # 4. 检查是否全部摧毁 (修正版：必须同时满足“无活跃目标”且“无待新增目标”)
            all_current_dead = (len(self._get_active_targets()) == 0)
            no_future_targets = (self.next_target_idx >= len(self.target_add_sequence))

            if all_current_dead:
                if no_future_targets:
                    print(f"\n所有波次目标均已摧毁！总耗时 {self.simulation_time:.1f}秒")
                    break
                else:
                    # 如果当前没目标，但未来还有，就仅打印日志，不退出
                    print(f"[{self.simulation_time:.0f}s] 当前波次已肃清，等待下一波目标出现...")

            # 5. 推进仿真时间（关键！）
            self.simulation_time += self.decision_interval  # 每轮前进5秒

            # 可选：真实等待一点点，防止CPU 100%
            time.sleep(0.01)

        # 循环结束后调用报告
        report_data = self._print_final_report()
        return report_data # 确保这里返回报告数据

    # 新增：真正执行动作 + 随机毁伤结算
    def _execute_action(self, action: np.ndarray):
        """把 MCTS 决策的 action 真正应用到真实战场（和 _transition 完全一致的随机过程）"""
        active_ids = [tid for tid, d in self.target_damage.items() if d < self.damage_threshold]
        ammo_list = self._get_ammo_list()

        # 1. 消耗弹药（确定性）
        for i, ammo in enumerate(ammo_list):
            used = int(action[i].sum())
            if used > 0:
                self.ammo_stock[ammo.id] = max(0, self.ammo_stock[ammo.id] - used)

        # 2. 对每个活跃目标进行随机毁伤
        for j, tid in enumerate(active_ids):
            rounds = sum(int(action[i, j]) for i in range(len(ammo_list)))
            if rounds == 0: continue

            # --- [新增] 记录该目标本轮消耗 ---
            if tid not in self.history_consumption:
                self.history_consumption[tid] = {}

            target = self.targets[tid]
            hit_prob = 0.0
            for i, ammo in enumerate(ammo_list):
                r = int(action[i, j])
                if r == 0: continue

                # 记录明细
                old_count = self.history_consumption[tid].get(ammo.id, 0)
                self.history_consumption[tid][ammo.id] = old_count + r

                p = self._single_hit_prob(ammo, target, tid)
                hit_prob += 1 - (1 - p) ** r

            hit_prob = min(1.0, hit_prob)

            delta = 0.0
            if random.random() < hit_prob:
                delta = self._sample_damage(rounds)

            old_dmg = self.target_damage[tid]
            self.target_damage[tid] = min(1.0, old_dmg + delta)

            if delta > 1e-6:
                print(f"  → 目标{tid} 本轮毁伤增量 +{delta:.3f} (当前 {old_dmg:.3f}→{self.target_damage[tid]:.3f})")

        # 在 _execute_action 最后加上已摧毁目标的提示
        if self.target_damage[tid] >= self.damage_threshold and old_dmg < self.damage_threshold:
            print(f"  → 目标{tid} 被成功摧毁！")

    # 新增：最终报告
    def _print_final_report(self):
        print("\n" + "-"*80)
        print("仿真结束 - 最终战果报告"+"\n")
        # print("-"*80)

        destroyed = sum(1 for d in self.target_damage.values() if d >= 0.8)
        total = len(self.targets)
        print(f"目标摧毁情况: {destroyed}/{total}  ({destroyed/total*100:.1f}%)")

        grand_total_cost = 0.0
        grand_total_rounds = 0

        # 按ID排序输出每个目标的情况
        for tid in sorted(self.targets.keys()):
            dmg = self.target_damage.get(tid, 0.0)
            status = "摧毁" if dmg >= 0.8 else f"残余 {dmg:.1%}"
            target_val = self.targets[tid].value

            print(f"\n🎯 目标{tid} (价值 {target_val}): {dmg:.1%} → {status}")

            # 输出该目标的弹药消耗明细
            if tid in self.history_consumption:
                t_cost = 0.0
                t_rounds = 0
                details = []
                # 按弹药ID排序
                for aid in sorted(self.history_consumption[tid].keys()):
                    count = self.history_consumption[tid][aid]
                    cost = count * self.ammos[aid].cost
                    details.append(f"弹药{aid}×{count}")
                    t_cost += cost
                    t_rounds += count

                print(f"   - 消耗明细: {', '.join(details)}")
                print(f"   - 单目标成本: {t_cost:.1f} 万元")

                grand_total_cost += t_cost
                grand_total_rounds += t_rounds
            else:
                print(f"   - 消耗明细: 无打击记录")

        # 计算总毁伤价值
        total_value_gained = sum(t.value for tid, t in self.targets.items()
                                         if self.target_damage.get(tid, 0) >= 0.8)
        print("-" * 80)
        print(f"总弹药消耗量: {grand_total_rounds} 发")
        print(f"总弹药消耗成本: {grand_total_cost:.1f} 万元")
        print(f"累计毁伤价值(仅计摧毁): {total_value_gained:.1f}")

        ratio = total_value_gained / grand_total_cost if grand_total_cost > 0 else 0
        print(f"最终费效比: {ratio:.3f}")
        print(f"平均决策时间: {np.mean(self.decision_times):.3f}s × {len(self.decision_times)}次")
        
        # --- [新增] 返回关键指标，供多轮测试收集 ---
        return {
            'total_rounds': grand_total_rounds,
            'total_cost': grand_total_cost,
            'total_value': total_value_gained,
            'final_ratio': ratio,
            'simulation_time': self.simulation_time
        }

def create_test_data():
    """论文级动态弹目匹配标准测试场景（9目标+8种弹药+分批出现+弹药补充）"""

    # 测试参数(9个目标)
    targets = [
            Target(1, 80.0, [(0.5, 1.0)]),  # 人员集群
            Target(2, 120.0, [(0.2, 1.0), (0.2, 1.0), (0.2, 1.0), (0.5, 1.0), (0.5, 1.0),
                              (0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
                              (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
                              (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
                              (0.8, 1.0), (0.8, 1.0)]),  # 地下指挥所
            Target(3, 100.0, [(0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0),
                              (0.8, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
                              (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
                              (0.8, 1.0)]),  # 陆基雷达站
            Target(4, 110.0, [(0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 1.0), (0.2, 1.0),
                              (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.5, 1.0),
                              (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.2, 1.0), (0.8, 1.0),
                              (0.5, 1.0)]),  # 机场
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
                              (0.8, 1.0), (0.8, 1.0)]),  # 阵地(雷达车15个，电源车13个，导弹发射车18个，指挥控制车17个)
            Target(6, 80.0, [(0.5, 1.0)]),  # 人员集群
            Target(7, 100.0, [(0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0),
                              (0.8, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
                              (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
                              (0.8, 1.0)]),  # 陆基雷达站
            Target(8, 100.0, [(0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.5, 1.0), (0.8, 1.0),
                              (0.8, 1.0), (0.5, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0),
                              (0.8, 1.0), (0.5, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
                              (0.8, 1.0)]),  # 陆基雷达站
            Target(9, 105.0, [(0.2, 1.0), (0.2, 1.0), (0.2, 1.0), (0.8, 1.0), (0.8, 1.0),
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
                              (0.8, 1.0), (0.8, 1.0)]),  # 阵地(雷达车15个，电源车13个，导弹发射车18个，指挥控制车17个)
        ]

    ammunitions = [
            Ammunition(1, 6.0, 10, {
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
                6: [0.8],
                7: [0.5, 0.2, 0.6, 0.7, 0.8,
                    0.2, 0.2, 0.2, 0.2, 0.2,
                    0.2, 0.2, 0.2, 0.2, 0.2,
                    0.2],
                8: [0.5, 0.2, 0.6, 0.7, 0.8,
                    0.2, 0.2, 0.2, 0.2, 0.2,
                    0.2, 0.2, 0.2, 0.2, 0.2,
                    0.2],
                9: [0.7, 0.7, 0.7, 0.8, 0.8,
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
            }),  # 杀爆1(当量大)
            Ammunition(2, 4.0, 10, {
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
                6: [0.7],
                7: [0.4, 0.2, 0.5, 0.6, 0.7,
                    0.2, 0.2, 0.2, 0.2, 0.2,
                    0.2, 0.2, 0.2, 0.2, 0.2,
                    0.2],
                8: [0.4, 0.2, 0.5, 0.6, 0.7,
                    0.2, 0.2, 0.2, 0.2, 0.2,
                    0.2, 0.2, 0.2, 0.2, 0.2,
                    0.2],
                9: [0.6, 0.6, 0.6, 0.7, 0.7,
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
            }),  # 杀爆2(当量小)
            Ammunition(3, 5.0, 10, {
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
                6: [0.1],
                7: [0.7, 0.7, 0.3, 0.3, 0.5,
                    0.7, 0.7, 0.7, 0.7, 0.7,
                    0.7, 0.7, 0.7, 0.7, 0.7,
                    0.7],
                8: [0.7, 0.7, 0.3, 0.3, 0.5,
                    0.7, 0.7, 0.7, 0.7, 0.7,
                    0.7, 0.7, 0.7, 0.7, 0.7,
                    0.7],
                9: [0.1, 0.1, 0.1, 0.1, 0.1,
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
            }),  # 侵爆1(1.8m)
            Ammunition(4, 7.0, 10, {
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
                6: [0.1],
                7: [0.75, 0.75, 0.35, 0.35, 0.55,
                    0.75, 0.75, 0.75, 0.75, 0.75,
                    0.75, 0.75, 0.75, 0.75, 0.75,
                    0.75],
                8: [0.75, 0.75, 0.35, 0.35, 0.55,
                    0.75, 0.75, 0.75, 0.75, 0.75,
                    0.75, 0.75, 0.75, 0.75, 0.75,
                    0.75],
                9: [0.1, 0.1, 0.1, 0.1, 0.1,
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
            }),  # 侵爆2(6m)
            Ammunition(5, 10.0, 10, {
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
                6: [0.2],
                7: [0.8, 0.8, 0.4, 0.4, 0.6,
                    0.8, 0.8, 0.8, 0.8, 0.8,
                    0.8, 0.8, 0.8, 0.8, 0.8,
                    0.8],
                8: [0.8, 0.8, 0.4, 0.4, 0.6,
                    0.8, 0.8, 0.8, 0.8, 0.8,
                    0.8, 0.8, 0.8, 0.8, 0.8,
                    0.8],
                9: [0.15, 0.15, 0.15, 0.15, 0.15,
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
            }),  # 侵爆3(61m)
            Ammunition(6, 7.0, 10, {
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
                6: [0.9],
                7: [0.55, 0.25, 0.65, 0.75, 0.85,
                    0.25, 0.25, 0.25, 0.25, 0.25,
                    0.25, 0.25, 0.25, 0.25, 0.25,
                    0.25],
                8: [0.55, 0.25, 0.65, 0.75, 0.85,
                    0.25, 0.25, 0.25, 0.25, 0.25,
                    0.25, 0.25, 0.25, 0.25, 0.25,
                    0.25],
                9: [0.75, 0.75, 0.75, 0.85, 0.85,
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
            }),  # 子母1
            Ammunition(7, 3.0, 10, {
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
                6: [0.1],
                7: [0.1, 0.1, 0.1, 0.3, 0.3,
                    0.4, 0.1, 0.1, 0.1, 0.1,
                    0.1, 0.1, 0.1, 0.1, 0.1,
                    0.1],
                8: [0.1, 0.1, 0.1, 0.3, 0.3,
                    0.4, 0.1, 0.1, 0.1, 0.1,
                    0.1, 0.1, 0.1, 0.1, 0.1,
                    0.1],
                9: [0.6, 0.6, 0.6, 0.7, 0.7,
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
            }),  # 聚能1(1.3m破甲)
            Ammunition(8, 2.0, 10, {
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
                6: [0.1],
                7: [0.1, 0.1, 0.1, 0.2, 0.2,
                    0.3, 0.1, 0.1, 0.1, 0.1,
                    0.1, 0.1, 0.1, 0.1, 0.1,
                    0.1],
                8: [0.1, 0.1, 0.1, 0.2, 0.2,
                    0.3, 0.1, 0.1, 0.1, 0.1,
                    0.1, 0.1, 0.1, 0.1, 0.1,
                    0.1],
                9: [0.6, 0.6, 0.6, 0.7, 0.7,
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
            })  # 聚能2(1.1m破甲)
        ]

    adaptability_matrix = np.array([
            [0.9, 0.25, 0.45, 0.25, 0.8, 0.9, 0.45, 0.45, 0.8],  # 杀爆1
            [0.8, 0.2, 0.4, 0.2, 0.7, 0.8, 0.4, 0.4, 0.7],  # 杀爆2
            [0.1, 0.7, 0.75, 0.75, 0.1, 0.1, 0.75, 0.75, 0.1],  # 侵爆1
            [0.1, 0.8, 0.8, 0.8, 0.1, 0.1, 0.8, 0.8, 0.1],  # 侵爆2
            [0.1, 0.9, 0.85, 0.85, 0.1, 0.1, 0.85, 0.85, 0.1],  # 侵爆3
            [0.9, 0.3, 0.5, 0.6, 0.9, 0.9, 0.5, 0.5, 0.9],  # 子母1
            [0.1, 0.2, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.7],  # 聚能1
            [0.1, 0.15, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.7],  # 聚能2
        ])
        # ==================== 4. 分三波出现（动态性拉满！） ====================
    now = 0.0
    target_add_sequence = {
        now + 0: [targets[0], targets[1], targets[2]],  # 第1波：人员集群 + 指挥所 + 雷达
        now + 80: [targets[3], targets[4]],  # 第2波：机场 + 导弹阵地（最硬！）
        now + 180: [targets[5], targets[6], targets[7], targets[8]],  # 第3波：增援4个
    }

        # ==================== 5. 两次弹药补充 ====================
    ammo_supply_sequence = {
        now + 120: [(1, 8), (2, 8), (6, 6)],  # 第一波补充：杀爆 + 子母
        now + 240: [(3, 6), (4, 6), (5, 4)],  # 第二波补充：侵爆弹（对付工事）
    }

    # 初始只放第1波，其余在序列中
    initial_targets = target_add_sequence[now + 0]

    return initial_targets, ammunitions, adaptability_matrix, target_add_sequence, ammo_supply_sequence

# initial_targets, initial_ammos, adapt_matrix, target_add_seq, ammo_supply_seq = create_test_data()

# solver = OfflineDynamicTargetWeaponAssignment(
#     initial_targets=initial_targets,
#     initial_ammos=initial_ammos,
#     adaptability_matrix=adapt_matrix,
#     target_add_sequence=target_add_seq,
#     ammo_supply_sequence=ammo_supply_seq,
#     max_decision_time=5.0,
#     discount_factor=0.95,
#     lambda_=15.0, mu_=0.08, eta_=0.8
# )
# solver.run_offline_simulation(500)

# ... (保持原来的 create_test_data 函数不变) ...

def run_multiple_tests(num_runs: int = 10, max_time: float = 400.0):
    """
    执行多轮仿真测试，并统计平均性能指标。
    """
    
    print("=" * 60)
    print(f"🚀 开始多轮动态弹目匹配测试 (总轮数: {num_runs})")
    print("=" * 60)
    
    # 存储每次运行的结果
    results = []
    
    # 定义需要平均的指标
    metrics = {
        'total_rounds': [],
        'total_cost': [],
        'total_value': [],
        'final_ratio': [],
        'total_time': [],
        'total_decisions': [],
    }

    for run_id in range(1, num_runs + 1):
        # 1. 每次循环都重新生成数据，确保状态独立
        initial_targets, initial_ammos, adapt_matrix, target_add_sequence, ammo_supply_sequence = create_test_data()

        # 2. 初始化动态匹配模型
        solver = OfflineDynamicTargetWeaponAssignment(
            initial_targets=initial_targets,
            initial_ammos=initial_ammos,
            adaptability_matrix=adapt_matrix,
            target_add_sequence=target_add_sequence,
            ammo_supply_sequence=ammo_supply_sequence,
            max_decision_time=5.0,
            discount_factor=0.9,
            lambda_=10.0, mu_=0.1, eta_=5.0
        )

        print(f"\n--- 第 {run_id}/{num_runs} 轮仿真开始 ---")
        
        # 3. 运行仿真，并接收返回的关键指标
        # 注意: 我们需要修改 _print_final_report 确保它返回指标
        report_data = solver.run_offline_simulation(max_time=max_time)
        
        if report_data:
            metrics['total_rounds'].append(report_data['total_rounds'])
            metrics['total_cost'].append(report_data['total_cost'])
            metrics['total_value'].append(report_data['total_value'])
            metrics['final_ratio'].append(report_data['final_ratio'])
            metrics['total_time'].append(report_data['simulation_time'])
            metrics['total_decisions'].append(len(solver.decision_times))
            results.append(report_data)

    # 4. 输出平均性能报告
    print("\n" + "=" * 60)
    print("✨ 最终多轮测试平均性能指标报告 ✨")
    print("=" * 60)
    
    if not results:
        print("未成功执行任何仿真。")
        return

    # 辅助函数：计算平均值并格式化输出
    def print_metric(label, key, unit=""):
        if metrics[key]:
            avg_value = np.mean(metrics[key])
            std_dev = np.std(metrics[key])
            print(f"{label}: {avg_value:.3f} {unit} (标准差: {std_dev:.3f})")
        
    print(f"总运行轮数: {len(results)}/{num_runs}")
    print("-" * 60)
    print_metric("平均仿真总时长", 'total_time', "秒")
    print_metric("平均决策次数", 'total_decisions', "次")
    print("-" * 60)
    
    # 用户要求的核心指标
    print_metric("平均总弹药消耗量", 'total_rounds', "发")
    print_metric("平均总弹药消耗成本", 'total_cost', "万元")
    print_metric("平均累计毁伤价值", 'total_value', "万元")
    print_metric("平均最终费效比", 'final_ratio')
    print("=" * 60)

def main():
    # 使用多轮测试函数替换单次运行
    run_multiple_tests(num_runs=1, max_time=350.0)

if __name__ == "__main__":
    main()