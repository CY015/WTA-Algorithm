import numpy as np
import math
import random
import time
from typing import List, Tuple, Dict, Callable
from StaticMatch.AHL_NSGA_II import AHLNSGAII_Solver, Target, Ammunition
from SituationReceiver import SituationReceiver  # 导入态势接收模块


class DynamicTargetWeaponAssignment:
    """仅支持目标新增的动态弹目匹配模型（在线增量态势模式）"""
    def __init__(self, initial_situation: Dict, adaptability_matrix: np.ndarray,
                 damage_threshold: float = 0.8, max_decision_time: float = 5.0):
        """
        初始化：加载初始态势，启动态势接收
        Args:
            initial_situation: 初始态势（含初始目标、初始弹药）
            adaptability_matrix: 弹药-目标适配性矩阵（shape: [弹药数, 最大目标ID]）
            damage_threshold: 目标毁伤阈值（默认0.8）
            max_decision_time: MCTS单次决策最大时间（秒，默认5）
        """
        # 1. 态势接收模块初始化
        self.situation_receiver = SituationReceiver()
        self.situation_receiver.start()

        # 2. 初始状态加载（从初始态势中提取）
        self.initial_timestamp = initial_situation["timestamp"]
        self.current_timestamp = self.initial_timestamp
        
        # 2.1 目标管理：{目标ID: Target对象}（仅新增，不删除）
        self.targets: Dict[int, Target] = {}
        for target in initial_situation["target_changes"]["add"]:
            self.targets[target.id] = target
        
        # 2.2 目标毁伤状态：{目标ID: 毁伤度}（初始0.0）
        self.target_damage: Dict[int, float] = {tid: 0.0 for tid in self.targets.keys()}
        
        # 2.3 弹药管理：{弹药ID: Ammunition对象}
        self.ammos: Dict[int, Ammunition] = {}
        for ammo in initial_situation["ammo_changes"]["initial"]:
            self.ammos[ammo.id] = ammo
        
        # 2.4 弹药库存：{弹药ID: 剩余数量}（初始为满库存）
        self.ammo_stock: Dict[int, int] = {aid: ammo.stock for aid, ammo in self.ammos.items()}

        # 3. 核心算法参数
        self.adaptability_matrix = adaptability_matrix  # 适配性矩阵（弹药数×最大目标ID）
        self.damage_threshold = damage_threshold
        self.max_decision_time = max_decision_time  # MCTS决策时间限制
        
        # MDP参数（匹配动态.docx定义）
        self.discount_factor = 0.95  # 折扣因子γ
        self.lambda_ = 10.0          # 毁伤奖励权重
        self.mu_ = 0.1               # 成本惩罚权重
        self.eta_ = 1.0              # 时间惩罚权重
        
        # MCTS参数
        self.exploration_constant = 1.41  # 探索常数（√2）
        
        # 4. 决策状态管理
        self.current_action: Optional[np.ndarray] = None  # 当前执行的弹药分配动作
        self.last_decision_timestamp = self.initial_timestamp  # 上次决策时间戳
        self.decision_history: List[Tuple[Dict, np.ndarray, float]] = []  # 决策历史记录

    # ------------------------------
    # 内部状态管理（增量更新）
    # ------------------------------
    def _update_state(self, new_situation: Dict) -> bool:
        """
        增量更新内部状态（仅处理目标新增、毁伤修正、弹药补充）
        Returns: 是否需要触发重新决策
        """
        # 1. 更新时间戳
        self.current_timestamp = new_situation["timestamp"]
        need_redecision = new_situation["need_redecision"]

        # 2. 处理目标新增
        for new_target in new_situation["target_changes"]["add"]:
            if new_target.id not in self.targets:
                self.targets[new_target.id] = new_target
                self.target_damage[new_target.id] = 0.0  # 新增目标初始毁伤度0.0
                print(f"🎯 新增目标：ID={new_target.id}，价值={new_target.value:.0f}，部件数={len(new_target.components)}")

        # 3. 处理目标毁伤度修正（如探测偏差）
        for update in new_situation["target_changes"]["update"]:
            tid = update["target_id"]
            new_damage = update["new_damage"]
            if tid in self.target_damage:
                old_damage = self.target_damage[tid]
                self.target_damage[tid] = new_damage
                print(f"🔄 修正目标{tid}毁伤度：{old_damage:.2f} → {new_damage:.2f}")

        # 4. 处理弹药补充
        for ammo_add in new_situation["ammo_changes"]["add"]:
            aid = ammo_add["ammo_id"]
            add_stock = ammo_add["add_stock"]
            if aid in self.ammo_stock:
                self.ammo_stock[aid] += add_stock
                print(f"💥 补充弹药{aid}：+{add_stock}发 → 总库存={self.ammo_stock[aid]}发")

        return need_redecision

    def _get_current_state_snapshot(self) -> Dict:
        """获取当前状态快照（用于决策历史记录）"""
        return {
            "timestamp": self.current_timestamp,
            "targets": {tid: t.value for tid, t in self.targets.items()},
            "target_damage": self.target_damage.copy(),
            "ammo_stock": self.ammo_stock.copy(),
            "active_targets": [tid for tid, d in self.target_damage.items() if d < self.damage_threshold]  # 未达毁伤阈值的目标
        }

    # ------------------------------
    # MCTS核心逻辑（适配目标新增）
    # ------------------------------
    class State:
        """MCTS状态类（适配动态目标数量）"""
        def __init__(self, target_damage: List[float], ammo_stocks: List[float], time_step: int):
            self.target_damage = target_damage  # 当前活跃目标的毁伤度列表（按ID排序）
            self.ammo_stocks = ammo_stocks      # 弹药库存列表（按ID排序）
            self.time_step = time_step          # 时间步（当前时间戳-初始时间戳）

        def __hash__(self):
            return hash((tuple(self.target_damage), tuple(self.ammo_stocks), self.time_step))

        def __eq__(self, other):
            return (tuple(self.target_damage) == tuple(other.target_damage) and
                    tuple(self.ammo_stocks) == tuple(other.ammo_stocks) and
                    self.time_step == other.time_step)

        def is_terminal(self, max_active_targets: int) -> bool:
            """终止条件：所有活跃目标已达毁伤阈值"""
            return len([d for d in self.target_damage if d < 0.8]) == 0 or max_active_targets == 0

    class MCTSNode:
        """MCTS节点类（UCT算法）"""
        def __init__(self, state: 'DynamicTargetWeaponAssignment.State', parent=None, action=None):
            self.state = state
            self.parent = parent
            self.action = action
            self.children = []
            self.visit_count = 0
            self.total_reward = 0.0
            self.untried_actions = None

        def is_fully_expanded(self) -> bool:
            return len(self.untried_actions) == 0

        def best_child(self, exploration_constant: float) -> 'DynamicTargetWeaponAssignment.MCTSNode':
            """UCT选择：平衡探索与利用"""
            best_score = -float('inf')
            best_child = None
            for child in self.children:
                if child.visit_count == 0:
                    score = float('inf')
                else:
                    exploitation = child.total_reward / child.visit_count
                    exploration = exploration_constant * math.sqrt(math.log(self.visit_count) / child.visit_count)
                    score = exploitation + exploration
                if score > best_score:
                    best_score = score
                    best_child = child
            return best_child

        def expand(self, action: np.ndarray, next_state: 'DynamicTargetWeaponAssignment.State') -> 'DynamicTargetWeaponAssignment.MCTSNode':
            """扩展节点：添加新子节点"""
            child = self.__class__(next_state, self, action)
            self.children.append(child)
            if action in self.untried_actions:
                self.untried_actions.remove(action)
            return child

    def _create_mcts_initial_state(self) -> State:
        """创建MCTS初始状态（按ID排序，仅包含未达毁伤阈值的目标）"""
        # 筛选活跃目标（未达毁伤阈值）
        active_tids = sorted([tid for tid, d in self.target_damage.items() if d < self.damage_threshold])
        # 目标毁伤度列表（按ID排序）
        target_damage = [self.target_damage[tid] for tid in active_tids]
        # 弹药库存列表（按ID排序）
        ammo_stocks = [self.ammo_stock[aid] for aid in sorted(self.ammos.keys())]
        # 时间步（当前时间戳-初始时间戳，取整）
        time_step = int(self.current_timestamp - self.initial_timestamp)
        return self.State(target_damage, ammo_stocks, time_step)

    def _get_active_targets(self) -> List[Target]:
        """获取当前活跃目标（未达毁伤阈值）"""
        return [self.targets[tid] for tid in sorted(self.targets.keys()) if self.target_damage[tid] < self.damage_threshold]

    def _get_ammo_list(self) -> List[Ammunition]:
        """获取弹药列表（按ID排序）"""
        return [self.ammos[aid] for aid in sorted(self.ammos.keys())]

    def _calculate_damage_efficiency(self, ammo_idx: int, target_idx: int, active_tids: List[int]) -> float:
        """计算单发弹药毁伤效能（适配活跃目标索引）"""
        ammo = self._get_ammo_list()[ammo_idx]
        target_id = active_tids[target_idx]
        target = self.targets[target_id]
        
        # 弹药对目标的毁伤概率剖面（无则默认0.1）
        damage_prob = ammo.damage_profiles.get(target_id, [0.1] * len(target.components))
        # 适配性系数（从矩阵中提取，目标ID从1开始，矩阵列索引从0开始）
        adaptability = self.adaptability_matrix[ammo_idx, target_id - 1] if target_id - 1 < self.adaptability_matrix.shape[1] else 0.5
        
        # 加权毁伤效能计算（复用AHL-NSGA-II逻辑）
        weighted_damage = 0.0
        for (weight, health), prob in zip(target.components, damage_prob):
            weighted_damage += weight * prob * health
        return adaptability * weighted_damage

    def _generate_candidate_actions(self, active_targets: List[Target], ammo_list: List[Ammunition]) -> List[np.ndarray]:
        """调用AHL-NSGA-II生成候选动作（仅针对活跃目标）"""
        if not active_targets:
            return [np.zeros((len(ammo_list), 0), dtype=int)]
        
        # 构造临时弹药（传递当前库存）
        class TempAmmo:
            def __init__(self, ammo: Ammunition, stock: int):
                self.id = ammo.id
                self.cost = ammo.cost
                self.stock = stock
                self.damage_profiles = ammo.damage_profiles
        
        temp_ammos = [TempAmmo(ammo, self.ammo_stock[ammo.id]) for ammo in ammo_list]
        
        # 构造当前活跃目标的适配性矩阵（仅提取活跃目标对应的列）
        active_tids = [t.id for t in active_targets]
        current_adapt_matrix = self.adaptability_matrix[:, [tid - 1 for tid in active_tids]]
        
        # 调用AHL-NSGA-II生成帕累托最优动作
        try:
            static_solver = AHLNSGAII_Solver(
                targets=active_targets,
                ammunitions=temp_ammos,
                adaptability_matrix=current_adapt_matrix,
                damage_threshold=self.damage_threshold
            )
            pareto_solutions, _ = static_solver.solve()
            return [sol[0] for sol in pareto_solutions]  # 提取分配矩阵作为候选动作
        except Exception as e:
            print(f"AHL-NSGA-II调用失败，生成随机动作：{str(e)}")
            return self._generate_random_actions(active_targets, ammo_list)

    def _generate_random_actions(self, active_targets: List[Target], ammo_list: List[Ammunition], num: int = 3) -> List[np.ndarray]:
        """生成随机动作（AHL-NSGA-II失效时备用）"""
        actions = []
        ammo_count = len(ammo_list)
        target_count = len(active_targets)
        
        for _ in range(num):
            action = np.zeros((ammo_count, target_count), dtype=int)
            for j in range(target_count):
                # 仅对活跃目标分配1-2发有库存的弹药
                available_ammos = [i for i in range(ammo_count) if self.ammo_stock[ammo_list[i].id] > 0]
                if available_ammos:
                    ammo_idx = random.choice(available_ammos)
                    max_rounds = min(2, self.ammo_stock[ammo_list[ammo_idx].id])
                    action[ammo_idx, j] = random.randint(1, max_rounds)
            actions.append(action)
        return actions

    def mcts_search(self, initial_state: State) -> np.ndarray:
        """MCTS核心搜索（按时间限制终止）"""
        start_time = time.time()
        root = self.MCTSNode(initial_state)
        active_targets = self._get_active_targets()
        ammo_list = self._get_ammo_list()
        root.untried_actions = self._generate_candidate_actions(active_targets, ammo_list)

        # 搜索循环（按时间限制终止）
        while time.time() - start_time < self.max_decision_time:
            # 1. 选择阶段：从根节点向下选择
            current_node = self._mcts_select(root)
            
            # 2. 扩展阶段：若未终止且未完全扩展，生成新子节点
            if not current_node.state.is_terminal(len(active_targets)) and not current_node.is_fully_expanded():
                action = random.choice(current_node.untried_actions)
                next_state = self._mcts_transition(current_node.state, action)
                current_node = current_node.expand(action, next_state)
            
            # 3. 模拟阶段：随机rollout到终止状态
            reward = self._mcts_simulate(current_node, active_targets, ammo_list)
            
            # 4. 回溯阶段：更新节点奖励
            self._mcts_backpropagate(current_node, reward)

        # 选择访问次数最多的动作（最可靠）
        return max(root.children, key=lambda c: c.visit_count).action if root.children else np.zeros((len(ammo_list), len(active_targets)), dtype=int)

    def _mcts_select(self, root: MCTSNode) -> MCTSNode:
        """MCTS选择阶段：递归选择最优子节点"""
        current_node = root
        active_target_count = len(self._get_active_targets())
        while not current_node.state.is_terminal(active_target_count) and current_node.is_fully_expanded():
            current_node = current_node.best_child(self.exploration_constant)
        return current_node

    def _mcts_transition(self, state: State, action: np.ndarray) -> State:
        """MCTS状态转移（模拟动作执行后的状态）"""
        new_state = self.State(state.target_damage.copy(), state.ammo_stocks.copy(), state.time_step + 1)
        active_tids = sorted([tid for tid, d in self.target_damage.items() if d < self.damage_threshold])
        
        # 1. 更新弹药库存
        for i in range(len(new_state.ammo_stocks)):
            new_state.ammo_stocks[i] = max(0, new_state.ammo_stocks[i] - np.sum(action[i, :]))
        
        # 2. 更新目标毁伤度
        for j in range(len(new_state.target_damage)):
            if new_state.target_damage[j] >= self.damage_threshold:
                continue
            # 计算综合毁伤概率
            survival_prob = 1.0
            for i in range(len(action)):
                rounds = action[i, j]
                if rounds <= 0:
                    continue
                e_ij = self._calculate_damage_efficiency(i, j, active_tids)
                survival_prob *= (1 - e_ij) ** rounds
            damage_prob = 1 - survival_prob
            
            # 抽样更新毁伤度
            if random.random() < damage_prob:
                damage_increment = random.betavariate(2, 5)  # Beta分布模拟毁伤增量
                new_state.target_damage[j] = min(1.0, new_state.target_damage[j] + damage_increment)
        
        return new_state

    def _mcts_simulate(self, node: MCTSNode, active_targets: List[Target], ammo_list: List[Ammunition]) -> float:
        """MCTS模拟阶段（启发式策略：优先高费效比弹药）"""
        current_state = self.State(node.state.target_damage.copy(), node.state.ammo_stocks.copy(), node.state.time_step)
        total_reward = 0.0
        discount = 1.0
        active_tids = sorted([tid for tid, d in self.target_damage.items() if d < self.damage_threshold])

        while not current_state.is_terminal(len(active_targets)):
            # 启发式动作生成：选择费效比最高的弹药
            action = np.zeros((len(ammo_list), len(current_state.target_damage)), dtype=int)
            for j in range(len(current_state.target_damage)):
                if current_state.target_damage[j] >= self.damage_threshold:
                    continue
                # 计算各弹药费效比（毁伤效能/成本）
                ammo_efficiency = []
                for i in range(len(ammo_list)):
                    if current_state.ammo_stocks[i] <= 0:
                        continue
                    e_ij = self._calculate_damage_efficiency(i, j, active_tids)
                    cost = ammo_list[i].cost
                    efficiency = e_ij / cost if cost > 1e-6 else 0.0
                    ammo_efficiency.append((i, efficiency))
                # 选择费效比最高的弹药分配
                if ammo_efficiency:
                    best_ammo_idx = max(ammo_efficiency, key=lambda x: x[1])[0]
                    max_rounds = min(2, current_state.ammo_stocks[best_ammo_idx])
                    action[best_ammo_idx, j] = max_rounds
            
            # 状态转移
            next_state = self._mcts_transition(current_state, action)
            
            # 计算奖励
            reward = self._calculate_reward(current_state, next_state, action, active_targets, ammo_list)
            total_reward += discount * reward
            
            # 更新状态和折扣因子
            current_state = next_state
            discount *= self.discount_factor

        return total_reward

    def _calculate_reward(self, state: State, next_state: State, action: np.ndarray,
                         active_targets: List[Target], ammo_list: List[Ammunition]) -> float:
        """计算MCTS奖励（匹配动态.docx公式）"""
        # 1. 毁伤奖励（新增目标毁伤价值）
        damage_reward = 0.0
        for j in range(len(state.target_damage)):
            damage_increase = next_state.target_damage[j] - state.target_damage[j]
            damage_reward += active_targets[j].value * damage_increase
        
        # 2. 成本惩罚（弹药消耗成本）
        cost_penalty = 0.0
        for i in range(len(action)):
            rounds_used = np.sum(action[i, :])
            cost_penalty += ammo_list[i].cost * rounds_used
        
        # 3. 时间惩罚（总弹药消耗量×单位时间）
        time_penalty = np.sum(action) * 0.5  # 复用AHL-NSGA-II时间计算逻辑
        
        # 4. 综合奖励（加权和）
        return self.lambda_ * damage_reward - self.mu_ * cost_penalty - self.eta_ * time_penalty

    def _mcts_backpropagate(self, node: MCTSNode, reward: float):
        """MCTS回溯阶段：更新节点访问次数和奖励"""
        current_node = node
        while current_node is not None:
            current_node.visit_count += 1
            current_node.total_reward += reward
            current_node = current_node.parent

    # ------------------------------
    # 在线决策循环
    # ------------------------------
    def _print_current_decision(self, action: np.ndarray, active_targets: List[Target], ammo_list: List[Ammunition]):
        """打印当前决策结果（弹药分配详情）"""
        print("\n" + "=" * 60)
        print(f"当前决策（时间戳：{self.current_timestamp:.0f}）")
        print(f"活跃目标数：{len(active_targets)}，剩余弹药总量：{sum(self.ammo_stock.values())}")
        print("=" * 60)
        
        for j, target in enumerate(active_targets):
            target_id = target.id
            ammo_details = []
            total_rounds = 0
            for i, ammo in enumerate(ammo_list):
                rounds = action[i, j]
                if rounds > 0:
                    ammo_details.append(f"弹药{ammo.id}（成本{ammo.cost}）：{rounds}发")
                    total_rounds += rounds
            if ammo_details:
                print(f"🎯 目标{target_id}（价值{target.value:.0f}，当前毁伤{self.target_damage[target_id]:.2f}）：")
                print(f"   分配：共{total_rounds}发 → {', '.join(ammo_details)}")
                # 预测毁伤度（基于当前分配）
                e_ij_list = [self._calculate_damage_efficiency(i, j, [t.id for t in active_targets]) for i in range(len(ammo_list))]
                survival_prob = 1.0
                for i, rounds in enumerate(action[:, j]):
                    if rounds > 0:
                        survival_prob *= (1 - e_ij_list[i]) ** rounds
                predicted_damage = min(1.0, self.target_damage[target_id] + (1 - survival_prob))
                print(f"   预测毁伤度：{predicted_damage:.2f}（{'达标' if predicted_damage >= self.damage_threshold else '未达标'}）")
            else:
                print(f"🎯 目标{target_id}（价值{target.value:.0f}，当前毁伤{self.target_damage[target_id]:.2f}）：无弹药分配")
        print("=" * 60 + "\n")

    def _simulate_action_execution(self, step_duration: float = 1.0):
        """模拟动作执行：每间隔1秒更新一次目标毁伤度（反映时间流逝）"""
        if self.current_action is None:
            return
        
        active_targets = self._get_active_targets()
        if not active_targets:
            return
        
        ammo_list = self._get_ammo_list()
        active_tids = [t.id for t in active_targets]
        
        # 按动作分配更新毁伤度（简化为每1秒更新一次）
        for j, target_id in enumerate(active_tids):
            if self.target_damage[target_id] >= self.damage_threshold:
                continue
            
            # 计算综合毁伤概率
            survival_prob = 1.0
            for i in range(len(ammo_list)):
                rounds = self.current_action[i, j] if j < self.current_action.shape[1] else 0
                if rounds <= 0:
                    continue
                e_ij = self._calculate_damage_efficiency(i, j, active_tids)
                survival_prob *= (1 - e_ij) ** rounds
            damage_prob = 1 - survival_prob
        
            # 按时间步比例更新毁伤度（1秒占总打击时间的比例）
            if random.random() < damage_prob * (step_duration / 10.0):  # 假设10秒完成一次完整打击
                damage_increment = random.betavariate(2, 5) * (step_duration / 10.0)
                self.target_damage[target_id] = min(1.0, self.target_damage[target_id] + damage_increment)
        
        # 更新当前时间戳（模拟时间流逝）
        self.current_timestamp += step_duration

    def run_online_loop(self, task_end_condition: Callable[['DynamicTargetWeaponAssignment'], bool]):
        """
        在线决策循环：不定时接收态势，触发决策或模拟动作执行
        Args:
            task_end_condition: 任务结束条件（如“所有目标摧毁”或“手动停止”）
        """
        print("\n" + "=" * 80)
        print("动态弹目匹配在线决策循环启动（仅支持目标新增）")
        print(f"初始状态：目标{list(self.targets.keys())}，弹药{list(self.ammos.keys())}，时间戳={self.initial_timestamp:.0f}")
        print("=" * 80 + "\n")

        # 首次决策（基于初始态势）
        active_targets = self._get_active_targets()
        if active_targets:
            print("🔍 基于初始态势启动首次决策...")
            initial_mcts_state = self._create_mcts_initial_state()
            self.current_action = self.mcts_search(initial_mcts_state)
            self.decision_history.append((
                self._get_current_state_snapshot(),
                self.current_action.copy(),
                self.current_timestamp
            ))
            self._print_current_decision(self.current_action, active_targets, self._get_ammo_list())
            self.last_decision_timestamp = self.current_timestamp

        try:
            while not task_end_condition(self):
                # 1. 检查是否有新态势（超时1秒避免阻塞）
                if self.situation_receiver.wait_for_situation(timeout=1.0):
                    new_situation = self.situation_receiver.get_new_situation()
                    if new_situation:
                        # 1.1 增量更新状态
                        need_redecision = self._update_state(new_situation)
                        # 1.2 若需要，触发重新决策
                        if need_redecision:
                            active_targets = self._get_active_targets()
                            if active_targets:
                                print("🔍 态势更新，启动重新决策...")
                                mcts_state = self._create_mcts_initial_state()
                                self.current_action = self.mcts_search(mcts_state)
                                # 记录决策历史
                                self.decision_history.append((
                                    self._get_current_state_snapshot(),
                                    self.current_action.copy(),
                                    self.current_timestamp
                                ))
                                # 打印决策结果
                                self._print_current_decision(self.current_action, active_targets, self._get_ammo_list())
                                self.last_decision_timestamp = self.current_timestamp
                            else:
                                print("ℹ️  无活跃目标，无需决策")
                else:
                    # 2. 无新态势：模拟动作执行（每1秒更新一次毁伤度）
                    self._simulate_action_execution(step_duration=1.0)
                    # 2.1 定期检查动作是否过期（如30秒未更新则重新决策，避免动作失效）
                    action_valid_duration = 30.0
                    if self.current_timestamp - self.last_decision_timestamp > action_valid_duration:
                        active_targets = self._get_active_targets()
                        if active_targets:
                            print(f"🔍 动作已过期（{action_valid_duration}秒），启动重新决策...")
                            mcts_state = self._create_mcts_initial_state()
                            self.current_action = self.mcts_search(mcts_state)
                            self.decision_history.append((
                                self._get_current_state_snapshot(),
                                self.current_action.copy(),
                                self.current_timestamp
                            ))
                            self._print_current_decision(self.current_action, active_targets, self._get_ammo_list())
                            self.last_decision_timestamp = self.current_timestamp

        except KeyboardInterrupt:
            print("\n⚠️  手动停止在线决策循环")
        finally:
            # 停止态势接收，输出决策历史
            self.situation_receiver.stop()
            self._print_decision_history()

    def _print_decision_history(self):
        """打印决策历史汇总"""
        print("\n" + "=" * 80)
        print(f"决策历史汇总（共{len(self.decision_history)}次决策）")
        print("=" * 80)
        for i, (state, action, timestamp) in enumerate(self.decision_history, 1):
            print(f"\n决策{i}（时间戳：{timestamp:.0f}）：")
            print(f"  活跃目标：{state['active_targets']}")
            print(f"  目标毁伤：{ {tid: f'{d:.2f}' for tid, d in state['target_damage'].items()} }")
            print(f"  弹药库存：{state['ammo_stock']}")
            print(f"  动作维度：{action.shape}（弹药数×目标数）")
        print("\n" + "=" * 80)


# ------------------------------
# 任务结束条件与测试入口
# ------------------------------
def task_end_condition(solver: DynamicTargetWeaponAssignment) -> bool:
    """任务结束条件：所有目标已达毁伤阈值 或 手动停止（Ctrl+C）"""
    all_destroyed = all(d >= solver.damage_threshold for d in solver.target_damage.values())
    if all_destroyed:
        print("\n🎉 所有目标已达毁伤阈值，任务结束！")
        return True
    return False


def main():
    # 1. 准备初始态势（第一次输入，含初始目标和弹药）
    initial_timestamp = time.time()
    initial_situation = {
        "timestamp": initial_timestamp,
        "target_changes": {
            "add": [
                Target(id=1, value=100.0, components=[(0.3, 1.0), (0.4, 1.0), (0.3, 1.0)]),  # 目标1：3部件，价值100
                Target(id=2, value=80.0, components=[(0.5, 1.0), (0.5, 1.0)])               # 目标2：2部件，价值80
            ]
        },
        "ammo_changes": {
            "initial": [
                Ammunition(id=1, cost=10.0, stock=30, damage_profiles={  # 弹药1：成本10，库存30
                    1: [0.6, 0.4, 0.5],  # 对目标1的部件毁伤概率
                    2: [0.7, 0.5],       # 对目标2的部件毁伤概率
                    3: [0.5, 0.6, 0.4, 0.3],  # 预留目标3的毁伤概率
                    4: [0.6, 0.4, 0.5],  # 预留目标4的毁伤概率
                    5: [0.7, 0.5]        # 预留目标5的毁伤概率
                }),
                Ammunition(id=2, cost=15.0, stock=30, damage_profiles={  # 弹药2：成本15，库存30
                    1: [0.8, 0.6, 0.7],
                    2: [0.6, 0.7],
                    3: [0.7, 0.5, 0.6, 0.5],
                    4: [0.8, 0.6, 0.7],
                    5: [0.6, 0.7]
                }),
                Ammunition(id=3, cost=25.0, stock=30, damage_profiles={  # 弹药3：成本25，库存30（高威力）
                    1: [0.9, 0.8, 0.85],
                    2: [0.85, 0.9],
                    3: [0.8, 0.7, 0.75, 0.8],
                    4: [0.9, 0.8, 0.85],
                    5: [0.85, 0.9]
                })
            ]
        }
    }

    # 2. 适配性矩阵（shape: [弹药数, 最大目标ID]，预留5个目标的适配性）
    adaptability_matrix = np.array([
        [0.8, 0.7, 0.6, 0.8, 0.7],  # 弹药1：目标1-5的适配性
        [0.9, 0.8, 0.7, 0.9, 0.8],  # 弹药2：目标1-5的适配性
        [0.7, 0.9, 0.8, 0.7, 0.9]   # 弹药3：目标1-5的适配性
    ])

    # 3. 初始化动态求解器
    solver = DynamicTargetWeaponAssignment(
        initial_situation=initial_situation,
        adaptability_matrix=adaptability_matrix,
        damage_threshold=0.8,
        max_decision_time=5.0
    )

    # 4. 启动在线决策循环（另起线程运行态势发送模拟）
    from SituationReceiver import SituationSender
    sender = SituationSender()
    sender_thread = threading.Thread(target=sender.simulate_irregular_sending, args=(3, 30, 60), daemon=True)
    sender_thread.start()

    # 5. 运行在线决策
    solver.run_online_loop(task_end_condition)


if __name__ == "__main__":
    main()