import socket
import json
import threading
import time
from typing import Optional, Dict, List
from StaticMatch.AHL_NSGA_II import Target, Ammunition  # 导入静态弹目匹配中的目标/弹药类


class SituationReceiver:
    """仅支持目标新增的态势接收模块：不定时接收外部态势更新（新增目标/弹药补充）"""
    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)  # 单连接（可扩展为多线程支持多源态势）
        
        self.new_situation: Optional[Dict] = None  # 缓存最新态势
        self.receive_event = threading.Event()  # 态势接收触发事件
        self.running = True  # 运行标志

    def _parse_target(self, target_dict: Dict) -> Target:
        """解析JSON格式的目标数据，转换为Target对象"""
        return Target(
            target_id=int(target_dict["id"]),
            value=float(target_dict["value"]),
            components=[(float(weight), float(health)) for weight, health in target_dict["components"]]
        )

    def _parse_situation(self, data: str) -> Optional[Dict]:
        """解析接收到的JSON态势数据，校验格式合法性"""
        try:
            situation = json.loads(data)
            # 1. 必选字段校验
            required_fields = ["timestamp", "target_changes", "ammo_changes"]
            for field in required_fields:
                if field not in situation:
                    raise ValueError(f"态势数据缺少必选字段：{field}")
            
            # 2. 解析时间戳（Unix秒级时间戳）
            situation["timestamp"] = float(situation["timestamp"])
            
            # 3. 解析新增目标（仅处理add字段，移除remove字段）
            target_changes = situation["target_changes"]
            situation["target_changes"] = {"add": [], "update": []}  # 固定格式，避免key不存在
            
            # 3.1 解析新增目标
            if "add" in target_changes and isinstance(target_changes["add"], list):
                parsed_targets = [self._parse_target(t) for t in target_changes["add"]]
                situation["target_changes"]["add"] = parsed_targets
            
            # 3.2 解析目标毁伤度修正（可选，如探测到实际毁伤与模型预测偏差）
            if "update" in target_changes and isinstance(target_changes["update"], list):
                parsed_updates = []
                for update in target_changes["update"]:
                    if "target_id" in update and "new_damage" in update:
                        parsed_updates.append({
                            "target_id": int(update["target_id"]),
                            "new_damage": min(1.0, max(0.0, float(update["new_damage"])))  # 毁伤度限制在0-1
                        })
                situation["target_changes"]["update"] = parsed_updates
            
            # 4. 解析弹药补充（可选，如后勤补给）
            ammo_changes = situation["ammo_changes"]
            situation["ammo_changes"] = {"add": []}  # 固定格式
            
            if "add" in ammo_changes and isinstance(ammo_changes["add"], list):
                parsed_ammo_add = []
                for ammo in ammo_changes["add"]:
                    if "ammo_id" in ammo and "add_stock" in ammo:
                        parsed_ammo_add.append({
                            "ammo_id": int(ammo["ammo_id"]),
                            "add_stock": max(0, int(ammo["add_stock"]))  # 补充数量非负
                        })
                situation["ammo_changes"]["add"] = parsed_ammo_add
            
            # 5. 标记是否需要重新决策（有新增目标/弹药补充则需要）
            need_redecision = (
                len(situation["target_changes"]["add"]) > 0 or
                len(situation["target_changes"]["update"]) > 0 or
                len(situation["ammo_changes"]["add"]) > 0
            )
            situation["need_redecision"] = need_redecision
            
            return situation
        except Exception as e:
            print(f"态势解析失败：{str(e)}，原始数据：{data[:200]}...")  # 打印前200字符避免过长
            return None

    def _receive_thread(self):
        """子线程：持续监听态势输入，接收后触发事件"""
        print(f"态势接收模块启动，监听 {self.host}:{self.port}（仅支持目标新增/弹药补充）")
        while self.running:
            try:
                conn, addr = self.socket.accept()
                with conn:
                    data = conn.recv(2048).decode("utf-8")  # 接收最大2KB数据（足够存储增量态势）
                    if data:
                        parsed_situation = self._parse_situation(data)
                        if parsed_situation:
                            self.new_situation = parsed_situation
                            self.receive_event.set()  # 触发主线程状态更新
                            print(f"✅ 接收新态势：时间戳={parsed_situation['timestamp']:.0f}，"
                                  f"新增目标数={len(parsed_situation['target_changes']['add'])}，"
                                  f"弹药补充数={len(parsed_situation['ammo_changes']['add'])}")
            except Exception as e:
                if self.running:  # 非停止状态下的异常才打印
                    print(f"态势接收异常：{str(e)}")

    def start(self):
        """启动态势接收线程"""
        self.receive_thread = threading.Thread(target=self._receive_thread, daemon=True)
        self.receive_thread.start()

    def get_new_situation(self) -> Optional[Dict]:
        """获取最新态势（消费后清空缓存）"""
        situation = self.new_situation
        self.new_situation = None
        self.receive_event.clear()  # 重置事件
        return situation

    def wait_for_situation(self, timeout: float = 1.0) -> bool:
        """等待新态势（超时返回False，避免阻塞）"""
        return self.receive_event.wait(timeout)

    def stop(self):
        """停止态势接收"""
        self.running = False
        self.socket.close()
        print("\n❌ 态势接收模块停止")


# ------------------------------
# 态势发送测试工具（模拟外部系统发送新增目标）
# ------------------------------
class SituationSender:
    """态势发送工具：模拟不定时发送新增目标/弹药补充（用于测试）"""
    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        self.host = host
        self.port = port

    def send(self, situation: Dict):
        """发送态势数据到接收端"""
        try:
            # 序列化：将Target对象转换为字典（避免JSON无法序列化）
            def obj2dict(obj):
                if isinstance(obj, Target):
                    return {
                        "id": obj.id,
                        "value": obj.value,
                        "components": obj.components
                    }
                raise TypeError(f"不支持的类型：{type(obj)}")
            
            situation_json = json.dumps(situation, default=obj2dict, ensure_ascii=False)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                s.sendall(situation_json.encode("utf-8"))
        except Exception as e:
            print(f"态势发送失败：{str(e)}")

    def simulate_irregular_sending(self, send_count: int = 3, min_interval: int = 30, max_interval: int = 120):
        """
        模拟不定时发送态势（仅新增目标/弹药补充）
        Args:
            send_count: 发送次数
            min_interval: 最小间隔（秒）
            max_interval: 最大间隔（秒）
        """
        print(f"\n启动态势发送模拟（共{send_count}次，间隔{min_interval}-{max_interval}秒）")
        base_target_id = 3  # 初始目标ID从3开始（假设初始态势已有1、2号目标）
        base_ammo_id = 1    # 弹药补充从1号开始
        
        for i in range(send_count):
            # 1. 随机间隔（模拟不定时）
            interval = random.randint(min_interval, max_interval)
            print(f"\n 第{i+1}次态势将在{interval}秒后发送")
            time.sleep(interval)
            
            # 2. 构造态势数据（每次新增1个目标，第2次开始随机补充弹药）
            timestamp = time.time()
            new_target = Target(
                target_id=base_target_id + i,
                value=random.uniform(80.0, 150.0),  # 目标价值80-150
                components=[(round(random.uniform(0.2, 0.5), 2), 1.0) for _ in range(random.randint(2, 4))]  # 2-4个部件
            )
            
            situation = {
                "timestamp": timestamp,
                "target_changes": {
                    "add": [new_target],  # 仅新增1个目标
                    "update": []  # 暂不修正毁伤度（测试时可手动添加）
                },
                "ammo_changes": {
                    "add": []  # 第2次发送后随机补充弹药
                }
            }
            
            # 第2次及以后发送时，50%概率补充弹药
            if i >= 1 and random.random() < 0.5:
                situation["ammo_changes"]["add"].append({
                    "ammo_id": base_ammo_id + random.randint(0, 2),  # 1-3号弹药随机
                    "add_stock": random.randint(5, 15)  # 补充5-15发
                })
            
            # 3. 发送态势
            self.send(situation)
            print(f"📥 第{i+1}次态势发送完成：新增目标ID={new_target.id}，价值={new_target.value:.0f}")
        
        print("\n📤 态势发送模拟结束")