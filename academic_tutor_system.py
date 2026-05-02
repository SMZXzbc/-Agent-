
"""
============================================================
多Agent协作智能学术助教系统
============================================================
作者: AI Assistant
版本: 1.0
描述: 包含意图解析Agent、知识检索Agent、生成校验Agent的
      多Agent协作系统，通过共享工作记忆实现协作。
      支持数学(级数收敛性/曲率)和政治(双碳)课程答疑。
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time

# ==================== 数据结构与类型定义 ====================

class QuestionType(Enum):
    """问题类型枚举"""
    CONCEPT = "概念理解"
    CALCULATION = "计算验证"
    APPLICATION = "拓展应用"
    COMPARISON = "对比分析"
    UNKNOWN = "未分类"

class DifficultyLevel(Enum):
    """难度等级"""
    BASIC = "基础"
    INTERMEDIATE = "中级"
    ADVANCED = "高级"

@dataclass
class KnowledgeFragment:
    """知识片段"""
    id: str
    title: str
    content: str
    source: str
    keywords: List[str]
    difficulty: DifficultyLevel
    related_ids: List[str] = field(default_factory=list)

@dataclass
class SharedMemory:
    """共享工作记忆 - 三Agent协作的核心"""
    original_question: str = ""
    parsed_intent: Dict = field(default_factory=dict)
    retrieved_fragments: List[KnowledgeFragment] = field(default_factory=list)
    draft_answer: str = ""
    verified_answer: str = ""
    confidence_score: float = 0.0
    reasoning_chain: List[str] = field(default_factory=list)

    def log_step(self, agent_name: str, action: str, detail: str):
        self.reasoning_chain.append(f"[{agent_name}] {action}: {detail}")

# ==================== 知识库构建 ====================

class KnowledgeBase:
    """课程知识库 - 模拟RAG检索源"""

    def __init__(self):
        self.fragments: Dict[str, KnowledgeFragment] = {}
        self._init_math_knowledge()
        self._init_politics_knowledge()

    def _init_math_knowledge(self):
        math_data = [
            {
                "id": "math-001",
                "title": "级数收敛性的定义",
                "content": """级数收敛性定义：对于级数 Σ(n=1→∞) aₙ，如果其部分和数列 Sₙ = a₁ + a₂ + ... + aₙ 的极限存在，即 lim(n→∞) Sₙ = S（S为有限数），则称该级数收敛，S称为级数的和。若极限不存在，则称级数发散。

判断级数收敛的基本方法：
1. 定义法：直接求部分和的极限
2. 必要条件：若级数收敛，则 lim(n→∞) aₙ = 0（反之不成立）
3. 比较判别法、比值判别法、根值判别法""",
                "source": "教材第三章",
                "keywords": ["级数", "收敛", "发散", "部分和", "极限"],
                "difficulty": DifficultyLevel.BASIC,
                "related": ["math-002", "math-003"]
            },
            {
                "id": "math-002",
                "title": "几何级数（等比级数）",
                "content": """几何级数 Σ(n=0→∞) arⁿ 的收敛性：
- 当 |r| < 1 时收敛，和为 a/(1-r)
- 当 |r| ≥ 1 时发散

这是判断其他级数收敛性的重要参照标准。例如，p-级数 Σ(1/nᵖ) 当 p > 1 时收敛，p ≤ 1 时发散。""",
                "source": "教材第三章",
                "keywords": ["几何级数", "等比级数", "收敛条件", "p级数"],
                "difficulty": DifficultyLevel.BASIC,
                "related": ["math-001", "math-004"]
            },
            {
                "id": "math-003",
                "title": "曲率的定义与计算",
                "content": """曲率描述曲线在某点处的弯曲程度。

定义：曲线 y = f(x) 在点M处的曲率 K = |y''| / (1 + y'²)^(3/2)

曲率半径：R = 1/K

计算步骤：
1. 求一阶导数 y'
2. 求二阶导数 y''
3. 代入曲率公式计算

例如：圆 x² + y² = R² 上任意点的曲率都是 1/R，曲率半径就是R本身。""",
                "source": "教材第五章",
                "keywords": ["曲率", "曲率半径", "导数", "弯曲程度"],
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "related": ["math-005"]
            },
            {
                "id": "math-004",
                "title": "比值判别法（达朗贝尔判别法）",
                "content": """比值判别法：对于正项级数 Σaₙ，计算 lim(n→∞) aₙ₊₁/aₙ = ρ
- ρ < 1：级数收敛
- ρ > 1：级数发散
- ρ = 1：判别法失效，需用其他方法

适用场景：通项含有阶乘、指数函数的情况。""",
                "source": "教材第三章",
                "keywords": ["比值判别法", "达朗贝尔", "收敛判别", "正项级数"],
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "related": ["math-001", "math-002"]
            },
            {
                "id": "math-005",
                "title": "参数方程的曲率公式",
                "content": """对于参数方程 x = x(t), y = y(t)：
曲率 K = |x'y'' - x''y'| / (x'² + y'²)^(3/2)

其中 x' = dx/dt, y' = dy/dt, x'' = d²x/dt², y'' = d²y/dt²

这个公式在计算椭圆、摆线等参数曲线的曲率时特别有用。""",
                "source": "教材第五章",
                "keywords": ["参数方程", "曲率公式", "导数计算"],
                "difficulty": DifficultyLevel.ADVANCED,
                "related": ["math-003"]
            }
        ]
        for data in math_data:
            self.fragments[data["id"]] = KnowledgeFragment(
                id=data["id"], title=data["title"], content=data["content"],
                source=data["source"], keywords=data["keywords"],
                difficulty=data["difficulty"], related_ids=data["related"]
            )

    def _init_politics_knowledge(self):
        politics_data = [
            {
                "id": "pol-001",
                "title": "双碳目标的政策背景",
                "content": """双碳目标：2030年前实现碳达峰，2060年前实现碳中和。

政策背景：
1. 2020年9月，习近平在联合国大会上首次提出
2. 写入"十四五"规划和2035远景目标纲要
3. 2021年发布《2030年前碳达峰行动方案》

核心逻辑：能源结构转型 + 产业升级 + 碳汇建设""",
                "source": "政治课PPT",
                "keywords": ["双碳", "碳达峰", "碳中和", "能源转型", "政策"],
                "difficulty": DifficultyLevel.BASIC,
                "related": ["pol-002"]
            },
            {
                "id": "pol-002",
                "title": "碳排放权交易机制",
                "content": """碳排放权交易（ETS）：通过市场机制控制碳排放总量。

运作原理：
1. 政府设定碳排放总量上限并分配配额
2. 企业实际排放低于配额可出售剩余配额
3. 超额排放需购买配额或面临处罚

中国全国碳市场2021年7月启动，首批纳入发电行业，覆盖约45亿吨CO₂排放量，成为全球最大碳市场。""",
                "source": "政治课PPT",
                "keywords": ["碳交易", "ETS", "碳市场", "配额", "市场机制"],
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "related": ["pol-001", "pol-003"]
            },
            {
                "id": "pol-003",
                "title": "双碳与高质量发展的关系",
                "content": """双碳目标与高质量发展的辩证关系：

不是简单的"去工业化"，而是：
1. 倒逼技术创新（新能源、储能、氢能）
2. 催生新产业（碳资产管理、碳核查、碳金融）
3. 重塑竞争优势（绿色产品获得国际认可）

课本联系：对应"新发展理念"中的"绿色发展"，是高质量发展的重要维度。""",
                "source": "政治课PPT",
                "keywords": ["高质量发展", "绿色发展", "新发展理念", "技术创新"],
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "related": ["pol-001", "pol-002"]
            }
        ]
        for data in politics_data:
            self.fragments[data["id"]] = KnowledgeFragment(
                id=data["id"], title=data["title"], content=data["content"],
                source=data["source"], keywords=data["keywords"],
                difficulty=data["difficulty"], related_ids=data["related"]
            )

    def search(self, query: str, top_k: int = 3) -> List[KnowledgeFragment]:
        query_keywords = set(query.lower().split())
        scored_fragments = []
        for frag in self.fragments.values():
            frag_keywords = set(k.lower() for k in frag.keywords)
            title_words = set(frag.title.lower().split())
            match_score = len(query_keywords & frag_keywords) * 2
            title_match = len(query_keywords & title_words) * 3
            content_match = sum(1 for word in query_keywords if word in frag.content.lower())
            total_score = match_score + title_match + content_match
            if total_score > 0:
                scored_fragments.append((total_score, frag))
        scored_fragments.sort(key=lambda x: x[0], reverse=True)
        return [frag for _, frag in scored_fragments[:top_k]]

    def get_by_id(self, frag_id: str) -> Optional[KnowledgeFragment]:
        return self.fragments.get(frag_id)

# ==================== Agent 1: 意图解析Agent ====================

class IntentParserAgent:
    """意图解析Agent - 通过长链推理分析学生问题的深层需求"""

    def __init__(self):
        self.type_indicators = {
            QuestionType.CONCEPT: ["什么是", "定义", "概念", "意思", "为什么", "原理"],
            QuestionType.CALCULATION: ["计算", "求解", "证明", "推导", "步骤", "怎么做"],
            QuestionType.APPLICATION: ["应用", "例子", "实际", "案例", "怎么用", "场景"],
            QuestionType.COMPARISON: ["区别", "比较", "对比", "不同", "联系", "和"]
        }
        self.subject_keywords = {
            "math": ["级数", "收敛", "曲率", "导数", "极限", "积分", "微分", "公式"],
            "politics": ["双碳", "碳达峰", "碳中和", "政策", "发展", "理念", "绿色"]
        }

    def parse(self, question: str, memory: SharedMemory) -> Dict[str, Any]:
        memory.log_step("意图解析Agent", "开始分析", f"原始问题: {question}")

        q_type = self._detect_question_type(question)
        memory.log_step("意图解析Agent", "类型识别", f"判定为: {q_type.value}")

        subject = self._detect_subject(question)
        memory.log_step("意图解析Agent", "领域识别", f"判定学科: {subject}")

        concepts = self._extract_concepts(question, subject)
        memory.log_step("意图解析Agent", "概念提取", f"核心概念: {concepts}")

        deep_intent = self._infer_deep_intent(question, q_type, concepts)
        memory.log_step("意图解析Agent", "深层推断", f"真实需求: {deep_intent}")

        difficulty = self._estimate_difficulty(question, q_type)
        memory.log_step("意图解析Agent", "难度评估", f"预期难度: {difficulty.value}")

        result = {
            "question_type": q_type,
            "subject": subject,
            "core_concepts": concepts,
            "deep_intent": deep_intent,
            "expected_difficulty": difficulty,
            "keywords_for_search": concepts + [q_type.value]
        }
        memory.parsed_intent = result
        return result

    def _detect_question_type(self, question: str) -> QuestionType:
        for q_type, indicators in self.type_indicators.items():
            if any(ind in question for ind in indicators):
                return q_type
        return QuestionType.UNKNOWN

    def _detect_subject(self, question: str) -> str:
        math_score = sum(1 for kw in self.subject_keywords["math"] if kw in question)
        politics_score = sum(1 for kw in self.subject_keywords["politics"] if kw in question)
        if math_score > politics_score: return "数学"
        elif politics_score > math_score: return "政治"
        return "通用"

    def _extract_concepts(self, question: str, subject: str) -> List[str]:
        if subject == "数学":
            concepts = []
            if "级数" in question or "收敛" in question:
                concepts.extend(["级数", "收敛性"])
            if "曲率" in question:
                concepts.append("曲率")
            if "几何级数" in question or "等比" in question:
                concepts.append("几何级数")
            return concepts
        elif subject == "政治":
            concepts = []
            if "双碳" in question or "碳达峰" in question or "碳中和" in question:
                concepts.extend(["双碳", "碳达峰", "碳中和"])
            if "交易" in question or "市场" in question:
                concepts.append("碳交易")
            return concepts
        return []

    def _infer_deep_intent(self, question: str, q_type: QuestionType, concepts: List[str]) -> str:
        if q_type == QuestionType.CONCEPT:
            if not concepts: return "学生可能对基础概念理解模糊，需要结构化解释"
            return f"学生需要理解{'、'.join(concepts)}的本质定义和直观意义，而非死记硬背"
        elif q_type == QuestionType.CALCULATION:
            return f"学生需要{'、'.join(concepts)}的完整计算流程，重点在于步骤逻辑而非仅答案"
        elif q_type == QuestionType.APPLICATION:
            return "学生需要将抽象概念与具体场景连接，建立直觉理解"
        elif q_type == QuestionType.COMPARISON:
            return "学生需要建立概念间的联系网络，理解异同背后的原理"
        return "学生需要基础信息补充"

    def _estimate_difficulty(self, question: str, q_type: QuestionType) -> DifficultyLevel:
        advanced_indicators = ["证明", "推导", "深入", "详细", "复杂", "参数方程"]
        if any(ind in question for ind in advanced_indicators):
            return DifficultyLevel.ADVANCED
        if q_type == QuestionType.CALCULATION or q_type == QuestionType.COMPARISON:
            return DifficultyLevel.INTERMEDIATE
        return DifficultyLevel.BASIC

# ==================== Agent 2: 知识检索Agent ====================

class KnowledgeRetrievalAgent:
    """知识检索Agent - 在知识库中进行RAG检索"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def retrieve(self, parsed_intent: Dict, memory: SharedMemory) -> List[KnowledgeFragment]:
        memory.log_step("知识检索Agent", "开始检索", f"关键词: {parsed_intent['keywords_for_search']}")

        search_query = " ".join(parsed_intent["keywords_for_search"] + parsed_intent["core_concepts"])
        primary_results = self.kb.search(search_query, top_k=3)
        memory.log_step("知识检索Agent", "基础检索", f"获得 {len(primary_results)} 条主结果")

        extended_results = []
        for frag in primary_results:
            for related_id in frag.related_ids:
                related_frag = self.kb.get_by_id(related_id)
                if related_frag and related_frag not in primary_results and related_frag not in extended_results:
                    extended_results.append(related_frag)

        memory.log_step("知识检索Agent", "扩展检索", f"获得 {len(extended_results)} 条关联结果")

        all_results = primary_results + extended_results

        expected_diff = parsed_intent["expected_difficulty"]
        if expected_diff == DifficultyLevel.BASIC:
            filtered = [f for f in all_results if f.difficulty == DifficultyLevel.BASIC]
            if filtered:
                all_results = filtered
                memory.log_step("知识检索Agent", "难度过滤", "过滤为仅基础内容")

        memory.retrieved_fragments = all_results
        return all_results

# ==================== Agent 3: 生成校验Agent ====================

class GenerationVerificationAgent:
    """生成校验Agent - 事实一致性检查与难度适配"""

    def __init__(self):
        self.max_answer_length = 800

    def generate(self, memory: SharedMemory) -> Dict[str, Any]:
        memory.log_step("生成校验Agent", "开始生成", "整合检索结果...")

        draft = self._generate_draft(memory)
        memory.draft_answer = draft
        memory.log_step("生成校验Agent", "草稿生成", f"草稿长度: {len(draft)}字")

        verification_result = self._verify_facts(memory)
        memory.log_step("生成校验Agent", "事实校验", f"置信度: {verification_result['confidence']:.2f}")

        adapted = self._adapt_difficulty(draft, memory.parsed_intent, verification_result)
        memory.log_step("生成校验Agent", "难度适配", f"适配策略: {verification_result['adaptation']}")

        final_answer = self._format_output(adapted, memory)
        memory.verified_answer = final_answer["content"]
        memory.confidence_score = verification_result["confidence"]

        memory.log_step("生成校验Agent", "输出完成", f"最终置信度: {memory.confidence_score:.2f}")
        return final_answer

    def _generate_draft(self, memory: SharedMemory) -> str:
        intent = memory.parsed_intent
        fragments = memory.retrieved_fragments

        if not fragments:
            return "抱歉，当前知识库中没有找到与你问题直接相关的内容。建议：1) 检查关键词是否准确；2) 咨询授课教师。"

        parts = []
        parts.append(f"📌 **问题理解**：你正在询问关于「{'、'.join(intent['core_concepts'])}」的{intent['question_type'].value}类问题。")
        parts.append("")
        parts.append("📚 **核心知识**：")

        for i, frag in enumerate(fragments[:3], 1):
            parts.append(f"
**{i}. {frag.title}**（来源：{frag.source}）")
            parts.append(frag.content)

        parts.append("")
        parts.append("💡 **学习提示**：")

        if intent["question_type"] == QuestionType.CONCEPT:
            parts.append("理解概念的关键是抓住定义中的条件和结论。建议尝试用自己的话复述定义，并思考'如果去掉某个条件会怎样？'")
        elif intent["question_type"] == QuestionType.CALCULATION:
            parts.append("计算类问题重在步骤清晰。建议先写出公式，再代入数据，最后检查单位是否一致。")
        elif intent["question_type"] == QuestionType.APPLICATION:
            parts.append("应用类问题需要建立模型。建议从'已知什么、求什么、用什么工具'三个角度拆解问题。")

        return "
".join(parts)

    def _verify_facts(self, memory: SharedMemory) -> Dict:
        draft = memory.draft_answer
        fragments = memory.retrieved_fragments

        confidence = 0.7
        for frag in fragments:
            if frag.title in draft:
                confidence += 0.05
            keyword_matches = sum(1 for kw in frag.keywords if kw in draft)
            confidence += min(keyword_matches * 0.02, 0.1)

        if "据我所知" in draft or "可能" in draft:
            confidence -= 0.1

        confidence = min(confidence, 0.95)

        adaptation = "保持当前难度"
        if memory.parsed_intent["expected_difficulty"] == DifficultyLevel.BASIC and confidence < 0.8:
            adaptation = "降低难度，增加解释"
        elif memory.parsed_intent["expected_difficulty"] == DifficultyLevel.ADVANCED and confidence > 0.9:
            adaptation = "可增加拓展内容"

        return {
            "confidence": confidence,
            "adaptation": adaptation,
            "issues": []
        }

    def _adapt_difficulty(self, draft: str, intent: Dict, verification: Dict) -> str:
        if verification["adaptation"] == "降低难度，增加解释":
            basic_prefix = "🎯 **入门引导**：让我们从基础开始理解这个问题...

"
            return basic_prefix + draft
        elif verification["adaptation"] == "可增加拓展内容":
            extension = "

🔬 **拓展思考**：学有余力的同学可以思考：这个结论能否推广到更一般的情况？相关的前沿研究有哪些？"
            return draft + extension
        return draft

    def _format_output(self, adapted: str, memory: SharedMemory) -> Dict[str, Any]:
        intent = memory.parsed_intent

        structured = {
            "content": adapted,
            "metadata": {
                "question_type": intent["question_type"].value,
                "subject": intent["subject"],
                "difficulty": intent["expected_difficulty"].value,
                "confidence": memory.confidence_score,
                "sources": [f.title for f in memory.retrieved_fragments],
                "reasoning_chain": memory.reasoning_chain
            },
            "interaction_guide": {
                "step_1": "先阅读'问题理解'部分，确认这是否是你想问的",
                "step_2": "浏览'核心知识'，标记不理解的地方",
                "step_3": "如果还有疑问，可以追问更具体的子问题"
            }
        }
        return structured

# ==================== 主系统：多Agent协作引擎 ====================

class AcademicTutorSystem:
    """智能学术助教系统 - 多Agent协作引擎"""

    def __init__(self):
        self.kb = KnowledgeBase()
        self.intent_agent = IntentParserAgent()
        self.retrieval_agent = KnowledgeRetrievalAgent(self.kb)
        self.generation_agent = GenerationVerificationAgent()
        self.query_count = 0
        self.total_response_time = 0

    def answer(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        start_time = time.time()
        memory = SharedMemory(original_question=question)

        if verbose:
            print("=" * 60)
            print("🎓 智能学术助教系统 - 多Agent协作启动")
            print("=" * 60)
            print(f"📨 学生提问: {question}")
            print("-" * 60)

        # Agent 1: 意图解析
        if verbose:
            print("
🔍 [Agent 1] 意图解析Agent - 长链推理分析中...")
        parsed = self.intent_agent.parse(question, memory)

        if verbose:
            print(f"   ├─ 问题类型: {parsed['question_type'].value}")
            print(f"   ├─ 学科领域: {parsed['subject']}")
            print(f"   ├─ 核心概念: {', '.join(parsed['core_concepts'])}")
            print(f"   ├─ 深层需求: {parsed['deep_intent']}")
            print(f"   └─ 预期难度: {parsed['expected_difficulty'].value}")

        # Agent 2: 知识检索
        if verbose:
            print("
📚 [Agent 2] 知识检索Agent - RAG检索中...")
        fragments = self.retrieval_agent.retrieve(parsed, memory)

        if verbose:
            print(f"   ├─ 检索到 {len(fragments)} 条知识片段:")
            for i, frag in enumerate(fragments, 1):
                print(f"   │  {i}. [{frag.id}] {frag.title} (难度: {frag.difficulty.value})")
            print(f"   └─ 来源分布: {set(f.source for f in fragments)}")

        # Agent 3: 生成校验
        if verbose:
            print("
✅ [Agent 3] 生成校验Agent - 事实校验与输出生成...")
        result = self.generation_agent.generate(memory)

        elapsed = time.time() - start_time
        self.query_count += 1
        self.total_response_time += elapsed

        if verbose:
            print(f"
{'=' * 60}")
            print("📊 系统统计")
            print(f"   ├─ 响应时间: {elapsed:.3f}秒")
            print(f"   ├─ 置信度评分: {result['metadata']['confidence']:.2f}")
            print(f"   ├─ 推理链长度: {len(memory.reasoning_chain)} 步")
            print(f"   └─ 累计服务: {self.query_count} 次提问")
            print(f"{'=' * 60}
")

        result["_internal"] = {
            "shared_memory": {
                "original_question": memory.original_question,
                "reasoning_chain": memory.reasoning_chain,
                "retrieved_fragment_ids": [f.id for f in memory.retrieved_fragments]
            }
        }
        return result

    def get_stats(self) -> Dict:
        avg_time = self.total_response_time / self.query_count if self.query_count > 0 else 0
        return {
            "total_queries": self.query_count,
            "avg_response_time": f"{avg_time:.3f}s",
            "knowledge_fragments": len(self.kb.fragments)
        }

# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 初始化系统
    tutor = AcademicTutorSystem()

    # 测试问题
    test_questions = [
        "什么是级数收敛性？能给我详细解释一下吗？",
        "双碳目标和高质量发展有什么关系？",
        "曲率公式怎么推导？能给我详细步骤吗？"
    ]

    for q in test_questions:
        result = tutor.answer(q)
        print(f"
{'='*60}")
        print("最终回答:")
        print(result["content"])
        print(f"{'='*60}
")

    # 输出系统统计
    print("
📊 系统运行统计:")
    print(tutor.get_stats())
