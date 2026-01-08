import datetime
import time

import matplotlib.font_manager as fm
import re
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import matplotlib as mpl
import numpy as np
from matplotlib import colormaps
import os
import json
from speak_function import DialogueAnalyzer

class DiscourseAnalyzer111:

    # 所有小类及其对应的大类
    SUBCATEGORIES = {
        # 知识理解子类
        "观察记忆": "知识理解",
        "概括理解": "知识理解",
        "说明论证": "知识理解",

        # 表达交流子类
        "经历经验": "表达交流",
        "主观看法": "表达交流",
        "情感态度": "表达交流",

        # 实践应用子类
        "分析计算": "实践应用",
        "推测解释": "实践应用",
        "简单问题解决": "实践应用",

        # 创造迁移子类
        "综合问题解决": "创造迁移",
        "猜想探究": "创造迁移",
        "发现创新": "创造迁移"
    }

    # 优质课堂话语功能一级指标（用于对比）
    BENCHMARK_RATIOS = {
        "知识理解": 56.83,  # 你这个优质课指标顺序填错了,话语形式顺序也不一样，你检查一下
        "表达交流": 13.81,
        "实践应用": 19.11,
        "创造迁移": 10.25,
    }

    # 优质课堂话语功能二级指标（用于小类对比）
    BENCHMARK_RATIOS_SUB = {
        "观察记忆": 27.49,
        "概括理解": 20.16,
        "说明论证": 9.39,
        "经历经验": 3.83,
        "主观看法": 4.40,
        "情感态度": 6.82,
        "分析计算": 4.65,
        "推测解释": 12.44,
        "简单问题解决": 4.58,
        "综合问题解决": 4.86,
        "猜想探究": 5.78,
        "发现创新": 2.29
    }

    def __init__(self, api_key, csv_path,output_dir,time_stamp):
        """
        初始化DiscourseAnalyzer

        参数:
            api_key (str): OpenAI API密钥
            base_url (str): API基础URL
            excel_path (str): Excel文件路径
        """
        self.a1 = None
        self.a2 = None
        self.a3 = None

        self.api_key = api_key
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.csv_path = csv_path

        # 生成唯一时间戳标识符
        self.timestamp = time_stamp

        # 创建结果文件夹
        # self.output_dir = f"结果展示_{self.timestamp}"
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"结果将保存在文件夹: {self.output_dir}")

        # 设置中文字体支持
        self._setup_fonts()

        # 配置OpenAI API
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=10.0,
            base_url=self.base_url
        )

        # 初始化计数
        self.subcategory_counts = {sub: 0 for sub in self.SUBCATEGORIES.keys()}
        self.main_category_counts = {
            "知识理解": 0,
            "表达交流": 0,
            "实践应用": 0,
            "创造迁移": 0
        }

        # 加载数据
        self.sentences = self.load_excel_content_with_roles()
        self.output_lines = []
        self.unknown_lines = []

    def _setup_fonts(self):
        """设置中文字体支持"""
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 尝试自动查找系统中可用的中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun', 'FangSong']
        available_fonts = set(f.name for f in fm.fontManager.ttflist)

        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                break

    def load_excel_content_with_roles(self):
        """从Excel文件加载对话内容，每行作为一个完整的句子"""
        try:
            # 读取Excel文件
            df = pd.read_csv(self.csv_path)
            print(f"成功读取Excel文件，共{len(df)}行数据")

            # 检查必要的列是否存在
            if '角色' not in df.columns or '内容' not in df.columns:
                raise ValueError("Excel文件中缺少'角色'或'内容'列")

            # 提取角色和内容列，每行作为一个完整的句子
            roles = df['角色'].fillna('未知').tolist()
            contents = df['内容'].fillna('').tolist()

            # 组合成"角色：内容"的形式，每行一个完整的句子
            combined = [f"{role}：{content}" for role, content in zip(roles, contents) if content.strip()]

            print(f"处理后得到 {len(combined)} 个带角色的句子")
            return combined
        except Exception as e:
            print(f"读取Excel文件时出错: {str(e)}")
            return []

    def classify_text(self, sentence):
        """使用API对文本进行分类"""
        system_content = """
    你是一个专业高效的文本分类器，请将以下课堂对话中的每个句子以一个对话轮次为单元分类到五大类及细分小类：
     **必须跳过以下类型的语句：**
     1. 师生问候（如"同学们好"，"老师好"）
     2. 教师指令（如"上课"，"下课"，"请打开课本"，"开始讨论"）
     3. 简单复述（如"他说要画图"，"你刚才说..."）
     4. 简单评价（如"很好"，"正确"，"不错"）
     5. 组织性语言（如"接下来"，"我们继续"）

      **特别注意：**
     - 对于分析论证类对话，关注学生深度剖析问题、呈现完整思维过程和严谨证明结论的能力
     - 对于创造迁移类对话，关注学生创造性应用知识解决新问题的能力
     - 对于表达交流类对话，关注学生分享真实经历和表达个人观点的能力
     - 请务必从以下12个小类中选择一个返回：观察记忆、概括理解、说明论证、经历经验、主观看法、情感态度、分析计算、推测解释、简单问题解决、综合问题解决、猜想探究、发现创新
    请严格按以下格式返回分类结果：
    [小类名称]

    **重要：请只返回小类名称本身，不要添加任何额外符号、括号或说明文字。**

    分类定义：

   ### 知识理解
   1. **观察记忆**：通过观察，从长时记忆中提取与呈现材料一致的知识或提取相关知识，包括再认和回忆，对应一些常识方面的知识和学科知识。
                  例如："数学家欧几里得在几何原本提到了线面垂直的定义"、"三角形的内角和是180度"、"一条直线垂直平面上任意一条直线，那么我们就能得到该直线与平面垂直"
   2. **概括理解**：把某些具有一些相同属性的事物抽取出本质属性，推广到具有这些属性的一切事物中，并正确地以多种方式表征数学知识（用数、图表、符号、图解或词语）。反映学生数学符号意义的概括、数量关系的概括、图形特征的概括以及简单关系和简单运算与推理的概括。
                  例如："那我们看现在看一下刚才大家通过你的分析得到的这个概率跟我们这个频率的稳定值是否一致？我们来看一下它的概率是 1/ 2，然后稳定值0.5，一致吗？"
   3. **说明论证**：指学生在记忆、概括的基础上，能够在知识内部，学生能提取相关知识，选择和运用简单的问题解决策略，使用基于不同信息来源的表征，对其进行直接推理，解释现实的问题。学生能将重要的和不重要的信息区分开来，然后专注于重要信息，根据数学规则、原理做出解释、推理、判断的能力。
                  例如："那其实告诉我们什么？当实验的次数越来越多的时候，那么掷得点 6 这个事情大概需要平均需要几次就可以出现一次？"、"那我现在想问一下，能不能告诉我正面向上的概率是多少？"
   ### 表达交流
   1. **经历经验**：表达个人在学习数学过程中的经历、实践经验或社会实践等，分享学习心得。
                  例如："每大家在看体育比赛的时候，在关键时刻总是会出现屏住呼吸，特别紧张"、
   2. **主观看法**：表达基于个人的经验、知识、情感等的观点或判断，这些观点不需要基于客观证据或事实。
                  例如："好，我们看我们做这个实验，你有什么感觉？这个人回答什么感受？"
   3. **情感态度**：表达对事物的感受、态度和信念，如好奇心、兴趣、求知欲等。

   ### 实践应用
   1. **分析计算**：能够在熟悉的数学问题情境中直接应用数学知识进行作图、列式、计算解决问题。在熟悉情境中，数学内容直接且呈现清晰的一步应用问题或简单的多步应用问题，以及几何领域有固定程序的作图问题、统计领域的统计图绘制问题。
   2. **推测解释**：在较熟悉的实际任务情境中，学生能提取相关知识，选择和运用简单的问题解决策略，使用基于不同信息来源的表征，对其进行直接推理，解释现实的问题。
   3. **简单问题解决**：在不熟悉的任务情境中，学生选择、提取有用的数学信息，自行组织数学策略，建立数学模型，解决问题并完整表达解决过程。问题一般包含较复杂或冗余的数学信息，学生需要根据问题情境提取有用的数学信息，选择适当的策略，寻找合适表征模式，通过较复杂的决策解决问题。
                    例如："把旗杆抽象成一条直线，把地面抽象成平面"，"把桥成抽象成一条直线，把江面抽象成一个平面"

   ### 创造迁移
   1. **综合问题解决**：指知识的综合、方法的多样化以及数学思想方法的综合运用。具有知识容量大、解题方法多、能力要求高、突显数学思想方法的运用以及要求学生具有一定的创新意识和创新能力等特点。   
   2. **猜想探究**：指在开放的问题情境中，借助已有的知识经验，对数学材料进行加工，创造性解决问题。
                  例如："黑板上的圆可以用。你可以尺规，也可以用刻度尺，我给你三角板了，都可以。我们先听前面两位同学跟大家介绍他们的做法和依据，然后咱们再来看。你先说你的，他先边画。"
   3. **发现创新**：能够从已有知识和技能出发，通过猜想与合情推理构建知识之间的远联系，或提出发现新的好问题。。发现创新将涉及高水平概括，发现知识本质的联系；发现新的知识或规律；在多个概念进行联系。


   补充：
   1.知识理解补充：
   对于观察某图表、画面，或者从记忆里提取某方面的知识，则归类为观察记忆（如“我们来看这个图表，从这个图表中它告诉了我们什么？”）
   2.实践应用补充：
   对于在前项回答的基础上，继续在其上进行推理判断，解释相关问题，可以归为推理解释（如“你能否通过我们实验中所得的数据？来预测一下，经过多长时间水温能够到达 60 度？”）
   对于简单问题解决，在原有定义基础上，可以简单理解为问题包含的领域少，涵盖的知识面窄，往往只涉及数学方面，只需要运用数学手段来解决即可
   3.创造迁移补充：
   对于不仅仅包含数学方面的，还涉及到生活上其他领域的各种复杂问题，可以归类为综合问题解决
    """

        try:
            response = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"请分类以下课堂对话句子:\n{sentence}"}
                ]
            )

            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            print(f"API调用错误: {str(e)}")
            return "API错误"

    def should_skip(self, sentence):
        """优化跳过逻辑"""
        # 只跳过真正无关的句子
        strict_patterns = [
            r'^同学(们)?好[！!。.]?$',
            r'^老师好[！!。.]?$',
            r'^上课[！!。.]?$',
            r'^下课[！!。.]?$',
            r'^请打开(课本|书本)[！!。.]?$',
            r'^很好[！!。.]?$',
            r'^正确[！!。.]?$',
            r'^真棒[！!。.]?$',
            r'^开始答题[！!。.]?$',
            r'^请打开[！!。.]?$',
        ]

        # 检查是否完全匹配严格模式
        for pattern in strict_patterns:
            if re.fullmatch(pattern, sentence):
                return True

        # 对于包含组织性语言的句子，判断是否过于简单
        organizational_words = ["接下来", "然后", "继续", "下面"]
        if any(word in sentence for word in organizational_words):
            # 如果句子很短且主要是组织性语言才跳过
            if len(sentence) < 10:
                return True

        return False

    def clean_category(self, raw_category):
        """清洗分类结果"""
        # 去除方括号、引号等特殊字符
        cleaned = raw_category.strip()
        cleaned = re.sub(r'[\[\]()"\'「」]', '', cleaned)  # 去除各种括号

        # 去除常见前缀/后缀
        prefixes = ["分类结果:", "类别:", "分类:", "结果:", "小类名称:"]
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        # 处理可能的完整句子响应
        if "是" in cleaned and "类" in cleaned:
            # 尝试提取分类名称
            match = re.search(
                r'(观察记忆|概括理解|说明论证|经历经验|主观看法|情感态度|分析计算|推测解释|简单问题解决|综合问题解决|猜想探究|发现创新)',
                cleaned)
            if match:
                cleaned = match.group(1)

        return cleaned

    def process_sentences(self):
        """处理所有句子并进行分类，跳过的句子标记为其他但不计数"""
        for i, sentence in enumerate(self.sentences):
            try:
                if len(sentence) < 5:  # 跳过过短的句子
                    self.output_lines.append(f"{sentence} | 其他 ")
                    continue

                # 检查是否需要跳过
                if self.should_skip(sentence):
                    print(f"跳过句子 {i + 1}/{len(self.sentences)}: {sentence[:50]}...")
                    self.output_lines.append(f"{sentence} | 其他 ")
                    continue

                print(f"\n处理句子 {i + 1}/{len(self.sentences)}: {sentence[:50]}...")
                raw_category = self.classify_text(sentence)
                print(f"原始分类: {raw_category}")

                # 清洗分类结果
                cleaned_category = self.clean_category(raw_category)
                print(f"清洗后分类: '{cleaned_category}'")

                # 处理分类结果
                matched = False

                # 1. 精确匹配
                if cleaned_category in self.SUBCATEGORIES:
                    self.subcategory_counts[cleaned_category] += 1
                    main_category = self.SUBCATEGORIES[cleaned_category]
                    self.output_lines.append(f"{sentence} | {main_category} | {cleaned_category}")
                    matched = True
                else:
                    # 2. 部分匹配 - 尝试匹配已知小类的部分
                    for known_sub in self.SUBCATEGORIES.keys():
                        if known_sub in cleaned_category or cleaned_category in known_sub:
                            print(f"部分匹配: '{cleaned_category}' -> '{known_sub}'")
                            self.subcategory_counts[known_sub] += 1
                            main_category = self.SUBCATEGORIES[known_sub]
                            self.output_lines.append(f"{sentence} | {main_category} | {known_sub}")
                            matched = True
                            break

                # 3. 未匹配处理 - 标记为其他
                if not matched:
                    print(f"! 无法识别分类: '{cleaned_category}' (原始返回: '{raw_category}')")
                    self.output_lines.append(f"{sentence} | 其他 | 其他")

            except Exception as e:
                print(f"处理句子时出错: {str(e)}")
                self.output_lines.append(f"{sentence} | 其他 | 其他")

        # 计算大类分布
        for sub, count in self.subcategory_counts.items():
            main_cat = self.SUBCATEGORIES[sub]
            self.main_category_counts[main_cat] += count

    def save_results(self):
        """保存分类结果到文件"""
        # 保存结果 - 所有句子都保存在一个文件中，跳过的句子标记为"其他"
        with open(os.path.join(self.output_dir, f"分类后文本_{self.timestamp}.txt"), "w", encoding="utf-8") as f:
            a1 = os.path.join(self.output_dir, f"分类后文本_{self.timestamp}.txt")
            self.a1 = a1
            f.write("\n".join(self.output_lines))

    def generate_charts(self):
        """生成所有图表"""
        # 准备饼图数据 - 只包含计数大于0的已知小类
        valid_subcategories = {k: v for k, v in self.subcategory_counts.items() if v > 0}
        labels = list(valid_subcategories.keys())
        sizes = list(valid_subcategories.values())
        total_known = sum(sizes)
        total_sentences = len(self.sentences)

        # 过滤掉计数为0的大类
        valid_main_categories = {k: v for k, v in self.main_category_counts.items() if v > 0}

        if total_known == 0:
            # 创建错误提示图
            plt.figure(figsize=(8, 5))
            plt.text(0.5, 0.5, "分类失败：所有分类计数均为0\n请检查API响应和分类逻辑",
                     ha='center', va='center', fontsize=12)
            plt.axis('off')
            plt.savefig(os.path.join(self.output_dir, f"错误提示_{self.timestamp}.png"), bbox_inches='tight')
            plt.close()
            print("错误：所有分类计数均为0，请检查分类函数和API响应")
            return False

        # ==================== 数据解读 ====================

    def generate_conclusion_with_llm(self):
        """使用LLM生成结语报告，结合具体课堂对话内容"""
        system_content = """
            您是一位资深教育专家，请基于课堂对话文本和分类统计数据，撰写一份专业的教育分析报告。
            
            ## 重要写作要求：
            1. 严格采用**总分总结构**，即**只需要三段**：**开头总述、中间分析、最后总结**。
            2. **完全避免使用任何markdown格式、项目符号、编号列表、加粗等特殊格式**
            3. 语言要像一篇流畅的教育分析文章，段落之间自然过渡
            4. 每个观点都要有具体数据支撑，特别是**所有一级和二级指标的数据**（尤其是与基准值有显著差异的指标）都需要被解读和分析
            5. 分析要深入，**必须以数据解读为核心**，所有的优缺点评价和建议都应紧密围绕对应的数据表现展开，避免过于冗长或脱离数据。
            6. **引入案例时，必须使用“例如”、“体现在”、“具体而言”等自然流畅的过渡词句，避免任何生硬的插入式或主谓结构不明确的表达。**
            
            ## 内容结构指引：
            
            **第一段：总述**
            概括课堂整体情况和主要特征，引用核心数据指标（一级指标）。
            
            **第二段：分析（核心段落）**
            1. **全面解读**所有一级和二级指标数据。
            2. **在数据解读过程中**，选取**一到两个核心数据维度**，基于其表现来分析教师的专业教学技巧，包括概念建构的方法、提问设计的艺术、反馈运用的时机等。
            3. **在对数据解读的过程中，自然地穿插融入两个具体的课堂教学片段（案例）**：一个案例用于支撑一级指标的分析，另一个案例用于支撑二级指标的分析。
            
            **第三段：总结**
            对课堂进行简要总结，对教师专业发展提出前瞻性建议。**此段应保持简洁，长度不宜过长**，既要肯定优势，也要指明发展方向。
            
            ## 注意
            1. 确保**输出内容就是完整的、严格三段的分析报告，不要有任何额外的说明文字**。
            2. 知识理解比例13.55%明显低于优质课20.71%的标准。像这样的明显低于，说的委婉一点，例如改为**仍有一定的提升空间可挖掘，目前存在一定距离，可进一步优化**等
            3. **在描述教师行为时，务必使用准确且自然的中文动词搭配**，例如应使用“采用了指名提问的方式”或“进行了追问”，**避免使用“设计了提问”等生硬或不自然的搭配**。
            """

        # 准备数据作为用户输入
        total_known = sum(self.subcategory_counts.values())
        total_sentences = len(self.sentences)

        if total_known == 0:
            return "无有效分类数据，无法生成分析报告"

        # 计算大类实际分布比例
        main_category_ratios = {}
        for main_cat in ["知识理解", "表达交流", "实践应用", "创造迁移"]:
            count = self.main_category_counts.get(main_cat, 0)
            main_category_ratios[main_cat] = count / total_known * 100 if total_known > 0 else 0

        # 计算小类实际分布比例
        subcategory_ratios = {}
        for sub in self.SUBCATEGORIES:
            count = self.subcategory_counts.get(sub, 0)
            subcategory_ratios[sub] = count / total_known * 100 if total_known > 0 else 0

        # 改进示例收集：收集更多示例供LLM选择
        category_examples = {}
        for line in self.output_lines:
            parts = line.split(" | ")
            if len(parts) >= 3:
                sentence = parts[0]
                main_cat = parts[1]
                sub_cat = parts[2]

                # 只收集非"其他"的分类示例
                if sub_cat != "其他" and sub_cat in self.SUBCATEGORIES:
                    if sub_cat not in category_examples:
                        category_examples[sub_cat] = []

                    # 收集更多示例，让LLM有更多选择
                    if len(category_examples[sub_cat]) < 5:  # 每个小类最多收集5个例子
                        # 截断过长的句子
                        if len(sentence) > 100:
                            sentence = sentence[:100] + "..."
                        category_examples[sub_cat].append(sentence)

        # 格式化示例文本 - 提供所有可用示例，让LLM自行选择最合适的
        examples_text = "课堂对话内容示例（请根据分析需要选择最具代表性的例子）：\n\n"
        for sub_cat in self.SUBCATEGORIES.keys():
            if sub_cat in category_examples and category_examples[sub_cat]:
                examples_text += f"{sub_cat}类对话示例（共{len(category_examples[sub_cat])}个）：\n"
                for i, example in enumerate(category_examples[sub_cat]):
                    examples_text += f"  示例{i + 1}: {example}\n"
                examples_text += "\n"

        # 添加提示，让LLM选择最合适的例子
        examples_text += "重要提示：请根据您的专业判断，从上述示例中选择最具代表性、最能说明问题的例子用于分析。如果某个类别的示例都不够典型，宁愿不举例也不要使用不恰当的例子。"

        user_content = f"""
    课堂对话分类统计数据：
    总句子数: {len(self.sentences)}
    已分类句子: {total_known} ({total_known / len(self.sentences) * 100:.1f}%)

    大类分布:
      知识理解: {main_category_ratios['知识理解']:.1f}% (优质课堂指标: {self.BENCHMARK_RATIOS['知识理解']}%)
      表达交流: {main_category_ratios['表达交流']:.1f}% (优质课堂指标: {self.BENCHMARK_RATIOS['表达交流']}%)
      实践应用: {main_category_ratios['实践应用']:.1f}% (优质课堂指标: {self.BENCHMARK_RATIOS['实践应用']}%)
      创造迁移: {main_category_ratios['创造迁移']:.1f}% (优质课堂指标: {self.BENCHMARK_RATIOS['创造迁移']}%)

    小类分布:
      观察记忆: {subcategory_ratios['观察记忆']:.1f}% (指标: {self.BENCHMARK_RATIOS_SUB['观察记忆']}%)
      概括理解: {subcategory_ratios['概括理解']:.1f}% (指标: {self.BENCHMARK_RATIOS_SUB['概括理解']}%)
      说明论证: {subcategory_ratios['说明论证']:.1f}% (指标: {self.BENCHMARK_RATIOS_SUB['说明论证']}%)
      经历经验: {subcategory_ratios['经历经验']:.1f}% (指标: {self.BENCHMARK_RATIOS_SUB['经历经验']}%)
      主观看法: {subcategory_ratios['主观看法']:.1f}% (指标: {self.BENCHMARK_RATIOS_SUB['主观看法']}%)
      情感态度: {subcategory_ratios['情感态度']:.1f}% (指标: {self.BENCHMARK_RATIOS_SUB['情感态度']}%)
      分析计算: {subcategory_ratios['分析计算']:.1f}% (指标: {self.BENCHMARK_RATIOS_SUB['分析计算']}%)
      推测解释: {subcategory_ratios['推测解释']:.1f}% (指标: {self.BENCHMARK_RATIOS_SUB['推测解释']}%)
      简单问题解决: {subcategory_ratios['简单问题解决']:.1f}% (指标: {self.BENCHMARK_RATIOS_SUB['简单问题解决']}%)
      综合问题解决: {subcategory_ratios['综合问题解决']:.1f}% (指标: {self.BENCHMARK_RATIOS_SUB['综合问题解决']}%)
      猜想探究: {subcategory_ratios['猜想探究']:.1f}% (指标: {self.BENCHMARK_RATIOS_SUB['猜想探究']}%)
      发现创新: {subcategory_ratios['发现创新']:.1f}% (指标: {self.BENCHMARK_RATIOS_SUB['发现创新']}%)

    {examples_text}

    请基于以上统计数据和具体的课堂对话内容，生成专业分析报告。请运用您的专业判断，从提供的示例中选择最具代表性、最能支持您分析观点的对话内容。如果某个类别的所有示例都不够典型，宁愿不举例也不要使用不恰当的例子。
    """

        try:
            response = self.client.chat.completions.create(
                model="deepseek-v3",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=5000,  # 增加token限制以容纳更详细的分析
                temperature=0.3
            )

            import re

            def remove_line_breaks(dialogue_str):
                """
                处理字符串对话中的换行符：多个连续换行符（\n、\r\n、\r）合并为一个，单个换行符保留
                （原需求：两个换行符换一个，扩展为所有连续换行符统一合并，更实用）

                参数:
                    dialogue_str (str): 需要处理的对话字符串

                返回:
                    str: 处理后的字符串
                """
                # 检查输入是否为字符串类型
                if not isinstance(dialogue_str, str):
                    raise TypeError("输入必须是字符串类型")

                # 正则规则说明：
                # 1. [\r\n] 匹配任意一种换行符（\n 或 \r）
                # 2. + 匹配一个或多个连续的换行符（不管是 \n\n、\r\r、\r\n\r\n 还是混合的 \r\n\n）
                # 3. 替换为单个 \n（统一换行符格式，避免跨平台兼容问题）
                return re.sub(r'[\r\n]+', '\n', dialogue_str)

            conclusion = remove_line_breaks(response.choices[0].message.content.strip())
            return conclusion
        except Exception as e:
            print(f"生成数据解读时出错: {str(e)}")
            return "无法生成数据解读"

    def save_classification_results_to_excel(self):
        """保存分类结果到Excel/CSV/JSON"""
        total_known = sum(self.subcategory_counts.values())
        total_sentences = len(self.sentences)

        # 创建数据列表
        data = []

        # 固定维度值
        dimension = "师生话语功能"

        # 大类顺序
        main_categories = ["知识理解", "表达交流", "实践应用", "创造迁移"]

        # 遍历每个大类
        for main_cat in main_categories:
            # 获取该类下的所有小类
            subcats = [sub for sub, cat in self.SUBCATEGORIES.items() if cat == main_cat]

            # 计算该大类的总频次
            main_count = self.main_category_counts.get(main_cat, 0)
            main_ratio = main_count / total_known * 100 if total_known > 0 else 0

            # 添加该大类下的小类数据
            for i, subcat in enumerate(subcats):
                sub_count = self.subcategory_counts.get(subcat, 0)
                sub_ratio = sub_count / total_known * 100 if total_known > 0 else 0

                # 如果是第一个小类，同时包含大类信息
                if i == 0:
                    row = {
                        "维度": dimension,
                        "一级指标": main_cat,
                        "二级指标": subcat,
                        "频次(次)": sub_count,
                        "总计(次)": total_known,
                        "占比": f"{sub_ratio:.1f}%",
                        "汇总": f"{main_ratio:.1f}%",
                        "优质课": f"{self.BENCHMARK_RATIOS[main_cat]}%"
                    }
                else:
                    # 后续小类行，只包含小类信息
                    row = {
                        "维度": "",
                        "一级指标": "",
                        "二级指标": subcat,
                        "频次(次)": sub_count,
                        "总计(次)": total_known,
                        "占比": f"{sub_ratio:.1f}%",
                        "汇总": "",
                        "优质课": ""
                    }
                data.append(row)

        # 创建DataFrame
        df = pd.DataFrame(data)

        # 调整列顺序（确保与示例完全一致）
        df = df[["维度", "一级指标", "二级指标", "频次(次)", "总计(次)", "占比", "汇总", "优质课"]]

        # 保存到CSV文件（使用utf-8-sig编码，确保中文正常显示）

        csv_file = os.path.join(self.output_dir, f"话语功能_分类结果汇总_{self.timestamp}.csv")
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")

        self.a2 = csv_file
        print(f"分类结果已保存为CSV文件: '{csv_file}'")

        # # 同时保存Excel文件（可选）
        # excel_file = os.path.join(self.output_dir, f"分类结果汇总_{self.timestamp}.xlsx")
        # df.to_excel(excel_file, index=False)
        # print(f"分类结果已保存为Excel文件: '{excel_file}'")

        # 将DataFrame转换为JSON
        json_file = os.path.join(self.output_dir, f"分类结果汇总_{self.timestamp}.json")

        # 创建更结构化的JSON格式
        json_data = {
            "维度": dimension,
            "总句子数": total_sentences,
            "已分类句子数": total_known,
            "分类覆盖率": f"{total_known / total_sentences * 100:.1f}%" if total_sentences > 0 else "0%",
            "分类结果": []
        }

        # 添加分类结果
        for main_cat in main_categories:
            main_count = self.main_category_counts.get(main_cat, 0)
            main_ratio = main_count / total_known * 100 if total_known > 0 else 0

            # 大类信息
            main_category_info = {
                "一级指标": main_cat,
                "频次": main_count,
                "占比": f"{main_ratio:.1f}%",
                "优质课参考值": f"{self.BENCHMARK_RATIOS[main_cat]}%",
                "子类": []
            }

            # 添加该大类下的小类信息
            for subcat in [sub for sub, cat in self.SUBCATEGORIES.items() if cat == main_cat]:
                sub_count = self.subcategory_counts.get(subcat, 0)
                sub_ratio = sub_count / total_known * 100 if total_known > 0 else 0

                sub_category_info = {
                    "二级指标": subcat,
                    "频次": sub_count,
                    "占比": f"{sub_ratio:.1f}%",
                    "优质课参考值": f"{self.BENCHMARK_RATIOS_SUB[subcat]}%"
                }
                main_category_info["子类"].append(sub_category_info)

            json_data["分类结果"].append(main_category_info)

        # # 保存JSON文件
        # with open(json_file, "w", encoding="utf-8") as f:
        #     json.dump(json_data, f, ensure_ascii=False, indent=2)
        # print(f"分类结果已转换为JSON格式保存到 '{json_file}'")
        #
        # return json_data

    def save_discourse_form_json(self, conclusion_report):
        """生成符合要求的discourse_form格式的JSON"""
        # 计算总分类数
        total_known = sum(self.subcategory_counts.values())

        # 避免除以零错误
        if total_known == 0:
            total_known = 1  # 防止除以零，但不影响实际数据（占比将为0）

        analyzer = DialogueAnalyzer()
        # 读取文件1（CSV）和文件2（TXT）

        analyzer.read_csv(self.csv_path) \
            .read_txt(self.a1) \
            .match_and_generate()

        print("\n##################生成的JSON结果：")
        # print(analyzer.get_json_result(indent=2, sort_keys=True))

        discourse_form = {
            "discourse_form": {
                "summary": conclusion_report,
                "classifications": [],
                "time_class": json.loads(analyzer.to_json())
            }
        }

        # 按一级指标组织数据
        for main_cat in ["知识理解", "表达交流", "实践应用", "创造迁移"]:
            main_count = self.main_category_counts.get(main_cat, 0)

            # 计算大类占比
            main_ratio = (main_count / total_known) * 100

            # 获取该大类下的所有小类
            secondary_classifications = []
            for sub, cat in self.SUBCATEGORIES.items():
                if cat == main_cat:
                    sub_count = self.subcategory_counts.get(sub, 0)

                    # 计算小类占比
                    sub_ratio = (sub_count / total_known) * 100

                    # 获取优质课二级指标占比
                    sub_benchmark_ratio = self.BENCHMARK_RATIOS_SUB.get(sub, 0)

                    secondary_classifications.append({
                        "secondary_category_name": sub,
                        "ratio": round(sub_ratio, 1),
                        "Quality_lessons": sub_benchmark_ratio  # 添加优质课二级指标占比
                    })

            # 获取优质课一级指标占比
            benchmark_ratio = self.BENCHMARK_RATIOS.get(main_cat, 0)

            # 添加一级分类信息
            discourse_form["discourse_form"]["classifications"].append({
                "primary_category_name": main_cat,
                "ratio": round(main_ratio, 1),
                "Quality_lessons": benchmark_ratio,  # 添加优质课一级指标占比
                "secondary_classifications": secondary_classifications
            })

        # 保存JSON文件到结果文件夹
        json_file = os.path.join(self.output_dir, f"话语功能_返回文本_{self.timestamp}.json")
        # global a3
        self.a3 = json_file
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(discourse_form, f, ensure_ascii=False, indent=2)
        print(f"符合要求的JSON格式已保存到 '{json_file}'")

        return discourse_form

    def analyze(self):
        """执行完整分析流程"""
        # 处理所有句子
        self.process_sentences()

        # 保存分类结果到文本文件
        self.save_results()

        # 生成图表
        charts_generated = self.generate_charts()

        # 生成数据解读
        conclusion_report = ""
        total_known = sum(self.subcategory_counts.values())
        if total_known > 0:
            conclusion_report = self.generate_conclusion_with_llm()

        # 保存分类结果到Excel/CSV/JSON
        classification_results = {}
        if total_known > 0:
            classification_results = self.save_classification_results_to_excel()

        # 保存为discourse_form格式的JSON
        discourse_form = {}
        if total_known > 0 and conclusion_report:
            discourse_form = self.save_discourse_form_json(conclusion_report)

        # 返回主要结果
        return {
            "output_dir": self.output_dir,
            "classification_results": classification_results,
            "discourse_form": discourse_form,
            "conclusion_report": conclusion_report,
            "a1": self.a1,
            "a2": self.a2,
            "a3": self.a3,
        }


if __name__ == "__main__":
    # 初始化分析器
    API_KEY = "sk-358a054218ff4741a254022eeeb56b04"
    INPUT_FILE = r"C:\Users\ym\Desktop\22\探索视觉与逻辑的奇妙交汇.csv"
    OUTPUT_DIR = r"C:\Users\ym\Desktop\22"

    time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    t1 = time.time()
    analyzer = DiscourseAnalyzer111(
        api_key = API_KEY,
        csv_path = INPUT_FILE,
        output_dir = OUTPUT_DIR,
        time_stamp = time_stamp
    )
    # discourse_analyzer.py
    results = analyzer.analyze()
    t2 = time.time()
    t = t2 - t1

    # print("分类结果:", results["classification_results"])
    print("\n话语形式结果:", results["discourse_form"])
    print("\n结语报告:", results["conclusion_report"])
    print("\n所有文件保存目录:", results["output_dir"])
    print(results["a1"])  # 分类后文本 _20250821_095304.txt
    print(results["a2"])  # 分类结果汇总_20250821_095304.csv
    print(results["a3"])  # 返回文本_20250821_095304.json
    print(f"总用时{t}")