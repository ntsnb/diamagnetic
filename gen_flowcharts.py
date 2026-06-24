import os

output_dir = r"C:\Users\admin\Desktop\项目流程图"
os.makedirs(output_dir, exist_ok=True)

S_START = "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontStyle=1;fontSize=12;"
S_TASK  = "rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=11;"
S_SUB   = "rounded=0;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;"
S_OUT   = "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontStyle=1;fontSize=11;"
S_TITLE = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontStyle=1;fontSize=14;"
S_DEC   = "rhombus;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=10;"

def cell(cid, val, style, x, y, w, h):
    return (
        '        <mxCell id="%s" value="%s" style="%s" vertex="1" parent="1">\n'
        '          <mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/>\n'
        '        </mxCell>' % (cid, val, style, x, y, w, h)
    )

def edge(eid, src, tgt, lbl=""):
    s = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
    return (
        '        <mxCell id="%s" value="%s" style="%s" edge="1" source="%s" target="%s" parent="1">\n'
        '          <mxGeometry relative="1" as="geometry"/>\n'
        '        </mxCell>' % (eid, lbl, s, src, tgt)
    )

def diagram(name, page, body):
    return (
        '<mxfile host="cli-anything">\n'
        '  <diagram id="d-%s" name="%s">\n'
        '    <mxGraphModel dx="1422" dy="762" grid="1" gridSize="10" page="1"'
        ' pageWidth="1169" pageHeight="827">\n'
        '      <root>\n'
        '        <mxCell id="0"/>\n'
        '        <mxCell id="1" parent="0"/>\n'
        '%s\n'
        '      </root>\n'
        '    </mxGraphModel>\n'
        '  </diagram>\n'
        '</mxfile>'
    ) % (name, page, body)

# ============================================================
# Topic 1
# ============================================================
def topic1():
    cx = 450
    rows = [
        cell("t1h", "课题一：配电网多源异构数据语义融合与拓扑动态演化辨识技术研究", S_TITLE, 50, 15, 1050, 45),
        cell("t1s", "开始", S_START, cx-60, 75, 120, 40),
        cell("t1i", "多源异构数据输入（量测终端 / 自动化系统 / 保护装置 / 设备监控系统）", S_TASK, cx-230, 145, 460, 55),
        cell("t1a", "研究任务1：多源异构数据融合建模与语义统一表达&#xa;（语义对齐 / 标准化场景资源库构建）", S_TASK, cx-230, 230, 460, 60),
        cell("t1b", "研究任务2：多源数据时空对齐与图模数协同建模&#xa;（时空对齐 / 缺失修复 / 动态增量关联更新机制）", S_TASK, cx-230, 320, 460, 60),
        cell("t1c", "研究任务3：数据驱动与机理融合的拓扑状态实时辨识&#xa;（多模态融合 / 物理机理约束 / 快速拓扑辨识方法）", S_TASK, cx-230, 410, 460, 60),
        cell("t1d", "研究任务4：拓扑动态演化跟踪与场景切换识别&#xa;（演化过程建模 / 在线跟踪 / 全局状态迁移感知）", S_TASK, cx-230, 500, 460, 60),
        cell("t1o", "输出：统一场景资源体系 + 在线状态输入体系&#xa;→ 支撑课题二（重构）/ 课题三（协控）/ 课题五（示范）", S_OUT, cx-240, 590, 480, 55),
        cell("t1e", "结束", S_START, cx-60, 675, 120, 40),
    ]
    edges = [
        ("e1_0","t1s","t1i"), ("e1_1","t1i","t1a"), ("e1_2","t1a","t1b"),
        ("e1_3","t1b","t1c"), ("e1_4","t1c","t1d"), ("e1_5","t1d","t1o"), ("e1_6","t1o","t1e"),
    ]
    body = "\n".join(rows) + "\n" + "\n".join(edge(a,b,c) for a,b,c in edges)
    return diagram("t1", "课题一流程图", body)

# ============================================================
# Topic 2
# ============================================================
def topic2():
    cx = 450
    rows = [
        cell("t2h", "课题二：配电网拓扑敏捷重构策略寻优与仿真推演引擎关键技术研究", S_TITLE, 50, 15, 1050, 45),
        cell("t2s", "开始", S_START, cx-60, 75, 120, 40),
        cell("t2i", "输入：网络阻塞问题（线路过载 / 电压越限 / 分布式电源消纳受阻）", S_TASK, cx-240, 145, 480, 50),
        cell("t2a", "研究任务1：逻辑拓扑场景标准化建模&#xa;（仿真模块标准化 / 概率场景生成 / 多场景并行推演引擎构建）", S_TASK, cx-240, 225, 480, 65),
        cell("t2b", "研究任务2：高维拓扑搜索空间降维与目标分解&#xa;（图结构编码 / 全局-局部目标分层解耦 / 约束裁剪）", S_TASK, cx-240, 320, 480, 65),
        cell("t2c", "研究任务3：网架拓扑动态迭代重构技术&#xa;（分层强化学习求解 / 仿真结果反馈驱动 / 在线自适应优化）", S_TASK, cx-240, 415, 480, 65),
        cell("t2v", "仿真推演引擎验证（多场景并行推演 / 实时评估 / 动态迭代反馈）", S_SUB, cx-240, 510, 480, 50),
        cell("t2o", "输出：动态重构决策框架 + 韧性评估指标体系&#xa;→ 支撑课题三（协同控制）/ 课题五（示范应用）", S_OUT, cx-240, 590, 480, 55),
        cell("t2e", "结束", S_START, cx-60, 675, 120, 40),
    ]
    edges = [
        ("e2_0","t2s","t2i"), ("e2_1","t2i","t2a"), ("e2_2","t2a","t2b"),
        ("e2_3","t2b","t2c"), ("e2_4","t2c","t2v"), ("e2_5","t2v","t2o"), ("e2_6","t2o","t2e"),
    ]
    body = "\n".join(rows) + "\n" + "\n".join(edge(a,b,c) for a,b,c in edges)
    return diagram("t2", "课题二流程图", body)

# ============================================================
# Topic 3
# ============================================================
def topic3():
    cx = 450
    rows = [
        cell("t3h", "课题三：基于智能体群的配电网异质资源柔性协同控制策略仿真推演技术研究", S_TITLE, 50, 15, 1100, 45),
        cell("t3s", "开始", S_START, cx-60, 75, 120, 40),
        cell("t3i", "输入：课题一状态感知数据 + 课题二重构拓扑结构", S_TASK, cx-250, 145, 500, 45),
        cell("t3a", "研究任务1：典型局部拓扑单元抽象与参数化建模&#xa;（环网/星型/链式/T型 → 可组合标准化建模基础）", S_TASK, cx-250, 220, 500, 60),
        cell("t3b", "研究任务2：异质资源智能体群构建与多约束保障机制&#xa;（线路/DER/储能/可调负荷/无功设备智能体群&#xa;层级化决策模型 + 物理安全约束显式嵌入策略搜索）", S_TASK, cx-250, 310, 500, 80),
        cell("t3c", "研究任务3：基于图神经网络的跨拓扑泛化方法&#xa;（元学习策略初始化 / 自适应拓扑表征 / 少样本快速迁移）", S_TASK, cx-250, 420, 500, 60),
        cell("t3d", "研究任务4：域随机化协同仿真推演与可组合迁移&#xa;（多场景多参数扰动协同仿真 / 物理引擎闭环验证）", S_TASK, cx-250, 510, 500, 60),
        cell("t3o", "输出：可组合调度协控机制 + 控制策略推演结果&#xa;→ 支撑课题五（示范）/ 提供课题二重构执行支撑", S_OUT, cx-250, 600, 500, 55),
        cell("t3e", "结束", S_START, cx-60, 685, 120, 40),
    ]
    edges = [
        ("e3_0","t3s","t3i"), ("e3_1","t3i","t3a"), ("e3_2","t3a","t3b"),
        ("e3_3","t3b","t3c"), ("e3_4","t3c","t3d"), ("e3_5","t3d","t3o"), ("e3_6","t3o","t3e"),
    ]
    body = "\n".join(rows) + "\n" + "\n".join(edge(a,b,c) for a,b,c in edges)
    return diagram("t3", "课题三流程图", body)

# ============================================================
# Topic 4
# ============================================================
def topic4():
    cx = 450
    rows = [
        cell("t4h", "课题四：面向智能体集体幻觉抑制的大模型可信交互技术研究", S_TITLE, 50, 15, 1050, 45),
        cell("t4s", "开始", S_START, cx-60, 75, 120, 40),
        cell("t4a", "研究任务1：多智能体集群协作架构与幻觉传播机理研究&#xa;（智能体角色定义 / 任务分解 / 幻觉扩散路径量化表征）", S_TASK, cx-250, 145, 500, 65),
        cell("t4d", "决策输出是否通过&#xa;双层可信校验？", S_DEC, cx-90, 245, 180, 70),
        cell("t4hd", "硬约束校验&#xa;电气物理规则约束&#xa;（电压/潮流/容量合规性）", S_SUB, 80, 240, 200, 80),
        cell("t4sd", "软约束校验&#xa;AI for Science 潮流仿真&#xa;（科学计算驱动的软约束）", S_SUB, cx+185, 240, 210, 80),
        cell("t4b", "研究任务2：双层可信校验及拦截机制&#xa;（逐级合规验证 / 不可信输出快速识别与拦截&#xa;保障智能体群输出物理合规性与安全可靠性）", S_TASK, cx-250, 355, 500, 75),
        cell("t4c", "研究任务3：智能体可信协作与人机问责机制&#xa;（信任评估 / 协作权重动态调整 / 决策分级呈现&#xa;权限边界设定 / 全链路可审核可追踪可问责）", S_TASK, cx-250, 460, 500, 80),
        cell("t4o", "输出：可信多智能体协作框架 + 人机协同交互流程&#xa;→ 支撑课题五（示范应用）", S_OUT, cx-250, 570, 500, 55),
        cell("t4e", "结束", S_START, cx-60, 655, 120, 40),
    ]
    edges = [
        ("e4_0","t4s","t4a"), ("e4_1","t4a","t4d"),
        ("e4_2","t4d","t4hd"), ("e4_3","t4d","t4sd"),
        ("e4_4","t4hd","t4b"), ("e4_5","t4sd","t4b"),
        ("e4_6","t4b","t4c"), ("e4_7","t4c","t4o"), ("e4_8","t4o","t4e"),
    ]
    body = "\n".join(rows) + "\n" + "\n".join(edge(a,b,c) for a,b,c in edges)
    return diagram("t4", "课题四流程图", body)

# ============================================================
# Topic 5
# ============================================================
def topic5():
    cx = 450
    rows = [
        cell("t5h", "课题五：配网异质资源全景推演与可信决策一体化微应用框架 - 集成示范研究", S_TITLE, 50, 15, 1100, 45),
        cell("t5s", "开始", S_START, cx-60, 75, 120, 40),
        cell("t5i", "整合课题一至四核心成果（状态感知 / 动态重构 / 协同控制 / 可信校验）", S_TASK, cx-260, 145, 520, 50),
        cell("t5a", "研究任务1：设计一体化微应用框架&#xa;全域感知层 / 智能计算层 / 可信校验层&#xa;（三层分层解耦架构设计）", S_TASK, cx-260, 225, 520, 75),
        cell("t5b", "研究任务2：集成动态图模数据底座&#xa;（多源语义融合封装为实时数据服务层&#xa;自动接收量测 → 实时生成并更新动态拓扑图）", S_TASK, cx-260, 330, 520, 75),
        cell("t5c", "研究任务3：封装敏捷重构+协同控制策略推演核心引擎&#xa;（统一消息总线 / 标准交互协议&#xa;重构方案生成 → 控制指令推演 毫秒级闭环）", S_TASK, cx-260, 435, 520, 75),
        cell("t5d", "研究任务4：嵌入可信大模型人机交互中枢&#xa;（防幻觉大模型组件 / 自然语言决策建议&#xa;+ 严谨物理数据支撑 / 可信可靠输出）", S_TASK, cx-260, 540, 520, 75),
        cell("t5f", "研究任务5：研制原型系统并开展示范试点&#xa;（典型高渗透台区现场部署 / 稳定性与实用性验证）", S_TASK, cx-260, 645, 520, 65),
        cell("t5o", "输出：可复制推广的标准化实施方案 + 工程化成果", S_OUT, cx-260, 740, 520, 50),
        cell("t5e", "结束", S_START, cx-60, 820, 120, 40),
    ]
    edges = [
        ("e5_0","t5s","t5i"), ("e5_1","t5i","t5a"), ("e5_2","t5a","t5b"),
        ("e5_3","t5b","t5c"), ("e5_4","t5c","t5d"), ("e5_5","t5d","t5f"),
        ("e5_6","t5f","t5o"), ("e5_7","t5o","t5e"),
    ]
    body = "\n".join(rows) + "\n" + "\n".join(edge(a,b,c) for a,b,c in edges)
    return diagram("t5", "课题五流程图", body)

# ============================================================
# Write all files
# ============================================================
files = [
    ("课题一_多源异构数据融合与拓扑辨识.drawio",     topic1()),
    ("课题二_拓扑敏捷重构与仿真推演引擎.drawio",     topic2()),
    ("课题三_智能体群柔性协同控制仿真推演.drawio",   topic3()),
    ("课题四_大模型可信交互与幻觉抑制.drawio",       topic4()),
    ("课题五_集成示范与原型研制.drawio",             topic5()),
]

for fname, content in files:
    fpath = os.path.join(output_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    print("OK:", fname)

print("\nDone. Output dir:", output_dir)
