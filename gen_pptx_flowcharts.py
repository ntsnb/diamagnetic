"""
Generate WPS-editable flowchart .pptx files for each research sub-topic.
Uses python-pptx: shapes (rectangle/diamond/rounded), connectors, colors.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
import pptx.oxml as oxml
from lxml import etree
from pptx.enum.dml import MSO_THEME_COLOR

OUTPUT = r"C:\Users\admin\Desktop\项目流程图_WPS"
os.makedirs(OUTPUT, exist_ok=True)

# Slide size: widescreen 33.87cm x 19.05cm
SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

# Colors
C_START  = RGBColor(0x84, 0xbe, 0x77)   # green
C_TASK   = RGBColor(0x5b, 0x9b, 0xd5)   # blue
C_SUB    = RGBColor(0xff, 0xc0, 0x00)   # yellow
C_OUT    = RGBColor(0xe0, 0x50, 0x50)   # red
C_DEC    = RGBColor(0xaf, 0x6f, 0xc1)   # purple
C_WHITE  = RGBColor(0xff, 0xff, 0xff)
C_TITLE  = RGBColor(0x1f, 0x39, 0x64)   # dark blue
C_BORDER = RGBColor(0x40, 0x40, 0x40)

def new_prs():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs

def blank_slide(prs):
    layout = prs.slide_layouts[6]  # blank
    return prs.slides.add_slide(layout)

def add_title(slide, text):
    """Add a title text box at the top of the slide."""
    tf = slide.shapes.add_textbox(
        Inches(0.2), Inches(0.1), SLIDE_W - Inches(0.4), Inches(0.55)
    )
    tf.text_frame.word_wrap = True
    p = tf.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.bold = True
    run.font.size = Pt(16)
    run.font.color.rgb = C_TITLE
    return tf

def add_rect(slide, text, x, y, w, h, fill=C_TASK, font_size=10, bold=False):
    """Add a rounded rectangle shape with text."""
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE = 1; rounded = use autoshape
        x, y, w, h
    )
    # Make it a rounded rectangle via XML
    sp = shape.element
    prstGeom = sp.find('.//' + qn('a:prstGeom'))
    if prstGeom is not None:
        prstGeom.set('prst', 'roundRect')
        avLst = prstGeom.find(qn('a:avLst'))
        if avLst is None:
            avLst = etree.SubElement(prstGeom, qn('a:avLst'))
        gd = etree.SubElement(avLst, qn('a:gd'))
        gd.set('name', 'adj')
        gd.set('fmla', 'val 30000')

    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = C_BORDER
    shape.line.width = Pt(0.75)

    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None

    from pptx.util import Pt as pt_util
    from pptx.oxml.ns import qn as qn2
    # Center text vertically
    tf.margin_top    = Pt(4)
    tf.margin_bottom = Pt(4)
    tf.margin_left   = Pt(6)
    tf.margin_right  = Pt(6)

    for i, line in enumerate(text.split('\n')):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = line
        run.font.size = Pt(font_size)
        run.font.color.rgb = C_WHITE
        if bold or i == 0:
            run.font.bold = True if i == 0 else False

    return shape

def add_diamond(slide, text, x, y, w, h, fill=C_DEC, font_size=9):
    """Add a diamond shape."""
    shape = slide.shapes.add_shape(4, x, y, w, h)  # 4 = diamond
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = C_BORDER
    shape.line.width = Pt(0.75)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_top = tf.margin_bottom = Pt(2)
    for i, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = line
        run.font.size = Pt(font_size)
        run.font.bold = True
        run.font.color.rgb = C_WHITE
    return shape

def add_connector(slide, x1, y1, x2, y2):
    """Add a straight arrow connector from (x1,y1) to (x2,y2)."""
    connector = slide.shapes.add_connector(1, x1, y1, x2, y2)  # 1=straight
    connector.line.color.rgb = RGBColor(0x40, 0x40, 0x40)
    connector.line.width = Pt(1.5)
    return connector

def center_bottom(shape):
    return (shape.left + shape.width // 2, shape.top + shape.height)

def center_top(shape):
    return (shape.left + shape.width // 2, shape.top)

def center_left(shape):
    return (shape.left, shape.top + shape.height // 2)

def center_right(shape):
    return (shape.left + shape.width, shape.top + shape.height // 2)

def connect_tb(slide, s1, s2):
    """Connect bottom of s1 to top of s2."""
    x1, y1 = center_bottom(s1)
    x2, y2 = center_top(s2)
    add_connector(slide, x1, y1, x2, y2)

def inches(v):
    return Inches(v)

# ============================================================
# Topic 1
# ============================================================
def gen_topic1(prs):
    slide = blank_slide(prs)
    add_title(slide, "课题一：配电网多源异构数据语义融合与拓扑动态演化辨识技术研究")

    cx = SLIDE_W / 2
    sw, sh = inches(7.5), inches(0.55)
    sx = cx - sw / 2
    gap = inches(0.15)

    shapes = []
    ys = [inches(0.72), inches(1.38), inches(2.04), inches(2.72), inches(3.40), inches(4.10), inches(4.78), inches(5.48)]

    defs = [
        ("开始",   C_START, True),
        ("多源异构数据输入\n（量测终端 / 自动化系统 / 保护装置 / 设备监控系统）", C_TASK, False),
        ("研究任务1：多源异构数据融合建模与语义统一表达\n（语义对齐方法 / 标准化场景资源库构建）", C_TASK, False),
        ("研究任务2：多源数据时空对齐与图模数协同建模\n（时空对齐 / 缺失修复 / 动态增量关联更新机制）", C_TASK, False),
        ("研究任务3：数据驱动与机理融合的拓扑状态实时辨识\n（多模态融合 / 物理机理约束 / 快速拓扑辨识方法）", C_TASK, False),
        ("研究任务4：拓扑动态演化跟踪与场景切换识别\n（演化过程建模 / 在线跟踪 / 全局状态迁移感知）", C_TASK, False),
        ("输出：统一场景资源体系 + 在线状态输入体系\n→ 支撑课题二（重构）/ 课题三（协控）/ 课题五（示范）", C_OUT, False),
        ("结束",   C_START, True),
    ]

    for i, (text, color, bold) in enumerate(defs):
        h = inches(0.5) if i in (0, 7) else inches(0.58)
        w = inches(3.5) if i in (0, 7) else sw
        x = cx - w / 2
        s = add_rect(slide, text, x, ys[i], w, h, fill=color, font_size=9, bold=bold)
        shapes.append(s)

    for i in range(len(shapes) - 1):
        connect_tb(slide, shapes[i], shapes[i+1])

    return slide

# ============================================================
# Topic 2
# ============================================================
def gen_topic2(prs):
    slide = blank_slide(prs)
    add_title(slide, "课题二：配电网拓扑敏捷重构策略寻优与仿真推演引擎关键技术研究")

    cx = SLIDE_W / 2
    sw, sh = inches(8.0), inches(0.58)
    sx = cx - sw / 2

    ys = [inches(0.72), inches(1.36), inches(2.02), inches(2.70), inches(3.38), inches(4.06), inches(4.72), inches(5.40)]

    defs = [
        ("开始", C_START),
        ("输入：网络阻塞问题（线路过载 / 电压越限 / 分布式电源消纳受阻）", C_TASK),
        ("研究任务1：逻辑拓扑场景标准化建模\n（仿真模块标准化 / 概率场景生成 / 多场景并行推演引擎构建）", C_TASK),
        ("研究任务2：高维拓扑搜索空间降维与目标分解\n（图结构编码 / 全局-局部目标分层解耦 / 约束裁剪机制）", C_TASK),
        ("研究任务3：网架拓扑动态迭代重构技术\n（分层强化学习求解 / 仿真结果反馈驱动 / 在线自适应优化）", C_TASK),
        ("仿真推演引擎验证\n（多场景并行推演 / 实时评估 / 动态迭代反馈闭环）", C_SUB),
        ("输出：动态重构决策框架 + 韧性评估指标体系\n→ 支撑课题三（协同控制）/ 课题五（示范应用）", C_OUT),
        ("结束", C_START),
    ]

    shapes = []
    for i, (text, color) in enumerate(defs):
        w = inches(3.5) if i in (0, 7) else sw
        h = inches(0.5) if i in (0, 7) else sh
        x = cx - w / 2
        s = add_rect(slide, text, x, ys[i], w, h, fill=color, font_size=9)
        shapes.append(s)

    for i in range(len(shapes) - 1):
        connect_tb(slide, shapes[i], shapes[i+1])

    return slide

# ============================================================
# Topic 3
# ============================================================
def gen_topic3(prs):
    slide = blank_slide(prs)
    add_title(slide, "课题三：基于智能体群的配电网异质资源柔性协同控制策略仿真推演技术研究")

    cx = SLIDE_W / 2
    sw = inches(8.5)
    sx = cx - sw / 2

    ys = [inches(0.72), inches(1.34), inches(1.95), inches(2.62), inches(3.38), inches(4.06), inches(4.72), inches(5.40)]

    defs = [
        ("开始", C_START),
        ("输入：课题一状态感知数据 + 课题二重构拓扑结构", C_TASK),
        ("研究任务1：典型局部拓扑单元抽象与参数化建模\n（环网/星型/链式/T型 → 可组合标准化建模基础）", C_TASK),
        ("研究任务2：异质资源智能体群构建与多约束保障机制\n（线路/DER/储能/可调负荷/无功设备智能体群\n层级化决策模型 + 物理安全约束显式嵌入策略搜索）", C_TASK),
        ("研究任务3：基于图神经网络的跨拓扑泛化方法\n（元学习策略初始化 / 自适应拓扑表征 / 少样本快速迁移）", C_TASK),
        ("研究任务4：域随机化协同仿真推演与可组合迁移\n（多场景多参数扰动协同仿真 / 物理引擎闭环验证）", C_TASK),
        ("输出：可组合调度协控机制 + 控制策略推演结果\n→ 支撑课题五（示范）/ 提供课题二重构执行支撑", C_OUT),
        ("结束", C_START),
    ]

    shapes = []
    for i, (text, color) in enumerate(defs):
        w = inches(3.5) if i in (0, 7) else sw
        h = inches(0.5)  if i in (0, 7) else (inches(0.68) if i == 3 else inches(0.58))
        x = cx - w / 2
        s = add_rect(slide, text, x, ys[i], w, h, fill=color, font_size=9)
        shapes.append(s)

    for i in range(len(shapes) - 1):
        connect_tb(slide, shapes[i], shapes[i+1])

    return slide

# ============================================================
# Topic 4  (with diamond decision node)
# ============================================================
def gen_topic4(prs):
    slide = blank_slide(prs)
    add_title(slide, "课题四：面向智能体集体幻觉抑制的大模型可信交互技术研究")

    cx = SLIDE_W / 2
    sw = inches(8.5)

    # Main column shapes
    s_start = add_rect(slide, "开始", cx - inches(1.75), inches(0.72), inches(3.5), inches(0.5), fill=C_START, font_size=10)
    s_t1    = add_rect(slide,
        "研究任务1：多智能体集群协作架构与幻觉传播机理研究\n（智能体角色定义 / 任务分解 / 幻觉扩散路径量化表征）",
        cx - sw/2, inches(1.34), sw, inches(0.58), fill=C_TASK, font_size=9)

    # Decision diamond
    dw, dh = inches(2.8), inches(0.9)
    s_dec = add_diamond(slide, "决策输出\n是否通过双层校验？",
        cx - dw/2, inches(2.05), dw, dh, fill=C_DEC, font_size=9)

    # Left branch: hard constraint
    hw, hh = inches(3.2), inches(0.75)
    s_hard = add_rect(slide,
        "硬约束校验\n电气物理规则约束\n（电压/潮流/容量合规性）",
        inches(0.3), inches(2.1), hw, hh, fill=C_SUB, font_size=9)

    # Right branch: soft constraint
    s_soft = add_rect(slide,
        "软约束校验\nAI for Science 潮流仿真\n（科学计算驱动的软约束）",
        SLIDE_W - hw - inches(0.3), inches(2.1), hw, hh, fill=C_SUB, font_size=9)

    s_t2   = add_rect(slide,
        "研究任务2：双层可信校验及拦截机制\n（逐级合规验证 / 不可信输出快速识别与拦截 / 物理合规保障）",
        cx - sw/2, inches(3.02), sw, inches(0.58), fill=C_TASK, font_size=9)
    s_t3   = add_rect(slide,
        "研究任务3：智能体可信协作与人机问责机制\n（信任评估 / 协作权重动态调整 / 决策分级呈现\n权限边界设定 / 全链路可审核可追踪可问责）",
        cx - sw/2, inches(3.72), sw, inches(0.68), fill=C_TASK, font_size=9)
    s_out  = add_rect(slide,
        "输出：可信多智能体协作框架 + 人机协同交互流程\n→ 支撑课题五（示范应用）",
        cx - sw/2, inches(4.52), sw, inches(0.58), fill=C_OUT, font_size=9)
    s_end  = add_rect(slide, "结束", cx - inches(1.75), inches(5.22), inches(3.5), inches(0.5), fill=C_START, font_size=10)

    # Connectors
    connect_tb(slide, s_start, s_t1)
    connect_tb(slide, s_t1, s_dec)
    # dec -> hard (left)
    add_connector(slide, *center_left(s_dec), *center_right(s_hard))
    # dec -> soft (right)
    add_connector(slide, *center_right(s_dec), *center_left(s_soft))
    # hard -> t2
    add_connector(slide, *center_bottom(s_hard), *center_top(s_t2))
    # soft -> t2
    add_connector(slide, *center_bottom(s_soft), *center_top(s_t2))
    connect_tb(slide, s_t2, s_t3)
    connect_tb(slide, s_t3, s_out)
    connect_tb(slide, s_out, s_end)

    return slide

# ============================================================
# Topic 5
# ============================================================
def gen_topic5(prs):
    slide = blank_slide(prs)
    add_title(slide, "课题五：配网异质资源全景推演与可信决策一体化微应用框架 — 集成示范研究")

    cx = SLIDE_W / 2
    sw = inches(8.5)
    sx = cx - sw / 2

    ys = [inches(0.72), inches(1.32), inches(1.95), inches(2.62), inches(3.30), inches(3.98), inches(4.68), inches(5.36), inches(5.98)]

    defs = [
        ("开始", C_START),
        ("整合课题一至四核心成果（状态感知 / 动态重构 / 协同控制 / 可信校验）", C_TASK),
        ("研究任务1：设计一体化微应用框架\n全域感知层 / 智能计算层 / 可信校验层（三层分层解耦架构）", C_TASK),
        ("研究任务2：集成「动态图模」数据底座\n（多源语义融合封装实时数据服务层 / 自动接收量测 → 实时生成动态拓扑图）", C_TASK),
        ("研究任务3：封装「敏捷重构+协同控制策略推演」核心引擎\n（统一消息总线 / 标准交互协议 / 毫秒级重构-控制闭环推演）", C_TASK),
        ("研究任务4：嵌入「可信大模型」人机交互中枢\n（防幻觉大模型组件 / 自然语言决策建议 + 严谨物理数据支撑）", C_TASK),
        ("研究任务5：研制原型系统并开展示范试点\n（典型高渗透台区现场部署 / 验证稳定性与实用性）", C_TASK),
        ("输出：可复制推广的标准化实施方案 + 工程化成果", C_OUT),
        ("结束", C_START),
    ]

    shapes = []
    for i, (text, color) in enumerate(defs):
        w = inches(3.5) if i in (0, 8) else sw
        h = inches(0.5) if i in (0, 8) else inches(0.58)
        x = cx - w / 2
        s = add_rect(slide, text, x, ys[i], w, h, fill=color, font_size=9)
        shapes.append(s)

    for i in range(len(shapes) - 1):
        connect_tb(slide, shapes[i], shapes[i+1])

    return slide

# ============================================================
# Main: build one pptx per topic
# ============================================================
topics = [
    ("课题一_多源异构数据融合与拓扑辨识.pptx",   gen_topic1),
    ("课题二_拓扑敏捷重构与仿真推演引擎.pptx",   gen_topic2),
    ("课题三_智能体群柔性协同控制仿真推演.pptx", gen_topic3),
    ("课题四_大模型可信交互与幻觉抑制.pptx",     gen_topic4),
    ("课题五_集成示范与原型研制.pptx",           gen_topic5),
]

for fname, fn in topics:
    prs = new_prs()
    fn(prs)
    path = os.path.join(OUTPUT, fname)
    prs.save(path)
    print("OK:", fname)

print("\nDone. Output:", OUTPUT)
