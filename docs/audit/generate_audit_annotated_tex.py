from pathlib import Path


SRC = Path("bare_jrnl_new_sample4.tex")
OUT = Path("bare_jrnl_new_sample4_audit_annotated.tex")


def add(replacements: list[tuple[str, str]], old: str, new: str) -> None:
    replacements.append((old, new))


def main() -> None:
    text = SRC.read_text(encoding="utf-8")

    preamble_old = r"""\usepackage{placeins}
\usepackage{balance}"""
    preamble_new = r"""\usepackage{placeins}
\usepackage{balance}
\ifXeTeX
  \usepackage{xeCJK}
  \setCJKmainfont{SimSun}
\fi
\usepackage{xcolor}
\definecolor{auditred}{RGB}{190,0,0}
\DeclareRobustCommand{\auditbad}[2]{{\color{auditred}#1}{\color{auditred}\textsuperscript{[#2]}}}
\DeclareRobustCommand{\auditnote}[2]{{\color{auditred}\footnotesize\,[#1：#2]}}
\newcommand{\auditid}[1]{\textcolor{auditred}{\textbf{[#1]}}}"""
    if preamble_old not in text:
        raise SystemExit("preamble insertion point not found")
    text = text.replace(preamble_old, preamble_new, 1)

    replacements: list[tuple[str, str]] = []

    add(replacements, r"\title{A Sample Article Using IEEEtran.cls\\ for IEEE Journals and Transactions}",
        r"\title{\auditbad{A Sample Article Using IEEEtran.cls\\ for IEEE Journals and Transactions}{P01}}")
    add(replacements, r"\author{IEEE Publication Technology,~\IEEEmembership{Staff,~IEEE,}",
        r"\author{\auditbad{IEEE Publication Technology,~\IEEEmembership{Staff,~IEEE,}}{P01}")
    add(replacements, r"\thanks{This paper was produced by the IEEE Publication Technology Group. They are in Piscataway, NJ.}% <-this % stops a space",
        r"\thanks{\auditbad{This paper was produced by the IEEE Publication Technology Group. They are in Piscataway, NJ.}{P01}}% <-this % stops a space")
    add(replacements, r"\thanks{Manuscript received April 19, 2021; revised August 16, 2021.}}",
        r"\thanks{\auditbad{Manuscript received April 19, 2021; revised August 16, 2021.}{P01}}}")
    add(replacements, r"\markboth{Journal of \LaTeX\ Class Files,~Vol.~14, No.~8, August~2021}%",
        r"\markboth{\auditbad{Journal of \LaTeX\ Class Files,~Vol.~14, No.~8, August~2021}{P01}}%")
    add(replacements, r"{Shell \MakeLowercase{\textit{et al.}}: A Sample Article Using IEEEtran.cls for IEEE Journals}",
        r"{\auditbad{Shell \MakeLowercase{\textit{et al.}}: A Sample Article Using IEEEtran.cls for IEEE Journals}{P01}}")
    add(replacements, r"\IEEEpubid{0000--0000/00\$00.00~\copyright~2021 IEEE}",
        r"\IEEEpubid{\auditbad{0000--0000/00\$00.00~\copyright~2021 IEEE}{P01}}")

    add(replacements, "This paper studies online multivariate regional load forecasting under delayed feedback and distribution shift.",
        r"This paper studies online multivariate regional load forecasting under \auditbad{delayed feedback}{P03} and distribution shift.")
    add(replacements, "The framework separates stable predictive structure from variant fluctuations, performs relevance-guided sparse updates only when experts become unstable, and applies transient drift compensation at inference time.",
        r"The framework separates stable predictive structure from variant fluctuations, \auditbad{performs relevance-guided sparse updates only when experts become unstable}{P13}, and applies transient drift compensation at inference time.")
    add(replacements, "Extensive experiments on multiple load forecasting datasets and multiple transformer-based backbones show that the proposed framework consistently improves over frozen deployment and representative online baselines under shifted online streams.",
        r"Extensive experiments on multiple load forecasting datasets and multiple transformer-based backbones show that the proposed framework \auditbad{consistently improves over frozen deployment and representative online baselines}{P04} under shifted online streams.\auditnote{P04}{需改为证据范围内的表述：相对 Frozen 有增益、对在线基线为 competitive，不能写成普遍超越。}")
    add(replacements, "online load forecasting, multivariate time series, distribution shift, invariant learning, transformer backbone, delayed feedback.",
        r"online load forecasting, multivariate time series, distribution shift, invariant learning, transformer backbone, \auditbad{delayed feedback}{P03}.")
    add(replacements, "naive online fine-tuning can be unstable under delayed labels and concept drift.",
        r"naive online fine-tuning can be unstable under \auditbad{delayed labels}{P03} and concept drift.")
    add(replacements, "Stage~I recalibrates the factor mixture under delayed labels, Stage~II performs relevance-guided sparse expert updates when instability is detected, and Stage~III applies transient correction based on the discrepancy between variant and invariant representations.",
        r"Stage~I recalibrates the factor mixture under \auditbad{delayed labels}{P03}, Stage~II \auditbad{performs relevance-guided sparse expert updates when instability is detected}{P13}, and Stage~III applies transient correction based on the discrepancy between variant and invariant representations.")
    add(replacements, "The same online controller can be combined with multiple transformer-based backbones",
        r"The same online controller can be combined with \auditbad{multiple transformer-based backbones}{P23}")
    add(replacements, "standard training minimizes the empirical risk over the training set",
        r"standard training minimizes the empirical risk over the training set\auditnote{P08}{此处集合上标仍用 N，但前文 N 已定义为变量数；训练样本数建议改为 M 或 $T_s$。}")
    add(replacements, "due to the feedback delay in Section~1, updates can only use labels arriving at least $L$ steps later",
        r"due to the feedback delay in \auditbad{Section~1}{P08}, updates can only use labels arriving at least \auditbad{$L$ steps later}{P08}")
    add(replacements, "Structural Causal Model (SCM) is introduced to represent latent drivers and causal mechanisms, providing a principled basis for reasoning.",
        r"\auditbad{Structural Causal Model (SCM) is introduced to represent latent drivers and causal mechanisms, providing a principled basis for reasoning}{P10}.")
    add(replacements, "The induced directed graph encodes direct causal effects.",
        r"\auditbad{The induced directed graph encodes direct causal effects.}{P10}")
    add(replacements, r"\textbf{RQ1:} ", r"\auditbad{\textbf{RQ1:}}{P06} ")
    add(replacements, r"\textbf{RQ2:} ", r"\auditbad{\textbf{RQ2:}}{P06} ")
    add(replacements, r"\textbf{RQ3:} ", r"\auditbad{\textbf{RQ3:}}{P06} ")
    add(replacements, "finally the two-stage online adaptation.", r"finally the \auditbad{two-stage online adaptation}{P05}.")
    add(replacements, r"\subsection{The decomposition of ... using Invariant Learning}",
        r"\subsection{\auditbad{The decomposition of ... using Invariant Learning}{P07}}")
    add(replacements, "The following invariance assumption is adopted:",
        r"The following \auditbad{invariance assumption}{P11} is adopted:")
    add(replacements, r"IRN ensures the residuals have mean 0 and variance $2-2\mathrm{cov}(\widehat{Y},Y)$",
        r"\auditbad{IRN ensures the residuals have mean 0 and variance $2-2\mathrm{cov}(\widehat{Y},Y)$}{P11}")
    add(replacements, r"an adversarial pair $\mathcal{M}_{\mathrm{EI}}$ and $\mathcal{M}_{\mathrm{IL}}$ is optimized in an alternating manner",
        r"\auditbad{an adversarial pair $\mathcal{M}_{\mathrm{EI}}$ and $\mathcal{M}_{\mathrm{IL}}$ is optimized in an alternating manner}{P12}")
    add(replacements, r"and optionally apply a temporal smoothing step via majority voting over a radius $r$",
        r"and optionally apply a \auditbad{temporal smoothing step via majority voting over a radius $r$}{P12}")
    add(replacements, r"\subsection{C}", r"\subsection{\auditbad{C}{P07}}")
    add(replacements, r"\subsection{D}", r"\subsection{\auditbad{D}{P07}}")
    add(replacements, "We now present the two-stage online adaptation",
        r"We now present the \auditbad{two-stage online adaptation}{P05}")
    add(replacements, "we use a parameter-efficient adaptation generator",
        r"we use a \auditbad{parameter-efficient adaptation generator}{P15}")
    add(replacements, "The same parameter-efficient generator is applied to both embeddings, and the resulting parameter shifts are optimized using the post-update prediction error.",
        r"The same \auditbad{parameter-efficient generator is applied to both embeddings}{P15}, and the resulting parameter shifts are \auditbad{optimized using the post-update prediction error}{P13}.")
    add(replacements, "The subsequent parameter update follows the same parameter-efficient adaptation generator as in Stage~II",
        r"The subsequent parameter update follows the same \auditbad{parameter-efficient adaptation generator as in Stage~II}{P15}")
    add(replacements, "The generator parameters are then updated using the prediction error of the adapted model",
        r"The \auditbad{generator parameters are then updated using the prediction error of the adapted model}{P15}")
    add(replacements, r"\subsection{E}", r"\subsection{\auditbad{E}{P07}}")
    add(replacements, "whether each online stage improves over a frozen deployment baseline under delayed feedback",
        r"whether each online stage improves over a frozen deployment baseline under \auditbad{delayed feedback}{P03}")
    add(replacements, "whether combining Stage~I, Stage~II, and Stage~III yields the strongest overall online performance.",
        r"whether combining Stage~I, Stage~II, and Stage~III yields the \auditbad{strongest overall online performance}{P25}.")
    add(replacements, "No random shuffling is applied across the temporal axis",
        r"\auditbad{No random shuffling is applied across the temporal axis}{P17}")
    add(replacements, "The remaining suffix is reserved for online evaluation under a shifted-stream protocol described later.",
        r"The remaining suffix is reserved for online evaluation under a \auditbad{shifted-stream protocol}{P18} described later.")
    add(replacements, r"Train/validation/online-test = 55\%/10\%/35\%",
        r"\auditbad{Train/validation/online-test = 55\%/10\%/35\%}{P18}")
    add(replacements, "Factor generator & Latent dimension 64; VAE gate in the default model",
        r"Factor generator & Latent dimension 64; \auditbad{VAE gate in the default model}{P26}")
    add(replacements, "Labels arrive with delayed feedback, and therefore parameter updates are only triggered after the corresponding delayed target becomes available.",
        r"Labels arrive with \auditbad{delayed feedback}{P03}, and therefore parameter updates are only triggered after the corresponding delayed target becomes available.\auditnote{P13}{代码审计显示 Stage II/Full 接受门存在当前标签使用风险，需修协议或标成 oracle-assisted。}")
    add(replacements, "Stage~II is triggered only when the relevance-map stability score indicates that an expert becomes unstable",
        r"\auditbad{Stage~II is triggered only when the relevance-map stability score indicates that an expert becomes unstable}{P13}")
    add(replacements, "whether each individual stage improves over the baseline",
        r"whether \auditbad{each individual stage improves over the baseline}{P25}")
    add(replacements, "whether the joint Stage~I+II+III configuration achieves the best overall online forecasting performance.",
        r"whether the joint Stage~I+II+III configuration achieves the \auditbad{best overall online forecasting performance}{P25}.")
    add(replacements, "we search within the later candidate region for a contiguous online test block that maximizes the distribution-shift score",
        r"we \auditbad{search within the later candidate region for a contiguous online test block that maximizes the distribution-shift score}{P18}")
    add(replacements, "differ substantially from the offline training distribution",
        r"differ \auditbad{substantially}{P20} from the offline training distribution")
    add(replacements, r"The remaining four are \emph{representative online forecasting baselines}",
        r"The remaining four are \auditbad{\emph{representative online forecasting baselines}}{P20}")
    add(replacements, r"\textbf{SOLID}: uses self-supervised online adaptation under drift.",
        r"\textbf{SOLID}: \auditbad{uses self-supervised online adaptation under drift}{P20}.")
    add(replacements, r"\textbf{SOLID++}: strengthens SOLID with a more aggressive online controller.",
        r"\textbf{SOLID++}: \auditbad{strengthens SOLID with a more aggressive online controller}{P20}.")
    add(replacements, r"\textbf{PROCEED}: performs proactive drift-aware online adaptation.",
        r"\textbf{PROCEED}: \auditbad{performs proactive drift-aware online adaptation}{P20}.")
    add(replacements, "representative forecasting-time adaptation baselines",
        r"\auditbad{representative forecasting-time adaptation baselines}{P20}")
    add(replacements, "the proposed controller remains competitive across all four datasets",
        r"the proposed controller remains \auditbad{competitive across all four datasets}{P21}")
    add(replacements, r"both Ours-Stage~III and Ours-Full consistently outperform \textbf{Frozen}, \textbf{GD}, \textbf{Head-GD}, \textbf{OneNet}, and \textbf{PROCEED} on the \textit{Lishui-9} family",
        r"both Ours-Stage~III and Ours-Full \auditbad{consistently outperform \textbf{Frozen}, \textbf{GD}, \textbf{Head-GD}, \textbf{OneNet}, and \textbf{PROCEED} on the \textit{Lishui-9} family}{P21}")
    add(replacements, r"\caption{Comparison with online forecasting baselines under the shifted-stream protocol on the iTransformer backbone.",
        r"\caption{Comparison with online forecasting baselines under the \auditbad{shifted-stream protocol}{P18} on the iTransformer backbone.")
    add(replacements, "the remaining methods are forecasting-time adaptation baselines",
        r"the remaining methods are \auditbad{forecasting-time adaptation baselines}{P20}")
    add(replacements, r"and 0.3027 on \textit{Lishui-9}",
        r"and \auditbad{0.3027}{P18} on \textit{Lishui-9}")
    add(replacements, r"SOLID & 0.2019 & 0.3060 & 0.0893 & 0.2030 & 0.2518 & 0.3639 & 0.2762 & 0.3966 \\",
        r"SOLID & 0.2019 & 0.3060 & 0.0893 & 0.2030 & 0.2518 & 0.3639 & \auditbad{0.2762}{P21} & \auditbad{0.3966}{P21} \\")
    add(replacements, r"SOLID++ & 0.4500 & 0.4913 & 0.0940 & 0.2097 & 0.2992 & 0.4015 & 0.2757 & 0.3964 \\",
        r"SOLID++ & 0.4500 & 0.4913 & 0.0940 & 0.2097 & 0.2992 & 0.4015 & \auditbad{0.2757}{P21} & \auditbad{0.3964}{P21} \\")
    add(replacements, r"Ours-Stage~III & 0.1977 & 0.3005 & 0.0824 & 0.1907 & 0.2415 & 0.3540 & 0.2778 & 0.3972 \\",
        r"Ours-Stage~III & 0.1977 & 0.3005 & 0.0824 & 0.1907 & 0.2415 & 0.3540 & \auditbad{0.2778}{P21} & \auditbad{0.3972}{P21} \\")
    add(replacements, r"Ours-Full & 0.1974 & 0.3005 & 0.0820 & 0.1901 & 0.2417 & 0.3535 & 0.2775 & 0.3969 \\",
        r"Ours-Full & 0.1974 & 0.3005 & 0.0820 & 0.1901 & 0.2417 & 0.3535 & \auditbad{0.2775}{P21} & \auditbad{0.3969}{P21} \\")

    add(replacements, "under a shared configuration and a shared online protocol",
        r"under a \auditbad{shared configuration and a shared online protocol}{P23}")
    add(replacements, "consistent online gains across all screened backbones",
        r"\auditbad{consistent online gains across all screened backbones}{P23}")
    add(replacements, "with a clear margin", r"with a \auditbad{clear margin}{P20}")
    add(replacements, r"\caption{Broad backbone screening on the three main load-forecasting datasets.",
        r"\caption{Broad backbone screening on the three main load-forecasting datasets.\auditnote{P23}{表格需绑定唯一 CSV/配置/checkpoint/seed 清单；当前只能中等置信追踪。}")
    add(replacements, "Reported numbers are means", r"\auditbad{Reported numbers are means}{P22}")
    add(replacements, r"\caption{Mean online MSE of the frozen baseline, Stage~III, and the full Stage~I+II+III pipeline across eight plug-in transformer backbones.}",
        r"\caption{Mean online MSE of the frozen baseline, Stage~III, and the full Stage~I+II+III pipeline across eight plug-in transformer backbones.\auditnote{P23}{图源存在，但缺少精确生成命令、checkpoint 和 run manifest。}}")

    add(replacements, "under the tuned online settings", r"under the \auditbad{tuned online settings}{P24}")
    add(replacements, "Stage~III remains the most reliable single online component across all eight backbones",
        r"\auditbad{Stage~III remains the most reliable single online component across all eight backbones}{P24}")
    add(replacements, "Stage~II is generally weaker than Stage~III",
        r"\auditbad{Stage~II is generally weaker than Stage~III}{P24}")
    add(replacements, r"\caption{Online variant comparison across eight transformer-based backbones under the tuned controller settings.",
        r"\caption{Online variant comparison across eight transformer-based backbones under the \auditbad{tuned controller settings}{P24}.")
    add(replacements, r"FEDformer & 0.2024 & 0.1892 & 0.1995 & 0.1995 \\",
        r"FEDformer & 0.2024 & \auditbad{0.1892}{P24} & 0.1995 & 0.1995 \\")
    add(replacements, r"Crossformer & 0.2442 & 0.1965 & 0.2412 & 0.2412 \\",
        r"Crossformer & 0.2442 & \auditbad{0.1965}{P24} & 0.2412 & 0.2412 \\")
    add(replacements, r"Informer & 0.3465 & 0.3048 & 0.3440 & 0.3436 \\",
        r"Informer & 0.3465 & \auditbad{0.3048}{P24} & 0.3440 & 0.3436 \\")
    add(replacements, r"Autoformer & 0.3728 & 0.3472 & 0.3704 & 0.3708 \\",
        r"Autoformer & 0.3728 & \auditbad{0.3472}{P24} & 0.3704 & 0.3708 \\")
    add(replacements, r"Pyraformer & 0.5554 & 0.5175 & 0.5535 & 0.5536 \\",
        r"Pyraformer & 0.5554 & \auditbad{0.5175}{P24} & 0.5535 & 0.5536 \\")
    add(replacements, r"Transformer & 0.5781 & 0.5631 & 0.5771 & 0.5770 \\",
        r"Transformer & 0.5781 & \auditbad{0.5631}{P24} & 0.5771 & 0.5770 \\")
    add(replacements, r"PatchTST & 0.6632 & 0.6510 & 0.6626 & 0.6628 \\",
        r"PatchTST & 0.6632 & \auditbad{0.6510}{P24} & 0.6626 & 0.6628 \\")
    add(replacements, r"\caption{Variant-level online comparison across eight transformer-based backbones under the tuned controller settings.}",
        r"\caption{Variant-level online comparison across eight transformer-based backbones under the \auditbad{tuned controller settings}{P24}.\auditnote{P24}{表图均需重建单一受控版本后才能支持 Stage II/III 排名结论。}}")

    add(replacements, "This confirms that Stage~II becomes beneficial once it operates inside the complete controller",
        r"\auditbad{This confirms that Stage~II becomes beneficial once it operates inside the complete controller}{P25}")
    add(replacements, r"Stage~I+II+III & 0.1974 & \textbf{0.0821} & \textbf{0.2411} & 0.2256 \\",
        r"Stage~I+II+III & \auditbad{0.1974}{P25} & \textbf{0.0821} & \textbf{0.2411} & \auditbad{0.2256}{P25} \\")
    add(replacements, r"\caption{Ablation on online stage combinations under the shifted-stream protocol.",
        r"\caption{Ablation on online stage combinations under the \auditbad{shifted-stream protocol}{P18}.")
    add(replacements, r"\caption{Stage-combination ablation under the shifted-stream protocol.",
        r"\caption{Stage-combination ablation under the \auditbad{shifted-stream protocol}{P18}.")

    add(replacements, "Removing RevIN or collapsing the multi-expert factor bank causes the largest degradation on both the public suite and the Lishui family",
        r"\auditbad{Removing RevIN or collapsing the multi-expert factor bank causes the largest degradation on both the public suite and the Lishui family}{P26}")
    add(replacements, "Removing the invariant branch also degrades the result on most datasets",
        r"\auditbad{Removing the invariant branch also degrades the result on most datasets}{P26}")
    add(replacements, r"MLP Gate, no VAE & \textbf{0.1813} & \textbf{0.0744} & \textbf{0.2408} & \textbf{0.2745} \\",
        r"\auditbad{MLP Gate, no VAE}{P26} & \auditbad{\textbf{0.1813}}{P26} & \auditbad{\textbf{0.0744}}{P26} & \auditbad{\textbf{0.2408}}{P26} & \auditbad{\textbf{0.2745}}{P26} \\")
    add(replacements, r"w/o Invariant & 0.2025 & 0.0852 & 0.2354 & 0.2816 \\",
        r"w/o Invariant & 0.2025 & 0.0852 & \auditbad{0.2354}{P26} & 0.2816 \\")
    add(replacements, r"w/o RevIN & 0.2879 & 0.1515 & 0.1735 & 0.9454 \\",
        r"w/o RevIN & 0.2879 & 0.1515 & \auditbad{0.1735}{P26} & 0.9454 \\")

    add(replacements, "one representative shifted segment", r"one \auditbad{representative shifted segment}{P27}")
    add(replacements, "automatically selected informative window centered on the largest sustained frozen-versus-full disagreement",
        r"\auditbad{automatically selected informative window centered on the largest sustained frozen-versus-full disagreement}{P27}")
    add(replacements, r"\caption{Qualitative case studies on shifted online segments from the four datasets.",
        r"\caption{Qualitative case studies on shifted online segments from the four datasets.\auditnote{P27}{窗口按最大分歧选择，只能作为 illustrative case，不能当代表性平均证据。}")
    add(replacements, "representative online steps selected from regions where the online correction produces one of the largest improvements over the frozen baseline",
        r"\auditbad{representative online steps selected from regions where the online correction produces one of the largest improvements over the frozen baseline}{P28}")
    add(replacements, "supports the interpretability claim", r"\auditbad{supports the interpretability claim}{P28}")
    add(replacements, "which is consistent with the operational semantics of short-term regional load forecasting",
        r"\auditbad{which is consistent with the operational semantics of short-term regional load forecasting}{P28}")

    add(replacements, "Following the qualitative protocol used in recent online forecasting studies",
        r"\auditbad{Following the qualitative protocol used in recent online forecasting studies}{P02}")
    add(replacements, "supports the intended role separation", r"\auditbad{supports the intended role separation}{P29}")
    add(replacements, "The Stage~I gate-space visualization also provides a direct interpretation",
        r"The Stage~I gate-space visualization also provides a \auditbad{direct interpretation}{P29}")
    add(replacements, "these plots indicate", r"these plots \auditbad{indicate}{P29}")
    add(replacements, r"\caption{t-SNE summary on the three main load datasets.",
        r"\caption{t-SNE summary on the three main load datasets.\auditnote{P29}{t-SNE 只能作为诊断可视化，不能作为机制因果证明；缺少统计检验。}")
    add(replacements, r"Tetouan & 0.5288 & 1.3497 & 0.0012 & 0.63\% & 16.88\% & 13.64 \\",
        r"Tetouan & 0.5288 & 1.3497 & 0.0012 & \auditbad{0.63\%}{P29} & \auditbad{16.88\%}{P29} & 13.64 \\")
    add(replacements, r"Spain Energy--Weather & 0.1619 & 0.2699 & -0.0003 & 8.75\% & 46.25\% & 17.58 \\",
        r"Spain Energy--Weather & 0.1619 & 0.2699 & \auditbad{-0.0003}{P29} & 8.75\% & \auditbad{46.25\%}{P29} & 17.58 \\")
    add(replacements, "The drift space exhibits a larger train--online separation than the invariant space",
        r"\auditbad{The drift space exhibits a larger train--online separation than the invariant space}{P29}")
    add(replacements, "Stage~I mainly induces a mild gate re-alignment",
        r"\auditbad{Stage~I mainly induces a mild gate re-alignment}{P29}")
    add(replacements, "expert embeddings form clearly separated clusters",
        r"expert embeddings form \auditbad{clearly separated clusters}{P29}")

    add(replacements, "For every dataset, we sample multiple shifted online blocks",
        r"For every dataset, we \auditbad{sample multiple shifted online blocks}{P30}")
    add(replacements, "the full controller is strongest at all three sampled shift scores",
        r"the full controller is \auditbad{strongest at all three sampled shift scores}{P30}")
    add(replacements, "the proposed online controller provides robust gains over a range of quantitatively measured train--test shifts",
        r"the proposed online controller provides \auditbad{robust gains over a range of quantitatively measured train--test shifts}{P30}")
    add(replacements, r"\caption{Quantitative shift-score analysis on the three main load datasets.",
        r"\caption{Quantitative shift-score analysis on the three main load datasets.\auditnote{P30}{每个公开数据集只有 3 个采样块且无不确定性；结论需改成 sampled blocks。}")
    add(replacements, "Dataset & Shift Score & Frozen & Stage~III & Full",
        r"Dataset & \auditbad{Shift Score}{P30} & Frozen & Stage~III & Full")
    add(replacements, r"Spain Energy--Weather & 0.4547 & 0.2609 & \textbf{0.2539} & 0.2558 \\",
        r"Spain Energy--Weather & 0.4547 & 0.2609 & \auditbad{\textbf{0.2539}}{P30} & \auditbad{0.2558}{P30} \\")
    add(replacements, r"Spain Energy--Weather & 0.5396 & 0.2411 & \textbf{0.2362} & 0.2367 \\",
        r"Spain Energy--Weather & 0.5396 & 0.2411 & \auditbad{\textbf{0.2362}}{P30} & \auditbad{0.2367}{P30} \\")
    add(replacements, r"\caption{Online MSE as a function of the exact shift score on the four datasets.",
        r"\caption{Online MSE as a function of the exact shift score on the four datasets.\auditnote{P30}{图含 Lishui 县级点，表只列三个公开数据集；需说明数据来源和不确定性。}")

    add(replacements, "To verify that the framework is not tied to public benchmarks only",
        r"To \auditbad{verify that the framework is not tied to public benchmarks only}{P31}")
    add(replacements, "the qualitative trend remains consistent",
        r"the qualitative trend remains \auditbad{consistent}{P31}")
    add(replacements, "the strongest external baseline", r"the \auditbad{strongest external baseline}{P20}")
    add(replacements, r"narrows the gap to \textbf{SOLID++} substantially",
        r"narrows the gap to \textbf{SOLID++} \auditbad{substantially}{P31}")
    add(replacements, r"the gap is reduced to approximately $1.3\times 10^{-3}$",
        r"the gap is reduced to approximately \auditbad{$1.3\times 10^{-3}$}{P31}")
    add(replacements, "it confirms that the framework transfers",
        r"it \auditbad{confirms that the framework transfers}{P31}")
    add(replacements, r"\caption{Additional evaluation on the \textit{Lishui-9} dataset family.",
        r"\caption{Additional evaluation on the \textit{Lishui-9} dataset family.\auditnote{P31}{需突出 SOLID++ 仍优于本文方法，不能包装成 SOTA 优势。}")
    add(replacements, r"Mean online MSE & 0.2271 & 0.2269 & 0.2271 & \textbf{0.2254} & 0.2269 & 0.2256 & \textbf{0.2254} & 0.2256 & Mean online MSE & 0.2265 & \textbf{0.2241} & 0.2254 & 0.2255 \\",
        r"Mean online MSE & 0.2271 & 0.2269 & 0.2271 & \textbf{0.2254} & 0.2269 & 0.2256 & \textbf{0.2254} & 0.2256 & Mean online MSE & 0.2265 & \auditbad{\textbf{0.2241}}{P31} & \auditbad{0.2254}{P31} & \auditbad{0.2255}{P31} \\")
    add(replacements, r"\caption{Summary of the newly curated \textit{Lishui-9} dataset family.",
        r"\caption{Summary of the newly curated \textit{Lishui-9} dataset family.\auditnote{P31}{图注应说明本文方法未超过 SOLID++，只是缩小差距。}")
    add(replacements, "compares all online baselines", r"compares \auditbad{all online baselines}{P20}")
    add(replacements, r"\caption{County-level ablation on the \textit{Lishui-9} family.",
        r"\caption{County-level ablation on the \textit{Lishui-9} family.\auditnote{P32}{县级图需补来源 CSV、窗口/县区索引和生成命令。}")
    add(replacements, "using the same informative-window selection strategy",
        r"using the same \auditbad{informative-window selection strategy}{P27}")
    add(replacements, r"\caption{County-level t-SNE visualization on the \textit{Lishui-9} family.",
        r"\caption{County-level t-SNE visualization on the \textit{Lishui-9} family.\auditnote{P29}{t-SNE 图不能支撑因果或机制充分证明。}")
    add(replacements, r"\caption{Additional county-level t-SNE views on the \textit{Lishui-9} family.",
        r"\caption{Additional county-level t-SNE views on the \textit{Lishui-9} family.\auditnote{P32}{需补 exact checkpoint、样本选择和 run manifest。}")

    add(replacements, "The current results support the two main claims of this paper",
        r"\auditbad{The current results support the two main claims of this paper}{P33}")
    add(replacements, "remain effective across a broad family of transformer-based forecasting backbones",
        r"\auditbad{remain effective across a broad family of transformer-based forecasting backbones}{P33}")
    add(replacements, "Stage~III consistently improves the frozen baseline",
        r"Stage~III \auditbad{consistently improves the frozen baseline}{P33}")
    add(replacements, "strongest external controller", r"\auditbad{strongest external controller}{P20}")
    add(replacements, "the experiments show that the proposed framework can simultaneously provide strong static forecasting accuracy and measurable online adaptation gains",
        r"the experiments show that the proposed framework can \auditbad{simultaneously provide strong static forecasting accuracy and measurable online adaptation gains}{P33}")
    add(replacements, "without changing the overall training or deployment protocol",
        r"without changing the overall \auditbad{training or deployment protocol}{P34}")
    add(replacements, "Stage~III uses the discrepancy", r"\auditbad{Stage~III uses the discrepancy}{P29}")
    add(replacements, "is empirically the most stable source of online gain",
        r"is \auditbad{empirically the most stable source of online gain}{P24}")
    add(replacements, "competitive static forecasting accuracy and additional online gains under delayed feedback",
        r"competitive static forecasting accuracy and additional online gains under \auditbad{delayed feedback}{P03}")
    add(replacements, "robust and lightweight improvement", r"\auditbad{robust and lightweight improvement}{P34}")
    add(replacements, "latency, stability, and update-budget constraints",
        r"\auditbad{latency, stability, and update-budget constraints}{P34}")
    add(replacements, "consistently improves over frozen deployment under shifted online streams",
        r"\auditbad{consistently improves over frozen deployment under shifted online streams}{P33}")
    add(replacements, "remains competitive against representative online forecasting baselines",
        r"remains competitive against \auditbad{representative online forecasting baselines}{P20}")
    add(replacements, "Stage~III provides the most reliable single-step gain",
        r"Stage~III provides the \auditbad{most reliable single-step gain}{P24}")
    add(replacements, "the joint controller yields the strongest average performance",
        r"the joint controller yields the \auditbad{strongest average performance}{P25}")
    add(replacements, r"\section*{Acknowledgments}", r"\section*{\auditbad{Acknowledgments}{P35}}")
    add(replacements, "This section will be completed in the final manuscript.",
        r"\auditbad{This section will be completed in the final manuscript.}{P35}")

    missing = []
    for old, new in replacements:
        if old not in text:
            missing.append(old[:180])
        else:
            text = text.replace(old, new, 1)

    if missing:
        print("Missing replacements:", len(missing))
        for item in missing:
            print("---", item)
        raise SystemExit(1)

    issue_index = r"""

\clearpage
\section*{\textcolor{auditred}{审计问题编号与中文修改意见}}
\small
\begin{enumerate}
\item[\auditid{P01}] 模板元数据未替换：标题、作者、页眉、版权号仍是 IEEE 示例信息。修改：投稿前必须替换为真实题名、作者、单位、基金和版权信息。
\item[\auditid{P02}] 文献支撑缺失：正文没有有效 \texttt{\textbackslash cite\{\}}，也未发现可用 \texttt{.bib}。修改：所有 related work、baseline、OOD、online learning、t-SNE protocol 等说法都要补真实可核验引用。
\item[\auditid{P03}] delayed-feedback 叙事过强：代码审计发现反馈延迟以 batch 计数推进，且 Stage II/Full 接受门存在当前标签使用风险。修改：修代码后重跑，或把受影响结果标为 oracle-assisted/diagnostic。
\item[\auditid{P04}] 摘要和结论存在普遍优越性过度声明。修改：不要写 consistently outperforms representative baselines；改为“在公开 shifted benchmarks 上相对 Frozen 有小幅增益，在 Lishui-9 上与若干在线代理基线竞争但未超过 SOLID++”。
\item[\auditid{P05}] two-stage 与 three-stage 表述冲突。修改：全稿统一为 three-stage controller，或重构方法小节使 Stage I/II/III 的边界一致。
\item[\auditid{P06}] RQ1--RQ3 是空白占位。修改：删除空 RQ，或写成真实研究问题并与实验表格一一对应。
\item[\auditid{P07}] 方法小节标题仍是占位符（The decomposition of ...、C、D、E）。修改：替换为可投稿标题，如 Invariant Representation Learning、Factor-Mechanism Backbone、Stage-wise Online Adaptation、Experimental Setup。
\item[\auditid{P08}] 记号混乱：N 同时表示变量数和样本数，Section~1 交叉引用不成立，反馈延迟被写成 L steps 也与 horizon/lookback 混淆。修改：新增 notation table，区分变量数 N、窗口样本数 M、lookback L、horizon H、feedback delay D。
\item[\auditid{P09}] OOD/因果动机缺少可核验引用和清晰假设边界。修改：把 covariate/label/conditional/latent shift 与能源负荷场景的文献来源对应起来。
\item[\auditid{P10}] SCM 段落重复且未连接到可验证算法。修改：若代码没有显式 DAG/SCM 推断，则把它降级为动机背景，避免声称“principled causal basis”。
\item[\auditid{P11}] 不变性假设、IRN 方差结论和损失推导缺少证明或实现映射。修改：补推导、实现文件映射和适用条件；否则删弱公式化叙事。
\item[\auditid{P12}] adversarial environment inference 与 temporal smoothing 需要实现证据；平滑公式含未来索引时还需说明只在离线阶段使用。修改：补训练流程或改成“offline representation pretraining heuristic”。
\item[\auditid{P13}] Stage II/Full 在线接受逻辑疑似使用当前标签。修改：这是 Tier 1 问题；若不修，所有 Stage II/Full delayed-feedback 结果都不能作为严格在线证据。
\item[\auditid{P14}] 反馈延迟单位不清：代码表现更接近 batch-level delay，不是逐 timestamp/window delay。修改：实验协议要明确 online unit、batch size、delay D 的物理含义。
\item[\auditid{P15}] AdaptationGenerator 方法描述与代码不一致：审计未找到对 generator 参数的训练优化证据。修改：训练 generator 并重跑，或改写为 bounded heuristic/random delta controller。
\item[\auditid{P16}] Stage III 的 post-update prediction error 表述可能暗含当前标签。修改：说明 Stage III 是 inference-only transient correction，还是 label-available 后的更新；两者实验含义不同。
\item[\auditid{P17}] “No random shuffling” 表述不准确：split 和 online stream 保持时间顺序，但训练 DataLoader 存在 shuffle。修改：改为“splits and online evaluation preserve chronological order; offline mini-batches may be shuffled within the training prefix”。
\item[\auditid{P18}] shifted-stream 协议写法不完整：表中 55/10/35 容易被理解为完整 35\% suffix 测试，但代码/配置显示是在 later candidate pool 中选 20\% 高 shift block。修改：明确 selected stress-test block，并讨论 selection bias。
\item[\auditid{P19}] implementation table 混合了多个 config 来源。修改：为每个表格固定唯一 config、command、seed、checkpoint、data path 和代码版本。
\item[\auditid{P20}] OneNet/SOLID/SOLID++/PROCEED 当前更像内部 proxy/inspired-by 实现，不能直接称 canonical representative baselines 或 external baseline。修改：补 faithful implementation/citation validation，或改名为 proxy baselines。
\item[\auditid{P21}] shifted SOTA 表不支持 Lishui-9 上超越 SOLID/SOLID++；Ours-Full 的 Lishui MSE 高于 SOLID++。修改：正文必须显式承认 remaining gap。
\item[\auditid{P22}] 表格基本是 single-seed point estimates，缺少 std/error bar/statistical test。修改：补 multi-seed 或删除 robust、substantial、consistent 等显著性暗示。
\item[\auditid{P23}] backbone screening 可支持 plug-in scope，但缺少完整 manifest，且“clear margin/consistent gains”语气偏强。修改：保留 tested transformer backbones 范围，补 figure/table provenance。
\item[\auditid{P24}] all-eight online variants 表与 Stage III 最可靠叙事冲突：多行 Stage II MSE 明显低于 Stage III，且可能混用不同 checkpoint/offline MSE。修改：重建单一受控表，或删除 Stage II/Stage III 排名结论。
\item[\auditid{P25}] Stage-combination 证据只显示 Full 在部分数据集最好，不支持“joint controller strongest average/universal best”。修改：改成 dataset-dependent ranking；Stage II beneficial inside full controller 需额外消融支持。
\item[\auditid{P26}] component ablation 削弱 VAE 叙事：MLP Gate, no VAE 在四列均最好；RevIN 与 invariant branch 也有数据集例外。修改：VAE 只能写作一种实现选择，核心贡献应转向多专家、归一化和在线协议。
\item[\auditid{P27}] qualitative case study 是按最大 frozen-versus-full disagreement 选择的 informative window。修改：只称 illustrative，不得作为代表性提升证据。
\item[\auditid{P28}] relevance heatmap 只能说明可解释性诊断，不能证明因果机制或物理真实性。修改：删除 causal/operational semantics 过强措辞，补变量含义和选择准则。
\item[\auditid{P29}] t-SNE 结果是可视化诊断，不是机制因果证明；Stage I \(\Delta\alpha\) 很小甚至为负，Stage III +Gain 也非普遍高。修改：用“consistent with”替代“supports/proves”，并补定量/统计证据。
\item[\auditid{P30}] shift-score 分析只有每个公开数据集 3 个采样块且无不确定性；Spain 表格还与“Full strongest at all three”文字矛盾。修改：改成 sampled shifted blocks，并修正文表不一致。
\item[\auditid{P31}] Lishui-9 证据支持 transfer 和 Frozen 改善，但不支持超过 SOLID++。修改：把 Lishui 定位为真实区域数据族和剩余挑战，而非 SOTA 胜利。
\item[\auditid{P32}] Lishui county figures、case study、t-SNE 图缺 exact command、checkpoint、县区/窗口索引 manifest。修改：补图表生成清单，否则只能作为辅助展示。
\item[\auditid{P33}] Analysis/Conclusion 把“当前能证明的范围”写得过宽。修改：结论应限制为 tested backbones、sampled shifted streams、competitive online adaptation。
\item[\auditid{P34}] deployment/latency/update-budget 论述没有实验支撑。修改：补在线更新时间、内存、参数更新数、更新频率；否则删除实践部署成本结论。
\item[\auditid{P35}] Acknowledgments 仍是占位。修改：投稿前补基金、项目、数据来源或删去占位。
\item[\auditid{P36}] 全稿缺 per-result evidence manifest。修改：每个 table cell/figure 绑定 CSV row、script、config、seed、checkpoint、data split、git hash；否则顶刊审稿中可复现性风险很高。
\end{enumerate}
\normalsize
"""

    text = text.replace(r"\end{document}", issue_index + "\n\\end{document}\n", 1)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {OUT} with {len(replacements)} replacements and 36 issue notes.")


if __name__ == "__main__":
    main()
