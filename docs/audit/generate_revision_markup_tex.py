import csv
from pathlib import Path


SRC = Path("bare_jrnl_new_sample4.tex")
OUT = Path("bare_jrnl_new_sample4_revision_markup.tex")


def add(replacements: list[tuple[str, str]], old: str, new: str) -> None:
    replacements.append((old, new))


def backbone_screening_tables() -> str:
    """Build dataset-specific replacement tables for the original averaged Table III."""
    csv_path = Path("logs/load_benchmarks/backbone_suite_v2.csv")
    order = [
        "itransformer",
        "fedformer",
        "crossformer",
        "informer",
        "autoformer",
        "pyraformer",
        "transformer",
        "patchtst",
    ]
    labels = {
        "itransformer": "iTransformer",
        "fedformer": "FEDformer",
        "crossformer": "Crossformer",
        "informer": "Informer",
        "autoformer": "Autoformer",
        "pyraformer": "Pyraformer",
        "transformer": "Transformer",
        "patchtst": "PatchTST",
    }
    datasets = [
        ("panama", r"\textit{Panama}", "panama"),
        ("tetouan", r"\textit{Tetouan}", "tetouan"),
        ("spain_energy_weather", r"\textit{Spain Energy--Weather}", "spain"),
    ]
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    by_key = {(row["dataset"], row["backbone"], row["variant"]): row for row in rows}

    def row_for(dataset: str, dataset_label: str, backbone: str) -> str:
        base = by_key[(dataset, backbone, "baseline")]
        stage3 = by_key[(dataset, backbone, "stage3")]
        full = by_key[(dataset, backbone, "stage1_stage2_stage3")]
        values = [
            dataset_label,
            labels[backbone],
            f"{float(base['offline_test_mse']):.4f}",
            f"{float(base['offline_test_mae']):.4f}",
            f"{float(base['adapted_online_mse']):.4f}",
            r"--",
            f"{float(stage3['adapted_online_mse']):.4f}",
            r"--",
            f"{float(full['adapted_online_mse']):.4f}",
            r"--",
        ]
        return " & ".join(values) + r" \\"

    table_rows = "\n".join(
        row_for(dataset_key, dataset_label, backbone)
        for dataset_key, dataset_label, _label_suffix in datasets
        for backbone in order
    )
    return rf"""\begin{{table*}}[!t]
\caption{{\newtext{{Dataset-specific backbone screening on the three main load-forecasting datasets. The rows report \textit{{Panama}}, \textit{{Tetouan}}, and \textit{{Spain Energy--Weather}} separately; no cross-dataset average is used. Each metric block uses MSE/MAE subcolumns. Lower is better.}}\why{{按要求去掉原均值表注和 Panel A/B/C 结构，改为一个合并表。Online MAE was not recorded in \texttt{{backbone\_suite\_v2.csv}}, so those cells are left as ``--'' rather than fabricated.}}}}
\label{{tab:backbone_screening}}
\label{{tab:backbone_screening_panama}}
\label{{tab:backbone_screening_tetouan}}
\label{{tab:backbone_screening_spain}}
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
{{\color{{revisionblue}}
\begin{{tabular}}{{llcccccccc}}
\toprule
Dataset & Backbone & \multicolumn{{2}}{{c}}{{Offline}} & \multicolumn{{2}}{{c}}{{Online Baseline}} & \multicolumn{{2}}{{c}}{{Online Stage~III}} & \multicolumn{{2}}{{c}}{{Online Full}} \\
\cmidrule(lr){{3-4}}\cmidrule(lr){{5-6}}\cmidrule(lr){{7-8}}\cmidrule(lr){{9-10}}
 &  & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE \\
\midrule
{table_rows}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}"""


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
\definecolor{revisionred}{RGB}{190,0,0}
\definecolor{revisionblue}{RGB}{0,82,170}
\DeclareRobustCommand{\oldtext}[1]{{\color{revisionred}#1}}
\DeclareRobustCommand{\newtext}[1]{{\color{revisionblue}#1}}
\DeclareRobustCommand{\why}[1]{{\color{revisionblue}{\footnotesize (#1)}}}
\DeclareRobustCommand{\rw}[3]{\oldtext{#1} \newtext{#2}\why{#3}}"""
    if preamble_old not in text:
        raise SystemExit("preamble insertion point not found")
    text = text.replace(preamble_old, preamble_new, 1)

    replacements: list[tuple[str, str]] = []

    add(
        replacements,
        r"\title{A Sample Article Using IEEEtran.cls\\ for IEEE Journals and Transactions}",
        r"\title{\rw{A Sample Article Using IEEEtran.cls\\ for IEEE Journals and Transactions}{A Plug-and-Play Stage-Wise Online Adaptation Framework for Load Forecasting Under Distribution Shift}{原题名是 IEEE 模板标题；修改为与稿件主题一致的工作题名，未加入未经证实的 SOTA 词汇。}}",
    )
    add(
        replacements,
        r"""\author{IEEE Publication Technology,~\IEEEmembership{Staff,~IEEE,}
        % <-this % stops a space
\thanks{This paper was produced by the IEEE Publication Technology Group. They are in Piscataway, NJ.}% <-this % stops a space
\thanks{Manuscript received April 19, 2021; revised August 16, 2021.}}""",
        r"""\author{\oldtext{IEEE Publication Technology,~\IEEEmembership{Staff,~IEEE,}}\\
\newtext{Author names, affiliations, and IEEE memberships to be inserted after author confirmation}\why{原作者信息为模板占位；不能编造真实作者与单位。}
\thanks{\oldtext{This paper was produced by the IEEE Publication Technology Group. They are in Piscataway, NJ.} \newtext{Funding, data-source, and contribution acknowledgments should be inserted after verification.}\why{删除模板说明，保留需作者确认的信息槽位。}}
\thanks{\oldtext{Manuscript received April 19, 2021; revised August 16, 2021.} \newtext{Manuscript history will be supplied by the journal workflow.}\why{模板日期不可用于投稿。}}}""",
    )
    add(
        replacements,
        r"\markboth{Journal of \LaTeX\ Class Files,~Vol.~14, No.~8, August~2021}%",
        r"\markboth{\newtext{Journal header pending final metadata}}%",
    )
    add(
        replacements,
        r"{Shell \MakeLowercase{\textit{et al.}}: A Sample Article Using IEEEtran.cls for IEEE Journals}",
        r"{\newtext{Author et al.: Stage-Wise Online Adaptation for Load Forecasting}}",
    )
    add(
        replacements,
        r"\IEEEpubid{0000--0000/00\$00.00~\copyright~2021 IEEE}",
        r"\IEEEpubid{\rw{0000--0000/00\$00.00~\copyright~2021 IEEE}{Publisher copyright line to be assigned after acceptance}{模板版权号不能作为投稿信息。}}",
    )

    add(
        replacements,
        "This paper studies online multivariate regional load forecasting under delayed feedback and distribution shift.",
        r"\rw{This paper studies online multivariate regional load forecasting under delayed feedback and distribution shift.}{This paper studies multivariate regional load forecasting under distribution shift with a batch-delayed online adaptation protocol.}{把 delayed-feedback 改成与当前实现更一致的 batch-delayed protocol，避免严格逐时间戳反馈承诺。}",
    )
    add(
        replacements,
        "Extensive experiments on multiple load forecasting datasets and multiple transformer-based backbones show that the proposed framework consistently improves over frozen deployment and representative online baselines under shifted online streams.",
        r"\rw{Extensive experiments on multiple load forecasting datasets and multiple transformer-based backbones show that the proposed framework consistently improves over frozen deployment and representative online baselines under shifted online streams.}{Experiments on the tested datasets and transformer-family backbones show improved performance over frozen deployment in the shifted-stream protocol and competitive behavior relative to the implemented online-adaptation baselines, with dataset-dependent rankings.}{删除 consistently 和 representative online baselines 的过度表述；保留已由审计支持的范围。}",
    )
    add(
        replacements,
        "online load forecasting, multivariate time series, distribution shift, invariant learning, transformer backbone, delayed feedback.",
        r"online load forecasting, multivariate time series, distribution shift, invariant learning, transformer backbone, \rw{delayed feedback}{batch-delayed online adaptation}{关键词与实现协议保持一致。}.",
    )
    add(
        replacements,
        "naive online fine-tuning can be unstable under delayed labels and concept drift.",
        r"\rw{naive online fine-tuning can be unstable under delayed labels and concept drift.}{unconstrained online fine-tuning can be unstable when labels are delayed and the deployment stream drifts.}{语法和措辞更正式，避免口语化 naive。}",
    )
    add(
        replacements,
        "Stage~I recalibrates the factor mixture under delayed labels, Stage~II performs relevance-guided sparse expert updates when instability is detected, and Stage~III applies transient correction based on the discrepancy between variant and invariant representations.",
        r"\rw{Stage~I recalibrates the factor mixture under delayed labels, Stage~II performs relevance-guided sparse expert updates when instability is detected, and Stage~III applies transient correction based on the discrepancy between variant and invariant representations.}{Stage~I recalibrates the factor mixture when delayed labels become available, Stage~II performs relevance-guided sparse expert updates under the implemented acceptance gates, and Stage~III applies transient correction based on the discrepancy between variant and invariant representations.}{限定 Stage II 触发条件，避免把实现风险写成严格在线保证。}",
    )
    add(
        replacements,
        "The following sections present the problem formulation, the method, and the empirical validation.",
        r"\rw{The following sections present the problem formulation, the method, and the empirical validation.}{The following sections present the formulation, model components, online protocol, and evidence-limited empirical evaluation.}{承认当前证据范围，降低“validation”过强语气。}",
    )

    add(
        replacements,
        r"$\mathcal{D}_{\mathrm{tr}}=\{(\mathbf{X}_t,\mathbf{Y}_t)\}_{t=1}^{N}$",
        r"\rw{$\mathcal{D}_{\mathrm{tr}}=\{(\mathbf{X}_t,\mathbf{Y}_t)\}_{t=1}^{N}$}{$\mathcal{D}_{\mathrm{tr}}=\{(\mathbf{X}_j,\mathbf{Y}_j)\}_{j=1}^{M_{\mathrm{tr}}}$}{避免与变量数 $N$ 冲突，用 $M_{\mathrm{tr}}$ 表示训练窗口样本数。}",
    )
    add(
        replacements,
        r"$\min_{\theta}\frac{1}{N}\sum_{t=1}^{N}\ell\big(f_{\theta}(\mathbf{X}_t),\mathbf{Y}_t\big)$",
        r"\rw{$\min_{\theta}\frac{1}{N}\sum_{t=1}^{N}\ell\big(f_{\theta}(\mathbf{X}_t),\mathbf{Y}_t\big)$}{$\min_{\theta}\frac{1}{M_{\mathrm{tr}}}\sum_{j=1}^{M_{\mathrm{tr}}}\ell\big(f_{\theta}(\mathbf{X}_j),\mathbf{Y}_j\big)$}{同步修正经验风险中的样本数记号。}",
    )
    add(
        replacements,
        "due to the feedback delay in Section~1, updates can only use labels arriving at least $L$ steps later",
        r"\rw{due to the feedback delay in Section~1, updates can only use labels arriving at least $L$ steps later}{because labels are available only after a feedback delay $D$, online updates can use only labels that have already arrived}{修复错误章节引用，并区分 lookback $L$ 与 feedback delay $D$。}",
    )
    add(
        replacements,
        "some distribution drift is caused by variant unobserved factors.",
        r"\rw{some distribution drift is caused by variant unobserved factors.}{some drift may be associated with unobserved or weakly observed drivers.}{避免在无因果识别证据时断言 drift 原因。}",
    )
    add(
        replacements,
        "Structural Causal Model (SCM) is introduced to represent latent drivers and causal mechanisms, providing a principled basis for reasoning. SCM consists of endogenous variables $V$, exogenous variables $U$, structural assignments $V_i \\leftarrow f_i(\\mathrm{Pa}(V_i), U_i)$ for each $V_i \\in V$, and a joint distribution $P(U)$ over the exogenous variables. The directed graph induced by the parent sets $\\mathrm{Pa}_i$ encodes direct causal effects among variables. The induced directed graph encodes direct causal effects.",
        r"\rw{Structural Causal Model (SCM) is introduced to represent latent drivers and causal mechanisms, providing a principled basis for reasoning. SCM consists of endogenous variables $V$, exogenous variables $U$, structural assignments $V_i \leftarrow f_i(\mathrm{Pa}(V_i), U_i)$ for each $V_i \in V$, and a joint distribution $P(U)$ over the exogenous variables. The directed graph induced by the parent sets $\mathrm{Pa}_i$ encodes direct causal effects among variables. The induced directed graph encodes direct causal effects.}{We use SCM terminology as conceptual motivation for separating stable and variant mechanisms. An SCM consists of endogenous variables $V$, exogenous variables $U$, structural assignments $V_i \leftarrow f_i(\mathrm{Pa}(V_i), U_i)$, and a joint distribution $P(U)$. In this manuscript, the SCM discussion motivates the model design; it should not be read as evidence that a causal graph is identified from the current experiments.}{删去重复句，并明确 SCM 是动机而非已验证的因果发现。}",
    )

    add(replacements, r"\textbf{RQ1:} ", r"\oldtext{\textbf{RQ1:}}\newtext{ }\why{空白 RQ 占位删除；若保留，必须写成可由实验回答的研究问题。}")
    add(replacements, r"\textbf{RQ2:} ", r"\oldtext{\textbf{RQ2:}}\newtext{ }\why{空白 RQ 占位删除。}")
    add(replacements, r"\textbf{RQ3:} ", r"\oldtext{\textbf{RQ3:}}\newtext{ }\why{空白 RQ 占位删除。}")
    add(
        replacements,
        "finally the two-stage online adaptation.",
        r"finally the \rw{two-stage online adaptation}{three-stage online adaptation}{全文实际使用 Stage I/II/III，应统一为三阶段。}.",
    )
    add(
        replacements,
        r"\subsection{The decomposition of ... using Invariant Learning}",
        r"\subsection{\rw{The decomposition of ... using Invariant Learning}{Invariant Representation Learning for Stable and Variant Components}{替换占位标题，明确本节内容。}}",
    )
    add(
        replacements,
        "The following invariance assumption is adopted:",
        r"\rw{The following invariance assumption is adopted:}{We use the following working assumption to motivate the loss design:}{降低未经证明的“assumption adopted”语气。}",
    )
    add(
        replacements,
        "IRN ensures the residuals have mean 0 and variance $2-2\\mathrm{cov}(\\widehat{Y},Y)$",
        r"\rw{IRN ensures the residuals have mean 0 and variance $2-2\mathrm{cov}(\widehat{Y},Y)$}{Under normalized residuals, this construction encourages scale-insensitive error comparison}{原方差结论缺少完整推导和实现映射；改为保守描述。}",
    )
    add(
        replacements,
        "At convergence, $\\mathcal{M}_{\\mathrm{IL}}$ preserves invariant features $\\phi^{*}(X)$ and $\\mathcal{M}_{\\mathrm{EI}}$ fails to predict environment labels $E_{\\mathrm{infer}}$.",
        r"\rw{At convergence, $\mathcal{M}_{\mathrm{IL}}$ preserves invariant features $\phi^{*}(X)$ and $\mathcal{M}_{\mathrm{EI}}$ fails to predict environment labels $E_{\mathrm{infer}}$.}{The intended outcome is that $\mathcal{M}_{\mathrm{IL}}$ suppresses environment-specific features while retaining information useful for forecasting.}{避免声称已证明收敛到真正不变特征。}",
    )
    add(
        replacements,
        "and optionally apply a temporal smoothing step via majority voting over a radius $r$:",
        r"and \rw{optionally apply a temporal smoothing step via majority voting over a radius $r$}{in the offline representation-learning stage, apply temporal smoothing via majority voting over a radius $r$ when this option is enabled}{平滑使用未来邻域时必须限定为离线阶段，避免在线信息泄漏。}:",
    )
    add(replacements, r"\subsection{C}", r"\subsection{\rw{C}{Factor-Mechanism Forecasting Backbone}{替换占位小节标题。}}")
    add(replacements, r"\subsection{D}", r"\subsection{\rw{D}{Stage-Wise Online Adaptation}{替换占位小节标题。}}")
    add(
        replacements,
        "We now present the two-stage online adaptation",
        r"We now present the \rw{two-stage online adaptation}{three-stage online adaptation}{与 Stage I/II/III 结构一致}",
    )
    add(
        replacements,
        "we use a parameter-efficient adaptation generator.",
        r"\rw{we use a parameter-efficient adaptation generator.}{we describe the intended parameter-efficient adaptation rule used to generate bounded parameter changes.}{当前代码证据不足以证明 generator 被单独训练，改为实现安全表述。}",
    )
    add(
        replacements,
        "The same parameter-efficient generator is applied to both embeddings, and the resulting parameter shifts are optimized using the post-update prediction error.",
        r"\rw{The same parameter-efficient generator is applied to both embeddings, and the resulting parameter shifts are optimized using the post-update prediction error.}{A parallel transient-correction rule uses the difference between the variant and invariant embeddings to construct bounded prediction-time adjustments.}{避免暗示已验证的 generator 训练与当前标签可用性。}",
    )
    add(
        replacements,
        "The generator parameters are then updated using the prediction error of the adapted model, which strengthens its ability to proactively adjust parameters under variant fluctuations.",
        r"\rw{The generator parameters are then updated using the prediction error of the adapted model, which strengthens its ability to proactively adjust parameters under variant fluctuations.}{The correction is applied transiently at prediction time and should be described as an inference-time adjustment unless future code evidence confirms a separately trained generator.}{修正方法描述与实现证据不一致的问题。}",
    )
    add(replacements, r"\subsection{E}", r"\subsection{\rw{E}{Experimental Setup and Evidence Scope}{替换占位小节标题，并提示实验范围。}}")

    add(
        replacements,
        "whether each online stage improves over a frozen deployment baseline under delayed feedback",
        r"whether each online stage improves over a frozen deployment baseline under the implemented batch-delayed online protocol",
    )
    add(
        replacements,
        "whether combining Stage~I, Stage~II, and Stage~III yields the strongest overall online performance.",
        r"\rw{whether combining Stage~I, Stage~II, and Stage~III yields the strongest overall online performance.}{whether combining Stage~I, Stage~II, and Stage~III yields dataset-dependent benefits relative to the individual stages.}{避免预设 Full 一定最强。}",
    )
    add(
        replacements,
        "No random shuffling is applied across the temporal axis so as to preserve causal ordering and maintain consistency with the online adaptation scenario.",
        r"\rw{No random shuffling is applied across the temporal axis so as to preserve causal ordering and maintain consistency with the online adaptation scenario.}{The train/validation/online splits and the online evaluation stream preserve chronological order; offline training mini-batches may be shuffled within the training prefix.}{与代码中训练 DataLoader 可 shuffle 的事实一致。}",
    )
    add(
        replacements,
        "The remaining suffix is reserved for online evaluation under a shifted-stream protocol described later.",
        r"\rw{The remaining suffix is reserved for online evaluation under a shifted-stream protocol described later.}{The later suffix is used as a candidate region from which the shifted online block is selected, as described below.}{区分完整 35\% suffix 与选出的高 shift block。}",
    )
    add(
        replacements,
        r"Train/validation/online-test = 55\%/10\%/35\%",
        r"\rw{Train/validation/online-test = 55\%/10\%/35\%}{Train/validation/candidate-online region = 55\%/10\%/35\%; reported shifted block selected from the later candidate region}{修正表格中容易误导的 split 描述。}",
    )
    add(
        replacements,
        r"\begin{tabular}{ll}",
        r"\begin{tabular}{lp{0.72\textwidth}}",
    )
    add(
        replacements,
        "Factor generator & Latent dimension 64; VAE gate in the default model",
        r"Factor generator & Latent dimension 64; \rw{VAE gate in the default model}{VAE gate used as the default instantiation}{避免把 VAE 写成已被消融证明的核心贡献。}",
    )
    add(
        replacements,
        "Labels arrive with delayed feedback, and therefore parameter updates are only triggered after the corresponding delayed target becomes available.",
        r"\rw{Labels arrive with delayed feedback, and therefore parameter updates are only triggered after the corresponding delayed target becomes available.}{The online stream is processed chronologically, and label-dependent updates are intended to use only labels available through the batch-level delay buffer.}{与实现更一致；Stage II/Full 的严格延迟有效性仍需代码修复或重跑。}",
    )
    add(
        replacements,
        "Stage~II is triggered only when the relevance-map stability score indicates that an expert becomes unstable, and only a small masked subset of expert parameters is allowed to change.",
        r"\rw{Stage~II is triggered only when the relevance-map stability score indicates that an expert becomes unstable, and only a small masked subset of expert parameters is allowed to change.}{Stage~II uses relevance-map stability signals and masking to restrict updates to a small subset of expert parameters when the implemented gates accept an update.}{避免把触发逻辑写得比实现证据更绝对。}",
    )
    add(
        replacements,
        "The main criterion in the online setting is whether each individual stage improves over the baseline and whether the joint Stage~I+II+III configuration achieves the best overall online forecasting performance.",
        r"\rw{The main criterion in the online setting is whether each individual stage improves over the baseline and whether the joint Stage~I+II+III configuration achieves the best overall online forecasting performance.}{The main criterion in the online setting is whether each stage or stage combination improves over the frozen baseline under the reported protocol, while recognizing that the best variant may differ by dataset and backbone.}{避免预设每个 stage 和 full controller 均优。}",
    )
    add(
        replacements,
        "we search within the later candidate region for a contiguous online test block that maximizes the distribution-shift score relative to the training windows.",
        r"\rw{we search within the later candidate region for a contiguous online test block that maximizes the distribution-shift score relative to the training windows.}{we select a contiguous stress-test block from the later candidate region using a distribution-shift score relative to the training windows.}{说明这是 stress-test 选择，不是普通部署随机抽样。}",
    )
    add(
        replacements,
        "indicating that the online streams used in this protocol differ substantially from the offline training distribution",
        r"\rw{indicating that the online streams used in this protocol differ substantially from the offline training distribution}{indicating measurable train--online covariate discrepancy under this selected-block protocol}{删除 substantially 的主观强度词。}",
    )
    add(
        replacements,
        r"The remaining four are \emph{representative online forecasting baselines}:",
        r"\rw{The remaining four are \emph{representative online forecasting baselines}:}{The remaining four are implemented proxy baselines inspired by online forecasting-time adaptation methods:}{当前仓库未证明它们是外部方法的 faithful reproduction。}",
    )
    add(
        replacements,
        "Table~\\ref{tab:shifted_online_sota} summarizes the shifted-stream comparison under a shared iTransformer backbone and a shared frozen offline checkpoint.",
        r"Table~\ref{tab:shifted_online_sota} summarizes the \rw{shifted-stream comparison}{selected shifted-block comparison}{协议名称更精确} under a shared iTransformer backbone and a shared frozen offline checkpoint.",
    )
    add(
        replacements,
        r"\textbf{SOLID}, \textbf{SOLID++}, and \textbf{PROCEED} are representative forecasting-time adaptation baselines.",
        r"\rw{\textbf{SOLID}, \textbf{SOLID++}, and \textbf{PROCEED} are representative forecasting-time adaptation baselines.}{\textbf{SOLID}, \textbf{SOLID++}, and \textbf{PROCEED} denote the implemented proxy variants used in this repository.}{避免把内部实现写成已核验外部基线。}",
    )
    add(
        replacements,
        "Naive online gradient updates are unstable under delayed feedback and explicit distribution shift.",
        r"\rw{Naive online gradient updates are unstable under delayed feedback and explicit distribution shift.}{Full-parameter online gradient updates are unstable in these reported shifted-block runs.}{限定到当前结果，避免泛化。}",
    )
    add(
        replacements,
        "In contrast, the proposed controller remains competitive across all four datasets:",
        r"\rw{In contrast, the proposed controller remains competitive across all four datasets:}{The proposed controller is competitive in this evidence set, but the ranking is dataset-dependent:}{更准确承认 Lishui 与 SOLID++ 差距。}",
    )
    add(
        replacements,
        r"\caption{Comparison with online forecasting baselines under the shifted-stream protocol on the iTransformer backbone.",
        r"\caption{\rw{Comparison with online forecasting baselines under the shifted-stream protocol on the iTransformer backbone.}{Comparison with implemented online-adaptation baselines under the selected shifted-block protocol on the iTransformer backbone.}{表注区分 proxy baseline 和 selected-block protocol。}",
    )

    add(
        replacements,
        "Table~\\ref{tab:backbone_screening} reports the broad plug-in screening over eight transformer-based backbones under a shared configuration and a shared online protocol.",
        r"\rw{Table~\ref{tab:backbone_screening} reports the broad plug-in screening over eight transformer-based backbones under a shared configuration and a shared online protocol.}{Table~\ref{tab:backbone_screening} reports the plug-in screening over eight transformer-based backbones in one merged dataset-specific table for \textit{Panama}, \textit{Tetouan}, and \textit{Spain Energy--Weather} under the reported shared configuration and online protocol.}{原 Table III 为三数据集均值；现在改为一个合并表，不再使用 Panel A/B/C。}",
    )
    add(
        replacements,
        "with a clear margin over the remaining candidates.",
        r"\rw{with a clear margin over the remaining candidates.}{with lower reported mean error than the remaining candidates.}{避免未给统计检验时使用 clear margin。}",
    )
    add(
        replacements,
        "the Stage~III transient correction yields consistent online gains across all screened backbones",
        r"\rw{the Stage~III transient correction yields consistent online gains across all screened backbones}{the Stage~III transient correction reduces online MSE in many dataset--backbone rows in Table~\ref{tab:backbone_screening}}{集中为一个表后，正文引用同步改为 Table III。}",
    )
    add(
        replacements,
        r"""\begin{table*}[t]
\caption{Broad backbone screening on the three main load-forecasting datasets. Reported numbers are means over \textit{Panama}, \textit{Tetouan}, and \textit{Spain Energy--Weather}. Lower is better.}
\label{tab:backbone_screening}
\centering
\begin{tabular}{lccccc}
\toprule
Backbone & Offline MSE & Offline MAE & Online Baseline & Online Stage~III & Online Full \\
\midrule
iTransformer & 0.2071 & 0.3122 & 0.2077 & 0.2046 & 0.2047 \\
FEDformer & 0.2018 & 0.3165 & 0.2024 & 0.1995 & 0.1995 \\
Crossformer & 0.2445 & 0.3418 & 0.2442 & 0.2412 & 0.2412 \\
Informer & 0.3463 & 0.4073 & 0.3465 & 0.3440 & 0.3436 \\
Autoformer & 0.3723 & 0.4296 & 0.3728 & 0.3704 & 0.3708 \\
Pyraformer & 0.5549 & 0.5539 & 0.5554 & 0.5535 & 0.5536 \\
Transformer & 0.5772 & 0.5676 & 0.5781 & 0.5771 & 0.5770 \\
PatchTST & 0.6623 & 0.6200 & 0.6632 & 0.6626 & 0.6628 \\
\bottomrule
\end{tabular}
\end{table*}""",
        backbone_screening_tables(),
    )
    add(
        replacements,
        "First, Stage~III remains the most reliable single online component across all eight backbones.",
        r"\rw{First, Stage~III remains the most reliable single online component across all eight backbones.}{First, the all-backbone table should be interpreted cautiously because several Stage~II entries are lower than Stage~III and the provenance of the variant rows must be frozen before drawing a ranking conclusion.}{修正文表冲突，不改实验数字。}",
    )
    add(
        replacements,
        "Second, Stage~II is generally weaker than Stage~III and often only yields modest gains.",
        r"\rw{Second, Stage~II is generally weaker than Stage~III and often only yields modest gains.}{Second, Stage~II behavior is sensitive to backbone and run provenance, so its role should be discussed as conditional rather than uniformly weaker.}{避免与表中 Stage II 数字冲突。}",
    )
    add(
        replacements,
        r"\caption{Online variant comparison across eight transformer-based backbones under the tuned controller settings.",
        r"\caption{\rw{Online variant comparison across eight transformer-based backbones under the tuned controller settings.}{Online variant comparison across eight transformer-based backbones under the reported controller settings; this table requires a frozen provenance manifest before supporting stage-ranking claims.}{表注加入证据边界。}",
    )
    add(
        replacements,
        r"\caption{Variant-level online comparison across eight transformer-based backbones under the tuned controller settings.}",
        r"\caption{\rw{Variant-level online comparison across eight transformer-based backbones under the tuned controller settings.}{Variant-level online comparison across eight transformer-based backbones under the reported controller settings.}{避免 tuned 一词暗示未说明的调参协议。}}",
    )

    add(
        replacements,
        "First, Stage~III remains the strongest single online module on all four datasets",
        r"\rw{First, Stage~III remains the strongest single online module on all four datasets}{First, Stage~III is the strongest single online module in the reported stage-combination table}{限定到表格证据。}",
    )
    add(
        replacements,
        "This confirms that Stage~II becomes beneficial once it operates inside the complete controller",
        r"\rw{This confirms that Stage~II becomes beneficial once it operates inside the complete controller}{This suggests that Stage~II may be useful in combination on selected datasets, but the benefit is not uniform}{避免“confirms”过强。}",
    )
    add(
        replacements,
        r"\caption{Ablation on online stage combinations under the shifted-stream protocol.",
        r"\caption{\rw{Ablation on online stage combinations under the shifted-stream protocol.}{Ablation on online stage combinations under the selected shifted-block protocol.}{与协议表述统一。}",
    )
    add(
        replacements,
        r"\caption{Stage-combination ablation under the shifted-stream protocol.",
        r"\caption{\rw{Stage-combination ablation under the shifted-stream protocol.}{Stage-combination ablation under the selected shifted-block protocol.}{与协议表述统一。}",
    )
    add(
        replacements,
        "The component ablation reveals a clear hierarchy of importance.",
        r"\rw{The component ablation reveals a clear hierarchy of importance.}{The component ablation identifies components with large effects, but the hierarchy is dataset-dependent.}{避免用 clear hierarchy 掩盖数据集例外。}",
    )
    add(
        replacements,
        "Removing RevIN or collapsing the multi-expert factor bank causes the largest degradation on both the public suite and the Lishui family",
        r"\rw{Removing RevIN or collapsing the multi-expert factor bank causes the largest degradation on both the public suite and the Lishui family}{Removing RevIN or collapsing the multi-expert factor bank causes large average degradation, although individual datasets show exceptions}{与 Spain 上 w/o RevIN 数字一致。}",
    )
    add(
        replacements,
        "Removing the invariant branch also degrades the result on most datasets",
        r"\rw{Removing the invariant branch also degrades the result on most datasets}{Removing the invariant branch degrades several reported datasets but not all of them}{承认 Spain 列例外。}",
    )
    add(
        replacements,
        r"\caption{Common component ablation under the shifted-stream protocol.",
        r"\caption{\rw{Common component ablation under the shifted-stream protocol.}{Common component ablation under the selected shifted-block protocol.}{与协议表述统一。}",
    )

    add(
        replacements,
        "we inspect one representative shifted segment from each of the four datasets.",
        r"\rw{we inspect one representative shifted segment from each of the four datasets.}{we inspect one illustrative shifted segment from each of the four datasets.}{窗口按最大差异选择，只能称 illustrative。}",
    )
    add(
        replacements,
        "automatically selected informative window centered on the largest sustained frozen-versus-full disagreement.",
        r"\rw{automatically selected informative window centered on the largest sustained frozen-versus-full disagreement.}{automatically selected informative window centered on a large sustained frozen-versus-full disagreement.}{避免选择策略被误读为代表性样本。}",
    )
    add(
        replacements,
        r"\caption{Qualitative case studies on shifted online segments from the four datasets.",
        r"\caption{\rw{Qualitative case studies on shifted online segments from the four datasets.}{Illustrative qualitative case studies on selected shifted online segments from the four datasets.}{图注明确 illustrative。}",
    )
    add(
        replacements,
        "representative online steps selected from regions where the online correction produces one of the largest improvements over the frozen baseline.",
        r"\rw{representative online steps selected from regions where the online correction produces one of the largest improvements over the frozen baseline.}{illustrative online steps selected from high-improvement regions.}{避免把有选择样本称为 representative。}",
    )
    add(
        replacements,
        "The visualization therefore supports the interpretability claim of the proposed framework",
        r"\rw{The visualization therefore supports the interpretability claim of the proposed framework}{The visualization provides diagnostic support for the interpretability discussion}{避免把 heatmap 作为强证明。}",
    )
    add(
        replacements,
        r"\caption{Input-layer relevance heatmaps for representative online steps from the four datasets.",
        r"\caption{\rw{Input-layer relevance heatmaps for representative online steps from the four datasets.}{Input-layer relevance heatmaps for illustrative online steps from the four datasets.}{样本选择不是随机代表性。}",
    )

    add(
        replacements,
        "Following the qualitative protocol used in recent online forecasting studies,",
        r"\rw{Following the qualitative protocol used in recent online forecasting studies,}{As a diagnostic visualization,}{删除未给引用支撑的 recent studies 表述。}",
    )
    add(
        replacements,
        "This supports the intended role separation of the two branches",
        r"\rw{This supports the intended role separation of the two branches}{This is consistent with the intended role separation of the two branches}{t-SNE 只能支持一致性观察，不是机制证明。}",
    )
    add(
        replacements,
        "The Stage~I gate-space visualization also provides a direct interpretation",
        r"\rw{The Stage~I gate-space visualization also provides a direct interpretation}{The Stage~I gate-space visualization provides a qualitative diagnostic view}{降低 direct interpretation 语气。}",
    )
    add(
        replacements,
        r"\caption{t-SNE summary on the three main load datasets.",
        r"\caption{\rw{t-SNE summary on the three main load datasets.}{t-SNE diagnostic summary on the three main load datasets.}{强调诊断性质。}",
    )
    add(
        replacements,
        "The drift space exhibits a larger train--online separation than the invariant space",
        r"\rw{The drift space exhibits a larger train--online separation than the invariant space}{In this visualization, the drift space shows larger train--online separation than the invariant space}{限定为可视化结果。}",
    )
    add(
        replacements,
        "while the right column shows that the expert embeddings form clearly separated clusters.",
        r"\rw{while the right column shows that the expert embeddings form clearly separated clusters.}{while the right column visualizes separation among expert embeddings.}{删除 clearly，避免无统计检验支撑。}",
    )

    add(
        replacements,
        "For every dataset, we sample multiple shifted online blocks from the post-validation region",
        r"\rw{For every dataset, we sample multiple shifted online blocks from the post-validation region}{For each public dataset, we evaluate three sampled shifted online blocks from the post-validation region}{表格实际每个公开数据集三块。}",
    )
    add(
        replacements,
        "the full controller is strongest at all three sampled shift scores",
        r"\rw{the full controller is strongest at all three sampled shift scores}{Stage~III or the full controller is strongest depending on the sampled shift score}{修正文表不一致，Spain 第一和第三行 Stage III 更低。}",
    )
    add(
        replacements,
        "the proposed online controller provides robust gains over a range of quantitatively measured train--test shifts",
        r"\rw{the proposed online controller provides robust gains over a range of quantitatively measured train--test shifts}{the reported online variants improve over the frozen baseline on these sampled shifted blocks}{避免 robust/range 过强且无不确定性。}",
    )
    add(
        replacements,
        r"\caption{Quantitative shift-score analysis on the three main load datasets.",
        r"\caption{\rw{Quantitative shift-score analysis on the three main load datasets.}{Quantitative shift-score analysis on sampled shifted blocks from the three main load datasets.}{说明采样块范围。}",
    )
    add(
        replacements,
        r"\caption{Online MSE as a function of the exact shift score on the four datasets.",
        r"\caption{\rw{Online MSE as a function of the exact shift score on the four datasets.}{Online MSE as a function of the exact shift score on sampled blocks and Lishui county points.}{图注与表格范围对齐。}",
    )

    add(
        replacements,
        "To verify that the framework is not tied to public benchmarks only",
        r"\rw{To verify that the framework is not tied to public benchmarks only}{To test whether the framework transfers beyond the public benchmarks}{verify 改为 test，避免过强验证语气。}",
    )
    add(
        replacements,
        "Table~\\ref{tab:lishui_family} shows that the qualitative trend remains consistent:",
        r"Table~\ref{tab:lishui_family} shows that the qualitative trend is \rw{consistent}{partly consistent}{Lishui-9 仍存在与 SOLID++ 的差距。}:",
    )
    add(
        replacements,
        r"narrows the gap to \textbf{SOLID++} substantially",
        r"\rw{narrows the gap to \textbf{SOLID++} substantially}{narrows the gap to \textbf{SOLID++} in the reported tuned comparison}{去除 substantially，避免无统计支撑。}",
    )
    add(
        replacements,
        "the strongest external baseline still remains slightly better",
        r"\rw{the strongest external baseline still remains slightly better}{the implemented SOLID++ baseline remains better in family-level mean MSE}{避免 external/strongest 未核验，直接陈述表格事实。}",
    )
    add(
        replacements,
        "it confirms that the framework transfers to a newly curated regional load dataset with real alignment noise",
        r"\rw{it confirms that the framework transfers to a newly curated regional load dataset with real alignment noise}{it suggests that the framework can transfer to a newly curated regional load dataset with real alignment noise, while leaving a remaining online-control gap}{更准确体现证据和剩余差距。}",
    )
    add(
        replacements,
        r"\caption{Additional evaluation on the \textit{Lishui-9} dataset family.",
        r"\caption{\rw{Additional evaluation on the \textit{Lishui-9} dataset family.}{Additional evidence on the \textit{Lishui-9} dataset family, including the remaining gap to SOLID++.}{图表标题直接提示差距，避免误读为胜出。}",
    )
    add(
        replacements,
        r"\caption{Summary of the newly curated \textit{Lishui-9} dataset family.",
        r"\caption{\rw{Summary of the newly curated \textit{Lishui-9} dataset family.}{Summary of the newly curated \textit{Lishui-9} dataset family and its tuned comparison against SOLID++.}{图注更明确比较对象。}",
    )
    add(
        replacements,
        "compares all online baselines and the proposed variants using online MSE.",
        r"\rw{compares all online baselines and the proposed variants using online MSE.}{compares the implemented online baselines and proposed variants using online MSE.}{避免 all online baselines 的不完整覆盖暗示。}",
    )

    add(
        replacements,
        "The current results support the two main claims of this paper.",
        r"\rw{The current results support the two main claims of this paper.}{The current results support two narrower evidence-backed claims.}{结论范围收窄。}",
    )
    add(
        replacements,
        "remain effective across a broad family of transformer-based forecasting backbones.",
        r"\rw{remain effective across a broad family of transformer-based forecasting backbones.}{can be reused across the tested transformer-based forecasting backbones.}{从效果断言改为可复用性事实。}",
    )
    add(
        replacements,
        "Stage~III consistently improves the frozen baseline",
        r"\rw{Stage~III consistently improves the frozen baseline}{Stage~III often improves the frozen baseline in the reported shifted-block experiments}{避免全称 consistent。}",
    )
    add(
        replacements,
        "Overall, the experiments show that the proposed framework can simultaneously provide strong static forecasting accuracy and measurable online adaptation gains on multivariate regional load forecasting benchmarks.",
        r"\rw{Overall, the experiments show that the proposed framework can simultaneously provide strong static forecasting accuracy and measurable online adaptation gains on multivariate regional load forecasting benchmarks.}{Overall, the experiments indicate that the framework can combine competitive static forecasting with measurable online adaptation gains in the reported benchmark settings.}{删除 strong 和泛化过强结论。}",
    )
    add(
        replacements,
        "without changing the overall training or deployment protocol.",
        r"\rw{without changing the overall training or deployment protocol.}{without changing the high-level training and deployment interface used in these experiments.}{避免承诺真实部署协议完全不变。}",
    )
    add(
        replacements,
        "The ranking in Table~\\ref{tab:backbone_screening} therefore has a clear methodological interpretation: the framework is portable across multiple transformer-based predictors, while the final absolute performance still depends on the representational strength of the chosen backbone.",
        r"\rw{The ranking in Table~\ref{tab:backbone_screening} therefore has a clear methodological interpretation: the framework is portable across multiple transformer-based predictors, while the final absolute performance still depends on the representational strength of the chosen backbone.}{The dataset-specific rows in Table~\ref{tab:backbone_screening} show that the framework can be instantiated across multiple transformer-based predictors, while absolute performance remains dataset- and backbone-dependent.}{Table III 已合并为一个逐数据集长表，讨论文字同步改为 dataset-specific rows。}",
    )
    add(
        replacements,
        "Stage~III alone already provides a robust and lightweight improvement over the frozen baseline.",
        r"\rw{Stage~III alone already provides a robust and lightweight improvement over the frozen baseline.}{Stage~III alone provides a lightweight improvement over the frozen baseline in many reported settings.}{删除 robust，并限定范围。}",
    )
    add(
        replacements,
        "latency, stability, and update-budget constraints without changing the overall framework.",
        r"\rw{latency, stability, and update-budget constraints without changing the overall framework.}{stability and update-budget constraints; latency and memory claims require dedicated cost measurements.}{未测延迟和内存，不能作为已验证结论。}",
    )
    add(
        replacements,
        "The experiments show that the framework is compatible with multiple transformer-based backbones, consistently improves over frozen deployment under shifted online streams, and remains competitive against representative online forecasting baselines.",
        r"\rw{The experiments show that the framework is compatible with multiple transformer-based backbones, consistently improves over frozen deployment under shifted online streams, and remains competitive against representative online forecasting baselines.}{The experiments show that the framework is compatible with the tested transformer-based backbones, improves over frozen deployment in the reported shifted-block settings, and is competitive with the implemented online-adaptation baselines while retaining dataset-dependent limitations.}{结论与审计证据一致，删除普遍优越性。}",
    )
    add(
        replacements,
        "The empirical evidence further indicates that Stage~III provides the most reliable single-step gain, while the joint controller yields the strongest average performance when the online update policy is sufficiently constrained.",
        r"\rw{The empirical evidence further indicates that Stage~III provides the most reliable single-step gain, while the joint controller yields the strongest average performance when the online update policy is sufficiently constrained.}{The empirical evidence further suggests that Stage~III is a useful lightweight component, while the joint controller's benefit is dataset- and backbone-dependent.}{修正 all-backbone 和 stage-combination 结论过强问题。}",
    )
    add(
        replacements,
        r"\section*{Acknowledgments}",
        r"\section*{\rw{Acknowledgments}{Acknowledgments to Be Completed}{占位标题改为待完成状态，避免误认为最终稿。}}",
    )
    add(
        replacements,
        "This section will be completed in the final manuscript.",
        r"\rw{This section will be completed in the final manuscript.}{Funding sources, data providers, and institutional acknowledgments must be inserted after author verification.}{保留待作者确认的信息，不编造资助或贡献。}",
    )

    scope_note = r"""

\begin{center}
\fbox{\parbox{0.94\linewidth}{\newtext{\textbf{Revision-scope note.} This marked-up version fixes wording, format, grammar, terminology, caption clarity, and evidence-scope statements that can be corrected without inventing new experiments or citations. Experimental-result limitations that require reruns, multi-seed evidence, faithful external baselines, or code/protocol changes are not fabricated here; they are reflected only as scoped wording or author-action notes.}\why{按用户要求，本稿只修可通过文本与格式处理的问题，不伪造实验改进。}}}
\end{center}
\noindent\rw{Journal of \LaTeX\ Class Files,~Vol.~14, No.~8, August~2021 / Shell et al.: A Sample Article Using IEEEtran.cls for IEEE Journals}{Journal header pending final metadata / Author et al.: Stage-Wise Online Adaptation for Load Forecasting}{原页眉为模板内容；实际作者和期刊元数据需作者确认后填写。}
"""
    text = text.replace(r"\section{Introduction}", scope_note + "\n\\section{Introduction}", 1)

    missing = []
    for old, new in replacements:
        if old not in text:
            missing.append(old[:180])
        else:
            text = text.replace(old, new, 1)

    if missing:
        print(f"Missing replacements: {len(missing)}")
        for item in missing:
            print("---", item)
        raise SystemExit(1)

    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {OUT} with {len(replacements)} red-blue revisions.")


if __name__ == "__main__":
    main()
