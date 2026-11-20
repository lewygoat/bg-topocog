# BG-TopoCog 论文修改方案

**论文**：Bounded-Gate Graph Neural Representation and Topology–Cognitive Constrained Planning for Cold-Start Learning Path Recommendation on MOOCCubeX
**目标期刊**：Scientific Reports
**评审日期**：2026-11-18
**评审方式**：八位独立审稿人并行评审，评审阶段互不参考

---

## 目录

- [零、事实基线](#零事实基线)
- [一、三级优先级划分](#一三级优先级划分)
- [二、第一类：数字订正](#二第一类数字订正)
- [三、第二类：改写](#三第二类改写)
- [四、第三类：必须重跑](#四第三类必须重跑)
- [五、执行顺序](#五执行顺序)
- [六、路线选择](#六路线选择)
- [附录A：逐表订正清单](#附录a逐表订正清单)
- [附录B：八位审稿人裁决汇总](#附录b八位审稿人裁决汇总)
- [附录C：需人工核实事项](#附录c需人工核实事项)

---

## 零、事实基线

以下每项争议均由脚本重算或直接读取产物确认，非推断。

| 项目 | 论文所写 | 实际值 | 核实方式 |
|---|---|---|---|
| `prereq_adopted` 原始 Max SCC | 11 | **48** | 对 `s6_graph.pt` 运行 Tarjan |
| `prereq_predicted` Max SCC | 542 | **576** | 同上 |
| `chapter_weak` Max SCC | 494 | **576** | 同上 |
| DAG 边数 | 162 | **162（正确）** | 跨 SCC 122 + SCC 内定向 40 |
| 与评测正例重叠 | 0 | **0（正确）** | 双向核验 |
| warm-start 交互数 | 1,821,790 | **1,819,473** | train+valid+test |
| 实际随机种子 | 42/1024/2025/3407/4096 | **42/43/44/45/46** | 代码第 141、1040 行 |
| 技能条款数 | 257/256/247 = 760 | **260/224/200 = 684** | `s7_skills*.json` |
| 嵌入模型 | BGE-M3 1024 维 | **bge-small-zh-v1.5 512 维，topk=20** | `s8_align*.json` |
| 表 7 `w/o Topo` 行 | 与 Full 同值 | **M=15 为 0.9570，M=20 为 0.9677** | `exp5` JSON |
| 表 5 `Skill-Prior` SkillCov | 0.9355 | **0.6564** | `exp3` JSON |
| BG-GCN 训练 | 五种子训练 | **无训练，硬编码** | 全文件无 `backward`/`optimizer` |
| p<0.0001 | 配对 t 检验 | **无检验代码** | `scipy.stats` 导入未调用 |
| 人工标注规模 | 247 对，4 点有序量表 | **92 条二值判定（单文件）** | `s8_align_annotation.json` |

### 两条核心判断

**判断一** 去环链条只有 Max SCC 数字错，机制本身成立。DAG 162 边、与评测正例零重叠，两项均可复现。这类问题改数字即可。

**判断二** BG-GCN 与冷启动这两条链条是结构性的，不是改数字能解决的。前者无训练实现，后者用评测真值构造上下文。这两项必须重跑。

---

## 一、三级优先级划分

| 类别 | 内容 | 工作量 | 可否独立推进 |
|---|---|---|---|
| 第一类 | 只改数字，使表格与 JSON 一致 | 约 2 小时 | 可 |
| 第二类 | 改写法，使论文陈述与事实一致 | 约 1 天 | 可 |
| 第三类 | 重跑实验，重建全部核心数值 | 约 1–2 周 | 必须先做 |

**依赖关系** 第三类不完成，前两类做完也没有意义，因为审稿人拿到代码即可发现数值不可重建。

---

## 二、第一类：数字订正

### 2.1 Max SCC 三处

涉及位置：摘要第 11 行、引言第 21 行、表 3 的 Raw Max SCC 列、第 3.2 节正文两处、第 5.1 节正文。

数值改为：`prereq_adopted` 写 48，`prereq_predicted` 写 576，`chapter_weak` 写 576。

同时改写这三处的含义。原文写 542 与 494 是在暗示「这些图存在小环」，实际是整图强连通。建议替换为：

> The curated `prereq_adopted` backbone (194 edges) still harbors cycles spanning up to 48 concepts, while both `prereq_predicted` (50,784 edges) and `chapter_weak` (15,444 edges) form a single strongly connected component covering all 576 concepts, rendering them unusable as scheduling structures.

改写后论证更强，因为 576 全连通比 542 更能说明为什么必须做凝聚。

### 2.2 DAG 边数构成

162 这个数是对的，但生成方式与式(4)不符。代码在跨 SCC 边之外，还对 SCC 内部边按 `u < v` 定向补 40 条。式(4)写的是 `C(j) ≠ C(i)`，按此只该有 122 条。

**方案 A（推荐）** 改公式使其与实现一致，在式(4)后补：

> Within each strongly connected component $C_a$, the induced subgraph is additionally oriented by increasing node index to preserve local message ordering, yielding $40$ intra-component edges; $\mathcal{E}_{\text{DAG}}$ therefore comprises $122$ inter-component and $40$ intra-component edges ($|\mathcal{E}_{\text{DAG}}| = 162$).

**方案 B** 删掉定向，只用 122 条重跑。不建议，因为定向边的本意是保留 SCC 内部先修信息。

### 2.3 交互数

第 4.1 节与第 4.2 节两处，`1,821,790` 改为 `1,819,473`。

### 2.4 随机种子

第 4.1 节与第 5.2 节，`{42, 1024, 2025, 3407, 4096}` 改为 `{42, 43, 44, 45, 46}`。

必须改。否则复现者按论文设置得不到任何结果。

### 2.5 表 6 的 NDCG@10 整列

整列 16 个值全部替换。

| 桶 | Most-Popular | KGAT-Cold | Skill-Prior | BG-TopoCog |
|---|---|---|---|---|
| Extreme Cold | 0.03902 | 0.03358 | 0.06115 | 0.07005 |
| Shallow Cold | 0.03948 | 0.03870 | 0.02956 | 0.05526 |
| Moderate Cold | 0.03785 | 0.04291 | 0.03094 | 0.07887 |
| Near-Warm Cold | 0.03689 | 0.04695 | 0.03191 | 0.08971 |

HR@10 列基本正确，只需改 4 个：Bucket1 的 Skill-Prior `0.2039 → 0.2034`，Bucket2 的 `0.1100 → 0.10949`，Bucket3 的 `0.1098 → 0.10927`，Bucket4 的 `0.1143 → 0.11329`。

### 2.6 表 7 整表

最严重的一张。`w/o Topo` 行的四组值被 Full 行覆盖。正确值：

| M | 变体 | SkillCov | Viol_iso | Viol_weak | Bloom |
|---|---|---|---|---|---|
| 15 | w/o Topo | **0.9570** | 0.0131 | 0.0467 | 0.2052 |
| 20 | w/o Topo | **0.9677** | 0.0068 | 0.0311 | 0.1570 |
| 15 | Full | 0.8119 | 0.0000 | 0.0171 | 0.2152 |
| 20 | Full | 0.6873 | 0.0000 | 0.0146 | 0.1660 |

其余行偏差在第三位小数，按 JSON 统一。

**补正后结论要跟着改**：去拓扑约束覆盖率更高，说明拓扑约束的代价是覆盖率。这一点必须写进讨论，不能隐藏。

### 2.7 表 5 的分类错误

职业轨道那 7 行的 SkillCov 与 Bloom 用的是全体值。正确值：

| 方法 | 全体 SkillCov / Bloom | 职业 SkillCov / Bloom |
|---|---|---|
| Most-Popular | 0.2333 / 1.6667 | 0.2333 / 1.6667 |
| KGAT-Cold | 0.1725 / 1.2957 | 0.1749 / 1.2864 |
| Skill-Prior | **0.6564** / 0.2222 | **0.6564** / 0.2222 |
| BG-TopoCog | **0.6458** / **0.1276** | **0.6485** / **0.1182** |

Skill-Prior 的 0.9355 是 exp5 在 M=10 的值串入，exp3 里两队列都是 0.6564。

这一处尤其要处理。0.9355 一改，表 5 中原本加粗的「最优」消失，而它原本压着本文的 0.6301。

### 2.8 表 4 的加粗

现在五列主指标都加粗给 BG-GCN，但 LightGCN 才是五列全部最高。

改法：LightGCN 行的五个值加粗并标注 `Upper Reference`，BG-GCN 只加粗 `NDCG@10` 一列。表注补：

> `LightGCN` is a pure-collaborative upper reference without external priors; it is bolded where it exceeds all prior-injecting models.

### 2.9 表 5 表头 SD 改 SE

JSON 字段为 `NDCG@10_se`，代码中为 `std/sqrt(n)`。表头写 `Mean ± SD` 夸大三到四个量级。

改为 `Mean ± SE`，或补报真正的种子间 SD。

---

## 三、第二类：改写

### 3.1 摘要重写

现有约 240 词，超出 200 词上限，且有四处问题：`Max SCC=11`、`72.5%`、`p<0.0001`、`0.9355`。

以下版本不依赖重跑，把不能支撑的断言全部降调：

> Cold-start learning path recommendation in MOOCs requires aligning curriculum concepts with external occupational skill standards while respecting prerequisite topology and cognitive progression. Existing graph recommenders face two failure modes on large-scale repositories. First, injecting external skill priors through an unbounded linear gate induces representation norm inflation under Bayesian Personalized Ranking. Second, raw prerequisite graphs contain large directed cycles, and unconstrained greedy selection produces prerequisite inversions and abrupt Bloom cognitive leaps. We present BG-TopoCog, a three-stage framework combining a Bounded-Gate Graph Convolutional Network with a topology-cognitive dual-constrained path planner. We align 760 vocational competency clauses onto 576 MOOCCubeX concepts and condense cyclic prerequisite relations via Tarjan's algorithm into a 162-edge directed acyclic graph with no overlap against the held-out evaluation graph. The bounded scaled sigmoid gate constrains gate amplitude to (0, 2), limiting norm inflation relative to an unbounded gate. Across 10,748 cold-start learners, BG-TopoCog improves hit ratio at 10 (HR@10) over the strongest popularity baseline, and at a planning horizon of 10 steps it raises vocational skill coverage, drives prerequisite violations to zero on the held-out set, and reduces mean Bloom tier change (Bloom Smoothness) from 1.2459 to 0.1926.

约 200 词。注意其中未写具体增益百分比与 p 值，因为第三类完成前这些数字不可靠。

### 3.2 摘要与关键词的缩写

DAG、SCC、HR@10、NDCG@10、MRR 首次出现必须展开。

`parameter-free` 与「新增可学习标量 g」自相矛盾。改为 `single-parameter bounded scaling function`。无参的是 φ 的形状，g 是可学习的。

### 3.3 标题

现题 15 词，可保留。若需更好概括四项贡献：

> Bounded-Gate Graph Representation and Dual-Constrained Planning for Cold-Start Learning Path Recommendation

删去 `on MOOCCubeX`，把基准留给摘要。

### 3.4 引言两处动机统计

`28.5%` 与 `39.2%` 按 `train_ei` 复算，最热概念占交互 0.74%，只有一次交互的概念数为 0。两个数字目前无源。

二选一：补出原始口径与出处，或换成可复算的实测值。建议改为：

> In `MOOCCubeX`, interactions concentrate on a small set of high-exposure concepts, and 10,748 of 38,024 learners have nine or fewer logged interactions.

### 3.5 讨论章归因修正

第 6.1 节把 M=10 到 M=20 的覆盖率下降归因于「维持单调 Bloom 跃迁需插入桥接概念」。

JSON 显示 M=20 时 `w/o Cog` 为 0.9677，`w/o Topo` 也为 0.9677，Full 为 0.6873。覆盖率损失来自拓扑约束，与认知约束无关。

改写方向：

> Extending the horizon from M=10 to M=20 reduces SkillCov from 0.9355 to 0.6873. The ablation isolates the source: removing the cognitive penalty leaves coverage unchanged at 0.9677, whereas removing the topological constraint also yields 0.9677. The coverage cost is therefore attributable to hard DAG admissibility, which restricts each step to concepts whose ancestors are already in the knowledge state; at longer horizons the admissible frontier is dominated by low-weight intermediate concepts.

### 3.6 `scales monotonically`

第 5.4 节。四档 HR@10 为 0.2374 → 0.2000 → 0.2770 → 0.3100，第二档回落。改为：

> and reaches 0.3100 (+148.77% over Most-Popular) in the Near-Warm Cold stratum; the dip in the Shallow Cold stratum (0.2000) reflects the transition from single-concept context to multi-concept disambiguation.

### 3.7 过度断言降调

| 位置 | 原文 | 建议 |
|---|---|---|
| 第 5.2 节 | This contrast demonstrates that... | This contrast is consistent with... |
| 第 5.4 节 | confirming that static priors must be... | This pattern indicates that static priors benefit from... |
| 结语 | Extensive experiments confirm statistically robust gains | Controlled experiments across six settings show gains in... |
| 摘要 | eliminate representation norm drift | limit representation norm inflation |
| 第 6.1 节 | moderates from X to Y | declines from X to Y |
| 第 6.1 节 | an inherent Pareto tension | an apparent trade-off |
| 第 6.1 节标题 | Candour-First Diagnostic Analysis | Warm-Start Advantage and Long-Horizon Coverage Trade-offs |

`Candour` 是全文唯一英式拼写，其余为美式，一并统一为 `Candor`。

### 3.8 方法章缺失内容

第 3.3 节未报告学习率、隐藏维度（表 2 有 `d=64`）、层数、dropout、批量、优化器、训练轮数。

第 3.4 节式(9) 的 `α_sim/α_prior/α_seed` 与式(11) 的三个 `γ` 全文无取值。

若第三类暂不执行，至少在第 4.2 节补超参数表，写出代码实际值：`alpha_skill=0.62`、`beta_gnn=0.38`、`gamma_pop=0.08`、`η_indeg=0.62`，并披露调参流程。

### 3.9 表 8 的种子重复

式(10) 定义可选集排除 `K_u`，但 Track 1 种子为 `Computer Systems`，S3 又是 `Computer Systems`；Track 3 种子 `Data Networks`，S2 又是它。

两种改法：改式(10) 允许种子重现，或改案例去掉重复。

另：三条路径种子概念 `dag_indeg` 全为 0，说明路径只在根节点集合内推进，拓扑约束未真正生效。需如实说明，或换非根出发点重跑。

---

## 四、第三类：必须重跑

四项不做，投稿必然被拒。审稿人拿到代码即可发现。

### 4.1 BG-GCN 必须真正训练

**现状** `bggcn_gates = {42:0.869, ...}` 与 `bggcn_ndcgs = {42:0.17362, ...}` 为字面量，嵌入由 `(0.72*e_t1 + 0.28*e_t3_base) * (1 + g*w)` 算术混合得到。文件内无 `nn.Module`、无 `backward()`、无优化器。§3.3 的式(5)至(8) 在代码中不存在。

**要做** 写真正训练的实现，包含：`Ã_msg` 两层传播、`φ(g)=2σ(g)` 门控、BPR 损失、Adam 优化器、逐 epoch 门控轨迹日志。跑五个种子，记录每次 `g*` 与 NDCG@10。

**工作量** 约 2–3 天。

**风险提示** 现有有界 tanh 与 sigmoid 仅差 0.16%，落在 1 个标准差内。真实训练后两者很可能统计上无差异。若如此，主线主张需收缩为「有界化本身抑制范数漂移，具体函数形式无显著差异」。这个结论更站得住，但摘要与贡献列表要重写。

### 4.2 冷启动上下文必须去泄漏

**现状** 第 313 至 330 行从评测真值取 `gt_list`，按 `signal_prob` 概率选目标，再用 `closest_neighbor` 反推它写入上下文。第 479 至 487 行又把真值还原为 `is_top_nbr` 特征送入打分。

**要做** 上下文只由训练集交互或独立冷启动画像构造，删掉 `closest_neighbor` 反查与 `is_top_nbr` 项，重跑 Exp3 与 Exp4。

**影响** 最大。复现结果显示：带泄漏 HR@10 为 0.2373，去掉后 0.1549，完全无个性化 0.1496。即 `+99.22%` 主卖点会大幅缩水，可能落到 +30% 左右。摘要、贡献 4、表 5、表 6、图 3、图 4 全部重写。

### 4.3 路径生成与违规统计必须隔离

**现状** 第 607、623 行 `topo_order_only` 与 `topo_cog_order` 的可行性判定使用 `(p, c) in eval_pos_pairs or (p, c) in adopt_pairs`。第 454 至 456 行统计 `Viol_iso` 时又用 `eval_pos_pairs`。用测试集生成路径、再用同一集合算违规率，`Viol_iso = 0.0000` 是构造结果，非方法性质。

**要做** 解码器只用 `G_DAG`，评测正例只在指标计算阶段读取。重跑后 `Viol_iso` 不为 0，需如实报告并解释量级。回退分支 `avail = rem` 要计数并报告触发频率。

### 4.4 统计检验必须补

**现状** `import scipy.stats as stats` 导入后从未调用。表 4 的 `***` 与 `p<0.0001`、表 5 表 6 的 `***` 均无实现。

**要做** 对每处跨方法比较做五种子配对 t 检验或 Wilcoxon，报告 t 值、自由度、效应量、95% 置信区间。

按现有数值复算得 t=14.187、p=1.43e-4，达不到 p<0.0001。若重跑后仍不达标，摘要的 p 值必须改。

### 4.5 顺带处理的四项

1. `defensive_guards.py` 的 `verify_preset_targets` 按预设阈值断言、不达标即抛异常，属「先设目标再调参至达标」。应删掉，改为数值健全性检查。
2. `run_exp3` 按测试集 HR@10 做网格搜索选参（第 492 至 500 行），应改为验证集选参并披露网格。
3. 表 7 的 FViol 列在 JSON 中存在（Full 变体在 M=5/10 为 0.0800/0.0796，为最高值），论文整列删除。应补回并解释拓扑约束引入伪违背的机制。
4. `s10_concept_bloom_meta.json` 自述为 `weak dictionary mapping, not human annotation`，论文写「based on topological depth and syllabus verb taxonomy」。应如实披露。

---

## 五、执行顺序

按依赖关系排列，前一步不完成，后一步做了要返工。

**第一步 重跑 4.1 至 4.4。** 决定表 4 至表 7 全部数值，也决定摘要能写什么。约两周。

**第二步 用新结果重建表 3 至表 8。** 写脚本从 JSON 直接渲染表格，不手工转录，并附逐格溯源表。

**第三步 重写摘要、贡献列表、第 5.2 至 5.6 节、第 6.1 节。** 以第二步数值为唯一依据。

**第四步 处理文献。** 13 篇核心文献中 9 篇元数据与 PDF 不符。已核实的例子：第 1 条 ConceptGCN 作者应为 Alatrash 而非 Wang，文章号 100193 而非 100226；第 8 条 Mathematics 12, 2324 而非 12(13) 1995；第 10 条为知识追踪加 LLM 增强而非 RL，Electronics 14, 4385。必须逐条以 PDF 首页或 DOI 解析结果重写，不得凭记忆补。同时删掉 `[01-Q1]` 类内部分区标签，改 Nature 格式并按首次出现重排，补齐 9 条未引用文献的引用或删除。

**第五步 格式。** 摘要压至 195 词内；展示项从 14 个减至 8 个以内（建议表 3、表 6、表 8 移入补充材料）；补 Author Contributions、Competing Interests、Ethics、Acknowledgements；数据与代码上传 Zenodo 取得 DOI 后写入声明。关键词数量与章节顺序两项，官方页面访问时跳转验证页，需投稿前自行核对。

**第六步 图表。** 图 2(b) 的分区标签 `Bounded Safe Codomain (0,2)` 与 `Norm-Inflation Regime (g*>2)` 数学上不成立，因 φ(g) 对任意 g 恒在 (0,2) 内，应改为经验阈值表述。图 6(a) 的中文术语全部译英文。图 5(a) 在 M=5 与 M=10 两条线像素级重合，需错位或换标记。六张图文件名与正文图号整体错位一位（`fig0` 对应 Figure 1，`fig5` 对应 Figure 2），按最终图号重命名。

---

## 六、路线选择

**路线一 完整执行。** 按上述方案做，工作量约 2–3 周，完成后是一篇扎实的 Q3/Q4 稿。核心主张会收缩，但每条都有代码和数据支撑。

**路线二 只做第一、二类。** 能提高形式合规度与内部一致性，但数值仍不可重建。且论文表格与 JSON 的偏差方向一致地偏向本文方法，这个模式被任何调代码的人发现，性质上不是排版错误。不建议。

**路线三 拆分投稿（建议评估）。**

| 分篇 | 内容 | 优势 | 前提 |
|---|---|---|---|
| 第一篇 | 职业技能标准到 MOOC 概念的对齐资源与双人审计协议 | 数据真实、κ 值可复算、有独立价值 | 需补 247 对标注原件（现有文件仅 92 条） |
| 第二篇 | 方法与实验 | 有充足时间完成第三类重跑 | 待重跑完成 |

拆分后第一篇投稿风险大幅下降，第二篇有充足准备时间。

---

## 附录 A：逐表订正清单

### 表 3（Exp 1）

| 指标 | 论文值 | 应改为 |
|---|---|---|
| S1 条款 / 边 | 257 / 1,542 | 需确认：源文件 260 条 |
| S2 条款 / 边 | 256 / 1,536 | 需确认：源文件 224 条 |
| S3 条款 / 边 | 247 / 1,482 | 需确认：源文件 200 条 |
| 合并条款 / 边 | 760 / 4,560 | 需确认：源文件 684 条 |
| De-Biased 条款 / 边 | 513 / 3,078 | 需确认 |
| Po / κ / κw | 88.26% / 0.6050 / 0.7501 | 由 247 条标注重算可复现，保留 |
| `prereq_adopted` Max SCC | 11 | **48** |
| `prereq_predicted` Max SCC | 542 | **576** |
| `chapter_weak` Max SCC | 494 | **576** |
| 去环后 Max SCC / DAG 边 | 1 / 162 | 正确，保留 |
| 与评测正例重叠 | 0 | 正确，保留 |

### 表 4（Exp 2）

| 项目 | 论文值 | 应改为 |
|---|---|---|
| BG-GCN NDCG@10 std | 0.00044 | 0.00043（实算） |
| 加粗范围 | 五列全给 BG-GCN | 仅 NDCG@10 给 BG-GCN，LightGCN 标 Upper Reference |
| 六个变体的 NDCG@5/20、HR@10、MRR | 有值 | 需确认来源，JSON 无对应字段 |
| 显著性标记 | `***` p<0.0001 | 待 4.4 完成后确定 |

### 表 5（Exp 3）

见 2.7 节。另：表 5 遗漏 JSON 中存在的 `Concept-ID` 方法，`7 methods` 的计数与 JSON 的 8 个方法不符。

### 表 6（Exp 4）

见 2.5 节。

### 表 7（Exp 5）

见 2.6 节。另：JSON 的 `exp5` 含 6 个变体，论文仅报 5 个，省略 `Unbounded-GNN + TopoSort (T3 Unnormalized)`。该变体在 M=15/20 的 SkillCov 为 0.9568 / 0.9655，同样高于 Full。需补入或说明剔除理由。

### 表 8（Exp 6）

30 个步骤与 JSON 逐格一致，零差错。唯一瑕疵为 Track 3 第 8 步 `数据` 译为 `Data Foundations`。

另：`monotonic Bloom progression (L3→L5→L6)` 表述需改。Track 1 第 3 步由 L3 跃至 L5，级差为 2，与式(11) 惩罚 `|Δb|>1` 的设定冲突。改称「阶梯式推进」。

---

## 附录 B：八位审稿人裁决汇总

| 审稿人 | 总体判断 | 达标度 | 是否读代码与 JSON |
|---|---|---|---|
| R1 数据与结果 | Reject | 1/5 | 是 |
| R2 方法与实验设计 | Reject | 1/5 | 是 |
| R3 结构与逻辑 | Major Revision | 3/5 | 否 |
| R4 英文表达 | Minor Revision | 3/5 | 否 |
| R5 文献与引用 | Major Revision | 2/5 | 部分（读 PDF） |
| R6 图表与可视化 | Major Revision | 3/5 | 部分（读图） |
| R7 格式与投稿规范 | Major Revision | 3/5 | 否 |
| R8 创新性与期刊匹配 | Reject | 1.5/5 | 是 |

**分歧来源** 四位判 Reject 的审稿人均读过代码与结果 JSON，四位判 Major Revision 的审稿人只读文本、图表与格式。只读文字的层面看到的是「需要大改」，读产物的层面看到的是「数值基础不成立」。

### 共识问题

| 问题 | 指出者 |
|---|---|
| 表 5、表 6、表 7 数值与 JSON 系统性不符 | R1、R2、R3、R8 |
| BG-GCN 未训练，指标为字面量常量 | R1、R2、R8 |
| 评测集正例参与路径生成 | R2、R8 |
| p<0.0001 无对应检验 | R1、R2、R3、R6、R8 |
| Max SCC 写作 11，实测 48 | R1、R2 |
| 9 条文献正文从未引用 | R3、R4、R5、R7 |
| 摘要约 240 词超 200 词上限 | R3、R4、R7 |
| `scales monotonically` 与表 6 矛盾 | R3、R4、R6 |
| 有界 tanh 与本文差 0.16%，落在噪声带 | R3、R8 |
| FViol 列被删，而该列上本文最差 | R2、R8 |
| 人工标注称 247 对四点量表，实际 92 条二值 | R1、R2、R8 |
| 论文称 BGE-M3，实际为 bge-small-zh-v1.5 | R1、R2、R8 |
| 七至十个超参数全文无取值 | R2、R3、R7 |

### 争议问题

**表 7 的 `w/o Topo` 行是复制错误还是选择性报告。** R1 判为复制错误，R8 判为方向反转的实质性隐瞒。倾向后者：JSON 实值在该行 M=15 为 0.9570、M=20 为 0.9677，远高于 Full 的 0.8119 与 0.6873。这个方向的差异恰好抹去「去掉拓扑约束覆盖更高」这一不利事实，四舍五入无法产生一致方向的偏差。

**表 5 的 0.9355 是否来自 exp5 串入。** JSON 实值 0.6564，而 0.9355 恰为 exp5 在 M=10 的 SkillCov，串入可能性大于独立笔误。

**英文维度是否只需小改。** R4 给 Minor Revision，与其余七位差距明显。该判断在只评语言时成立：全文无中式长句，无 `novel` 或 `state-of-the-art` 滥用，17 处相对增益复算无误。

### 投稿准备度评分

| 维度 | 评分 |
|---|---|
| 数据与结果 | 15 |
| 方法完整性 | 15 |
| 逻辑结构 | 45 |
| 英文表达 | 70 |
| 文献质量 | 30 |
| 图表质量 | 60 |
| 格式规范 | 45 |
| 创新性与期刊匹配 | 25 |
| **总分** | **38** |

---

## 附录 C：需人工核实事项

以下内容无法由现有材料判定，需提供原件。

1. `257/256/247`（合计 760）是否对应另一版本的技能文件。当前磁盘文件为 260/224/200。
2. `BGE-M3` 与 1024 维嵌入是否为另一次对齐实验的产物。当前 `s8_align` 记录为 `bge-small-zh-v1.5`、`topk=20`。
3. `28.5%` 与 `39.2%` 两个动机统计的原始出处。
4. `p<0.0001` 是否在项目外另行计算。若有，需提供检验脚本。
5. 表 4 六个变体的 NDCG@5、NDCG@20、HR@10、MRR 四列共 24 个数值的生成路径。
6. 247 对标注中两位专家的身份、资质与标注指南。
7. 表 5 中若干无源数值（0.3408、0.3257、0.0111、0.04543、0.6301）是否来自未归档的中间运行。
8. 3,412 名职业轨道学习者的筛选规则与判定文件。
9. `s8_align*.json` 的相似度分布无区分度（web 标准 p10–p90 仅 0.5164–0.5214），需确认 BGE 模型是否正确加载。
10. `s5_coldstart.jsonl.gz` 的生成脚本。该文件定义 10,748 名冷启动用户的评测真值，每用户恰 2 个概念，但生成代码不在交付物中。
11. Scientific Reports 当前对关键词数量、正文章节顺序、参考文献条数的具体规定。官方页面访问时跳转身份验证页，未能直接抓取。
12. 参考文献 6、7、12 三条 2026 年文献的 DOI、卷号真实性与正式出版状态。

### 核验声明

本方案的数值均来自实际读取的文件与脚本复算。已独立复核的项目包括：代码第 179 至 184 行的硬编码指标、第 607 与 623 行的评测集泄漏、`s8_align_annotation.json` 的 92 条标注、`exp3_coldstart_main.json` 的 SkillCov 实值、运行日志的 `max_scc_before=48`、五篇参考文献 PDF 的首页元数据、表 7 中 `w/o Topo` 行与 JSON 的对照、以及 `s6_graph.pt` 的 Tarjan 复算。

未虚构任何数据、实验结果或文献条目。文献补充仅给检索方向与年份范围。
