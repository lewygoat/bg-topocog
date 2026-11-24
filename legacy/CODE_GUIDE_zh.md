# 代码说明

**项目**：融合有界门控图神经网络与拓扑-认知双约束的冷启动学习路径推荐研究
**整理日期**：2026-11-20
**运行环境**：Python 3.9.6 / torch 2.8.0 / numpy 2.0.2 / scipy 1.13.1 / matplotlib 3.9.4
**硬件**：Apple M4，10 核，32 GB，MPS 可用

---

## 一、代码清单

项目共 5 个 Python 脚本，合计约 2,284 行。

| 脚本 | 行数 | 路径 | 职责 | 状态 |
|---|---|---|---|---|
| `run_bg_topocog_experiments.py` | 1,077 | `实验/代码/` | 六项实验主流程 | 可用，含已知缺陷 |
| `defensive_guards.py` | 255 | `实验/代码/` | 断言、监控、Tarjan | 可用，含需删除的函数 |
| `generate_additional_figures.py` | 478 | `实验/代码/` | 图 1、图 2、图 6 生成 | 可用 |
| `export_and_verify_manuscript.py` | 240 | `实验/代码/` | Markdown 转 Word | 可用，含硬编码路径 |
| `bggcn_train.py` | 234 | `实验/重跑/` | BG-GCN 真实训练 | 冒烟通过，未跑整体 |

---

## 二、主流程脚本

### run_bg_topocog_experiments.py

**入口**

```
python 实验/代码/run_bg_topocog_experiments.py --smoke-test   # 冒烟，约 2 秒
python 实验/代码/run_bg_topocog_experiments.py --full          # 全量，约 17 秒
```

**模块划分**

| 行号 | 函数 | 职责 |
|---|---|---|
| 58 | `load_and_verify_foundation` | 加载图与先验，跑泄漏断言与 Tarjan 凝聚 |
| 141 | `run_exp1_and_exp2_bggcn` | Exp1 对齐审计；Exp2 门控变体 |
| 247 | `run_exp3_and_exp4_coldstart` | Exp3 冷启动主实验；Exp4 稀疏分层 |
| 566 | `run_exp5_multistep_path_ablation` | Exp5 四视界路径规划消融 |
| 742 | `run_exp6_vocational_tracks` | Exp6 三轨道案例 |
| 801 | `generate_figures_and_markdown` | 生成汇总 Markdown 与报告 |
| 1029 | `main` | 串联六个实验 |

**执行顺序**

```
load_and_verify_foundation  →  Exp1+Exp2  →  Exp3+Exp4  →  Exp5  →  Exp6
                                      ↓
                          generate_figures_and_markdown
```

**输入依赖**（全部位于 `实验/中间产物/`）

| 文件 | 内容 | 是否由本项目代码生成 |
|---|---|---|
| `s6_graph.pt` | 图结构、交互切分、先修边 | 否，外部预置 |
| `s9_skill_weight*.npy` | 五组技能权重向量 | 否，外部预置 |
| `s10_concept_bloom.npy` | Bloom 层级 | 否，外部预置 |
| `s5_coldstart.jsonl.gz` | 冷启动用户与真值 | 否，外部预置 |
| `嵌入/emb_t1_s*.pt`、`emb_t3_s*.pt` | 基线嵌入 | 否，外部预置 |

**输出**

`实验/结果/` 下六个 exp JSON、`all_experiments_master.json`、`smoke_test_report.json`；`实验/图表/` 下六张 PNG。

---

## 三、辅助模块

### defensive_guards.py

| 行号 | 名称 | 用途 | 建议 |
|---|---|---|---|
| 16 | `GuardViolation` | 异常类 | 保留 |
| 21 | `ExperimentMonitor` | 结构化日志 | 保留 |
| 53 | `atomic_write_json` | 原子写 | 保留 |
| 61 | `atomic_save_torch` | 原子存张量 | 保留 |
| 69 | `check_disk_space` | 磁盘检查 | 保留 |
| 76 | `assert_no_leakage` | 切分重叠断言 | 保留 |
| 99 | `assert_finite_array` | NaN/Inf 与坍缩检查 | 保留 |
| 111 | `tarjan_scc_and_condense_dag` | Tarjan 凝聚与 Kahn 校验 | 保留，核心 |
| 205 | `verify_preset_targets` | 按预设阈值断言达标 | **建议删除** |

`verify_preset_targets` 按硬编码阈值（如 NDCG ≥ 0.1730、SkillCov ≥ 0.8800）断言，不达标抛异常。这是「先设目标再调参至达标」，两轮评审均判为研究设计倒置。

### generate_additional_figures.py

| 行号 | 函数 | 产出 |
|---|---|---|
| 54 | `generate_fig0_architecture_and_motivation` | 图 1 架构图 |
| 249 | `generate_fig4_vocational_tracks_case` | 图 6 轨道案例 |
| 398 | `generate_fig5_bggcn_gate_dynamics` | 图 2 门控动力学 |

图 3、图 4、图 5 由主流程的 `generate_figures_and_markdown` 生成。

本次已修正三处：`ROOT_DIR` 从 Windows 硬编码路径改为相对路径推导；图 1 的 `Max SCC = 11` 改为 48；图 6 的中文概念名改为英文。

### export_and_verify_manuscript.py

Markdown 转 Word 并做一致性审计。第 14 行的 `ROOT_DIR` 仍为 Windows 硬编码路径，在其他机器上无法直接运行。

### bggcn_train.py（本次新增）

| 部分 | 行号 | 说明 |
|---|---|---|
| `Monitor` | 18 | 逐行 JSON 日志 |
| `build_adj` | 43 | 先修图邻接矩阵对称归一化 |
| `BoundedGateGCN` | 66 | 两层传播 + 有界门控 |
| `bpr_loss` | 100 | 成对 BPR |
| `evaluate` | 110 | HR@10 / NDCG@10 / MRR |
| `train_one` | 135 | 单种子训练与早停 |
| `main` | 200 | 命令行入口 |

**支持四种门控**：`sigmoid`（本文）、`tanh`、`clip`（硬截断）、`unbounded`（基线）。

**命令行**

```
python 实验/重跑/bggcn_train.py --smoke --modes sigmoid --seeds 42 --epochs 8
python 实验/重跑/bggcn_train.py --modes sigmoid,tanh,unbounded --seeds 42,43,44,45,46 --epochs 30
```

**已知限制**：MPS 后端不支持稀疏张量，邻接矩阵改用稠密矩阵（576×576，开销可忽略）。

---

## 四、代码覆盖缺口

以下环节论文有描述，但项目内没有对应实现。

| 缺口 | 论文位置 | 现状 |
|---|---|---|
| Stage I 技能-概念对齐 | §3.2 式(1)(2) | **无代码**。全文无 BGE 编码调用（`grep -c "SentenceTransformer\|encode(\|bge"` 结果为 0），对齐结果从 `s8_align*.json` 预加载 |
| 技能条款抽取 | §4.1 | 无代码，`s7_skills*.json` 为外部预置 |
| Bloom 层级标注 | §3.4 | 无代码，`s10_concept_bloom.npy` 为外部预置 |
| 冷启动用户划分 | §4.1 | 无代码，`s5_coldstart.jsonl.gz` 为外部预置 |
| 图构建与交互切分 | §4.1 | 无代码，`s6_graph.pt` 为外部预置 |

也就是说，三阶段流程中只有 Stage II 的表示层与 Stage III 的规划层在本项目内，Stage I 完全依赖外部产物。这与论文第 3 章把三阶段描述为同一框架存在落差。

---

## 五、已知代码缺陷

按严重程度排序。

| 编号 | 位置 | 问题 | 影响 |
|---|---|---|---|
| 1 | 第 179–183 行 | `bggcn_gates`、`bggcn_ndcgs`、`bggcn_hrs`、`bggcn_mrrs` 为硬编码字典 | 表 4 的门控值与指标非训练所得 |
| 2 | 第 313–330 行 | 冷启动上下文由评测真值经 `closest_neighbor` 反推 | 冷启动指标含泄漏 |
| 3 | 第 607、623 行 | `topo_*_order` 用 `eval_pos_pairs` 判可行性 | 路径生成读取测试集 |
| 4 | 第 454–456 行 | `Viol_iso` 用同一集合统计 | 违规率由构造保证为零 |
| 5 | 第 927–933 行 | 六个变体的 `HR@10`、`MRR` 为字符串常量 | 表 4 中 12 格无源 |
| 6 | 第 205 行（guards） | `verify_preset_targets` 按预设阈值断言 | 研究设计倒置 |
| 7 | 第 492–500 行 | 按测试集 HR@10 网格搜索选参 | 测试集调参 |
| 8 | 第 912–914 行 | 汇总表内嵌 257/256/247 字面量 | 与源文件 260/224/200 不符 |
| 9 | 第 14 行（export） | `ROOT_DIR` 为 Windows 路径 | 跨平台不可运行 |

---

## 六、运行方式（当前可用）

```bash
cd "/Volumes/拓展盘/扫地僧/SCI/慕尼黑/王璐/融合有界门控图神经网络与拓扑-认知双约束的冷启动学习路径推荐研究"

# 复现全部表格与图件（约 17 秒）
python 实验/代码/run_bg_topocog_experiments.py --full

# 重新生成三张补充图
python 实验/代码/generate_additional_figures.py

# BG-GCN 训练冒烟（约 16 秒）
python 实验/重跑/bggcn_train.py --smoke --modes sigmoid --seeds 42 --epochs 8
```

**注意**：`--full` 会调用 `verify_preset_targets`，若某指标低于预设阈值将抛 `GuardViolation` 并中止。

---

## 七、建议的整理顺序

若要提升代码质量，按此顺序：

1. 删除 `verify_preset_targets`，改为区间健全性检查（第 205 行，guards）
2. 补齐 Exp2 的训练实现，替换第 179–183 行硬编码（`bggcn_train.py` 已有雏形）
3. 移除冷启动上下文的 `closest_neighbor` 反查（第 313–330 行）
4. 隔离路径生成与指标统计的评测集（第 607、623、454 行）
5. 补 `requirements.txt`
6. 修正 `export_and_verify_manuscript.py` 的 `ROOT_DIR`
7. 补 Stage I 的对齐代码，或将论文第 3 章改为声明其为外部预处理
