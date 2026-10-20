# 数据列说明：新版五任务公开训练材料

2026-10-20。**新版数据、模板、工具同步中。** 本页描述拟采用合同，不证明这些文件已进入当前公开 main。旧 `data/public/training` 与旧 `v0.3.1` 只作历史，不与新版混用。

## 拟采用目录

| 任务 | 拟采用 CSV 路径 | 训练行数 | 列与标签 |
| --- | --- | --- | --- |
| LR | `data/training/lr/train.csv` | 200 | `f0,...,f56,y`，57 个特征，末列二分类标签 |
| SVM | `data/training/svm/train.csv` | 200 | 同 LR；共享原训练记录 |
| GBDT | `data/training/gbdt/train.csv` | 200 | `label,x000,...,x855`，首列九类标签，其余 856 个真实词频特征 |
| KMeans | `data/training/kmeans/train.csv` | 200 | `f0,...,f6`，7 个特征，无类别标签列 |
| HMM | `data/training/hmm/train.csv` | 200 | `sequence_id,step,observation,state_label`，10 条完整序列 × 20 步 |

拟采用五任务合并元数据为 `data/training/manifest.json`，署名、许可与变更记录为 `data/training/DATA-SOURCES.md`；GBDT 单项公开元数据保留 `data/training/gbdt/manifest.json`。**以实际同步后的同版文件为准**，不是把旧总 manifest 改名后直接使用。旧总 manifest 包含旧 GBDT 回归数据，不代表新版。

## LR / SVM：Spambase

200 条公开训练记录，57 维，末列 `y` 为整数 `0/1`。LR 与 SVM 共享相同固定 CSV 记录，不能视为两个独立数据集。

特征沿用原 formal-train 统计标准化后的固定值；不能声称标准化只由这 200 条公开记录拟合，也不要擅自重复标准化。本次复用不重新清理、取样或 fit 变换。

来源：[UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase)，DOI `10.24432/C53G6X`。作者 Hopkins, M.; Reeber, E.; Forman, G.; Suermondt, J.。固定来源记录为 CC BY 4.0，完整署名与变更沿配套来源文档。

## GBDT：CNAE-9 九类分类

必须保留 **856** 个真实词频特征，标签从原 `1..9` 映射为 **0..8**。CSV 首列 `label`，后续顺序为 `x000` 至 `x855`。特征为非负整数词频；不是旧回归数据，也不接受浮点回归预测。

公开训练 200 条，类别配额为 `23,23,22,22,22,22,22,22,22`。200 条子集中只有部分列出现非零值，不改变每行 856 维合同，不补维、不裁列。来源：[UCI CNAE-9](https://archive.ics.uci.edu/dataset/233/cnae+9)，DOI `10.24432/C51G7P`，Ciarelli, P. & Oliveira, E. (2009)，固定来源记录为 CC BY 4.0。

公开源可识别，不承诺抗查表的秘密 holdout。本地公开 GBDT 检查只将 200 条训练自身作为查询，只核输出数量、整数和标签域；常量合法输出也可通过，**不是泛化分或正式参考分**。本页不提供教师评估记录、选择索引或隐藏标签。

## KMeans：Rice Cammeo and Osmancik

200 条公开训练记录，7 维，无类别标签列。`f0,...,f6` 依次对应 Area、Perimeter、Major_Axis_Length、Minor_Axis_Length、Eccentricity、Convex_Area、Extent。沿用原训练集 z-score 变换后的固定字节，不重新拟合。`k=2` 沿公开合同，不从 CSV 伪造评估 truth。

来源：[UCI Rice Cammeo and Osmancik](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik)，UCI DOI `10.24432/C5MW4Z`；Cinar and Koklu (2019)，论文 DOI `10.18201/ijisae.2019355381`。固定来源记录为 CC BY 4.0。

## HMM：课程生成的监督序列

200 行不是 200 条独立序列，而是 **10 条完整序列，每条 20 步**。同一 `sequence_id` 内按 `step=0..19` 保持顺序；`observation` 域为 `0..3`，`state_label` 域为 `0..2`。监督 fit 同时使用真实观测与状态标签；状态含义跨序列一致，不任意拆分或拼接。

复用既有课程生成训练 CSV，不重新生成，不公开生成参数或教师评估 truth。**独立许可证据未在固定来源记录中注明，许可状态 PARTIAL，license 为 null**；它不是上述 UCI 数据，不能自动套用 CC BY 4.0。来源整理不等于补造授权，最终公开权利表述需单独明确。

## 版本与执行边界

数据目录存在、CSV 哈希一致或新文档发布，均不证明服务器已经使用同版正式评估集、源码已合规、算法有效或正式成绩可用。未运行指标保持 `NOT_RUN`，缺测保持 `None/null`。baseline 后置，不将公开自测当实测。

返回[学生指南](student-guide.md)或[当前试运行评分说明](current-trial-scoring.md)。
