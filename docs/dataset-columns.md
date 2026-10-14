# 训练 CSV 字段与接口

适用 `course-uci-20261014-v3`，数据身份 `0f23098f6a76cf60424ca04bc1bfe1d7e96ad1ef1a59a632dd2b319ab162c671`（配套 v0.3.1 尚未发行，已发布 v0.3.0 仍为v1，不混用）；CSV位于 `data/public/training/`。五文件各200数据行：连续任务各200条正式train代表性子集，HMM10完整序列×20步。服务器正式fit不是公开200：LR/SVM1194train/606test、GBDT1199/601、Rice1199/601；HMM60/30完整序列。test与教师答案不随课程材料发布。

连续任务按`f0..`列序组成`x`。特征已用**正式train**均值/尺度标准化，勿再次缩放；以下单位指原始量，CSV数值已无原始单位。`y`未标准化、不进`x`；预测不提供答案。公开200可自行划验证集；提交后服务器使用完整正式train训练。

## LR与SVM：`lr-train.csv`、`svm-train.csv`

两文件逐字相同，均为`f0..f56,y`，UCI Spambase邮件特征；同一正式split、同一标准化、同一公开200。分别实现LogisticRegression与LinearSVM，不是两个独立数据来源。

- `f0..f47`：48个指定词频率，原值=该词次数/总词数×100%；如`f0=word_freq_make`、`f1=word_freq_address`，不能重编号。
- `f48..f53`：字符`; ( [ ! $ #`频率，依此次序；原值=该字符次数/总字符数×100%。
- `f54..f56`：大写连续段长度平均值、最长值、总和（`capital_run_length_average/longest/total`）。

`y=1`垃圾邮件，`y=0`非垃圾邮件；`LogisticRegression.fit(x,y,seed)`或`LinearSVM.fit(x,y,seed)`，`predict(x)`按输入行序返回0/1。

## GBDT：`gbdt-train.csv`

`f0,f1,f2,f3,y`，UCI CCPP第1工作表。原字段依次为AT环境温度（°C）、V排气真空度（cm Hg）、AP环境气压（毫巴）、RH相对湿度（%）。`y=PE`，每小时净电力输出（MW）。`GBDTRegressor.fit(x,y,seed)`，`predict(x)`逐行返回实数PE，不是类别。

## KMeans：`kmeans-train.csv`

仅`f0..f6`，UCI Rice Cammeo and Osmancik形态特征，依次为：

| 列 | 原字段 | 含义 |
|---|---|---|
| f0 | Area | 稻粒区域像素面积 |
| f1 | Perimeter | 稻粒边界周长 |
| f2 | Major_Axis_Length | 等效椭圆长轴长度 |
| f3 | Minor_Axis_Length | 等效椭圆短轴长度 |
| f4 | Eccentricity | 等效椭圆偏心率 |
| f5 | Convex_Area | 凸包像素面积 |
| f6 | Extent | 稻粒面积/外接矩形面积 |

**无类别/答案列**。`KMeans.fit(x,2,seed)`；`predict(x)`逐行返回簇编号0..1，簇编号不等于品种标签，允许置换。教师品种标签仅用于划分、审核与ARI/NMI评价，不进入公开fit。正式数据选1800真行，近重复canonical组不跨train/test；不复制补样本。

## HMM：`hmm-train.csv`

固定合成监督序列，非UCI。列为`sequence_id,step,observation,state_label`：

- `sequence_id`：公开序列本地编号0..9，只用于分组，不是特征，也不是教师内部sequence_id。
- `step`：序列内步号0..19；重组10条完整公开序列，不拆为独立样本。服务器正式train60条、test30条，各20步。
- `observation`：符号编号0..3，组成`observations`。
- `state_label`：监督隐藏状态编号0..2，组成`state_labels`；与符号是两套编号。

`DiscreteHMM.fit(observations,state_labels,3,4,seed)`：3状态/4符号。`predict(observations)`每条返回等长状态路径，编号沿用训练标签，不重编号。真实参数与正式test状态路径不发布。

来源：[Spambase](https://archive.ics.uci.edu/dataset/94/spambase)、[CCPP](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)、[Rice](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik)。许可与引用见[数据来源与许可](../data/public/training/DATA-SOURCES.md)。公开原始UCI标签可能找回，课程holdout未发布不等于保密或反作弊保证。旧v1/v2产物保持原版，不与v3混用。
