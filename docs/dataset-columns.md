# 训练 CSV 字段与接口

适用 `course-uci-20261014-v1`，CSV在 `data/public/training/`。连续任务每行按`f0..`列序组成`x`。特征已用正式train均值/尺度标准化，勿重复；下列单位指原始量，CSV特征无原始单位。`y`未标准化、不进`x`；预测不提供答案。

## LR：`lr-train.csv`

`f0..f29,y`，UCI WDBC细胞核图像特征。每块10列依次：radius（半径）、texture（纹理）、perimeter（周长）、area（面积）、smoothness（平滑度）、compactness（紧致度）、concavity（凹陷）、concave points（凹点）、symmetry（对称性）、fractal dimension（分形维数）。

- `f0..f9`：上述量的均值。
- `f10..f19`：标准误。
- `f20..f29`：worst，即各量最大三个值的均值。

`y=0`良性B，`y=1`恶性M；`LogisticRegression.fit(x,y,seed)`用训练答案，`predict(x)`按输入行序返回0/1。

## SVM：`svm-train.csv`

`f0..f56,y`，UCI Spambase邮件特征，原顺序分三组：

- `f0..f47`：48个指定词频率，原值=该词次数/总词数×100%；如`f0=word_freq_make`、`f1=word_freq_address`，不能重编号。
- `f48..f53`：字符`; ( [ ! $ #`频率，依此次序；原值=该字符次数/总字符数×100%。
- `f54..f56`：大写连续段长度平均值、最长值、总和（`capital_run_length_average/longest/total`）。

`y=1`垃圾邮件，`y=0`非垃圾邮件；`LinearSVM.fit(x,y,seed)`，`predict(x)`逐行返回0/1。

## GBDT：`gbdt-train.csv`

`f0,f1,f2,f3,y`，UCI CCPP第1工作表。原字段依次为AT环境温度（°C）、V排气真空度（cm Hg）、AP环境气压（毫巴）、RH相对湿度（%）。`y=PE`，每小时净电力输出（MW）。`GBDTRegressor.fit(x,y,seed)`，`predict(x)`逐行返回实数PE，不是类别。

## KMeans：`kmeans-train.csv`

仅`f0..f6`，UCI Seeds依次：面积A、周长P、紧致度C=4πA/P²、籽粒长度、宽度、不对称系数、沟槽长度。**无类别/答案列**。`KMeans.fit(x,3,seed)`；`predict(x)`逐行返回簇编号0..2，不代表籽粒品种类别。

## HMM：`hmm-train.csv`

固定合成监督序列，非UCI。列为`sequence_id,step,observation,state_label`：

- `sequence_id`：训练序列编号0..59，仅分组，不是特征。
- `step`：序列内步号0..19；重组60条整序列，不拆为独立样本。
- `observation`：符号编号0..3，组成`observations`。
- `state_label`：答案，隐藏状态编号0..2，组成`state_labels`；与符号是两套编号。

`DiscreteHMM.fit(observations,state_labels,3,4,seed)`：3状态/4符号。`predict(observations)`每条返回等长状态路径，编号沿用训练标签，不重编号。

来源：[WDBC](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names)、[Spambase](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names)、[CCPP](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)、[Seeds](https://archive.ics.uci.edu/dataset/236/seeds)。许可：同目录`DATA-SOURCES.md`。
