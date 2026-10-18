# ML Assignment 1 · C++17 机器学习算法实践

[下载实验材料](https://github.com/ZeroGen-Lab/ML-Assignment-1/archive/refs/heads/main.zip) · [v0.3.1 发行包](https://github.com/ZeroGen-Lab/ML-Assignment-1/releases/tag/v0.3.1) · [作业任务](#作业任务) · [评分方式](#评分方式) · [各项效果及格值](docs/trial-pass-thresholds.md) · [运行与提交](#运行与提交) · [学生指南](docs/student-guide.md)

本次作业要求使用 C++17 实现机器学习算法，完成数据读取、模型训练、预测与结果分析。

**实验材料已开放，可以开始实现与本地自测。** 下载后进入包含 `starter/`、`student_tools/` 和 `data/` 的目录，按下方命令检查环境、实现算法并打包源码。在线提交、成绩查询和排行榜暂未开放。

- **完成要求**：五项任务中任选四项；完成五项时，取最高四项成绩。
- **成绩构成**：每项满分 25 分，作业满分 100 分，占课程总评的 35%。
- **截止时间**：正式课程发布后 3 周（北京时间）。

## 作业任务

使用 **C++ 标准库**从零实现下列算法，不得使用第三方机器学习库。请保留模板中的 `assignment` 命名空间、类名及公开方法签名。

|任务|学习与实现重点|特征 / 结构|正式训练 / 测试|
|---|---|---|---|
|Logistic Regression|二分类概率建模、损失函数与优化|57 维，0/1 分类|1194 / 606 条|
|Linear SVM|线性分类、间隔与正则化|57 维，0/1 分类|1194 / 606 条|
|GBDT Regression|树模型与逐轮残差拟合|4 维，实数回归|1199 / 601 条|
|K-Means|聚类分配与中心更新|7 维，K=2|1199 / 601 条|
|Discrete HMM|监督参数估计与 Viterbi 路径解码|3 状态、4 观测符号，每条序列 20 步|60 / 30 条完整序列|

## 数据使用

请下载 **v0.3.1 配套学生包**。下文的 `data/` 路径均相对于学生包解压根目录；v0.3.0 的数据与本版不同，请勿混用不同版本的数据和工具。

### 公开训练数据

训练文件位于 `data/public/training/`：

|文件|输入与目标|
|---|---|
|`lr-train.csv`|`f0`…`f56`，目标 `y` 为 0/1|
|`svm-train.csv`|`f0`…`f56`，目标 `y` 为 0/1；与 LR 使用同一公开子集|
|`gbdt-train.csv`|`f0`…`f3`，目标 `y` 为实数|
|`kmeans-train.csv`|`f0`…`f6`，K=2，无类别标签列|
|`hmm-train.csv`|`sequence_id,step,observation,state_label`；每条序列 20 步|

LR、SVM、GBDT 与 K-Means 的公开训练文件各含 200 条记录，均为对应正式训练集的代表性子集。LR 与 SVM 共享数据划分、标准化方式及公开子集，仅需实现不同算法。HMM 的公开训练文件含 10 条完整序列，共 200 行。

连续特征已使用正式训练集的统计量进行 z-score 标准化，**无需再次标准化**。正式评测使用表中所列的完整训练集与测试集；正式测试数据及其真值不随学生包提供。

建议先从公开训练数据中划分训练集与验证集，检查实现并调整参数。HMM 必须按**完整序列**划分，不得将同一序列的不同位置分入训练集与验证集。详细字段与读取说明见 [数据列说明](docs/dataset-columns.md)。

### 公开诊断数据

`data/public/` 提供本地诊断输入、相应诊断答案与 `manifest.json`。诊断数据从同版公开训练子集中划分：前四项各使用 160 条记录训练、40 条记录预测；HMM 使用 8 条完整序列训练、2 条完整序列预测。K-Means 诊断不提供类别真值。

公开诊断用于检查接口、数据格式和示例行为，不替代自行划分的验证集或正式评测。请勿将诊断输入、答案或训练数据放入源码提交包，也不得硬编码答案。

## 评分方式

当前试运行采用以下已确定的规则：每项满分25分，由**算法效果18分、正确性4分、鲁棒性2分、效率1分**组成。算法效果按每项自己的综合质量指标Q换算；**不是将整个25分都按Q计算**。

|任务|综合质量指标Q|效果及格值T|达到T的效果分|
|---|---|---:|---:|
|LR|`(Accuracy + Macro-F1) / 2`|0.75|10.8 / 18|
|SVM|`(Accuracy + Macro-F1) / 2`|0.75|10.8 / 18|
|GBDT|`0.6/(1+NRMSE) + 0.4/(1+NMAE)`|0.60|10.8 / 18|
|K-Means|`0.6×ARI + 0.4×NMI`|0.50|10.8 / 18|
|HMM|`0.8×state_accuracy + 0.2×exact_sequence_accuracy`|0.60|10.8 / 18|

表中阈值针对完整Q，不是单独一个准确率、ARI或RMSE。GBDT归一化使用正式训练目标的统计量；聚类编号置换不直接算错。这些阈值是暂定教学标准，不是服务器硬件推导或五项实测拟合结果。

令 `q = min(1, max(0, Q))`，效果分按下式计算：

```text
q ≤ T：效果分 = 10.8 × q / T
q > T：效果分 = 10.8 + 7.2 × (q - T) / (1 - T)
任务分 = 效果分 + 正确性分 + 鲁棒性分 + 效率分
```

因此达到及格值拿10.8/18，超过它继续加分，Q达到1拿18分。例如LR/SVM的Q为0.80时，效果分为12.24/18。其余7分保留各自真实评测规则，不默认赠送；缺测分项显示未就绪，不能把未测CPU效率补成成绩。效果分已测得时可先展示，完整任务分待所需分项齐备后求和。

合法选择四项或五项，完成五项时取最高四项，作业满分100分，占课程总评35%；同一任务各组成分来自同一次提交。该试运行方案**不再等待随机baseline或班级前10%统计进行二次换分**。原始指标、分项与政策版本随提交持久保存，目标是同一提交在刷新、重开查询后保持一致。

政策版本：`trial-quality18-anchors-20261018-v1`。完整公式、例子和未就绪情况见 [五项效果及格值与18分换算](docs/trial-pass-thresholds.md)。旧日期评分概要保留为历史版本，不作为当前试运行的换分规则。及格值只在文档说明，排行榜不添加虚构baseline或及格线成绩。

**在线提交、评测得分和持久查分仍在联调，尚未开放。** 当前v0.3.1发行包用于实现算法与本地自测；试运行入口和兼容工具包将另行公布。请勿把本地自测结果当作课程成绩，也不要向猜测地址提交源码或身份信息。

## 实现与提交文件

```text
starter/
  include/assignment.hpp
  src/assignment.cpp
  README.md
  AI_USAGE.md
student_tools/         # 本地检查与打包工具
data/public/training/  # 公开训练 CSV 与 manifest
data/public/           # 公开诊断数据
docs/                  # 学生指南、字段说明与评分规则
```

- 在 [assignment.cpp](starter/src/assignment.cpp) 中实现算法，公开接口以 [模板头文件](starter/include/assignment.hpp) 为准。
- 可调整 private 成员，并在 `include/`、`src/` 中增加符合要求的标准库辅助源码。GBDT 不限定为模板中的树桩结构。
- 配套 driver 负责公开 API 适配，无需自行编写评测命令行程序，也不应将 driver 或工具加入提交包。
- 完成 `starter/README.md` 与 `starter/AI_USAGE.md`，不要保留未填写的模板内容。
- 提交 ZIP 根目录须含 `README.md`、`AI_USAGE.md`、`include/assignment.hpp`、`src/assignment.cpp`；其他文件仅允许为 `include/` 下的 `.h/.hpp` 和 `src/` 下的 `.cpp`。
- 不得提交数据、答案、模型、可执行文件、凭据或第三方机器学习库。

文件数量、大小与格式限制见 [模板说明](starter/README.md) 和 [学生指南](docs/student-guide.md)。

## 环境配置

本地需要 **Python 3.10+** 与支持 **C++17** 的 `g++` 或 `clang++`。工具不会自动安装编译器；正式评分统一使用服务器端 Linux / GCC 12 环境。

- **macOS**：可使用 Apple clang++。
- **Linux**：可使用 g++。
- **Windows**：可使用原生 Python、PowerShell 与 MinGW-w64 g++，或兼容的 clang++；WSL2 为可选环境。MSVC `cl` 不兼容本作业的自测命令。

本地自测会编译并执行 C++ 程序，**不是安全沙箱，请仅运行自己编写或已审查的代码**。

## 运行与提交

以下命令均在 **v0.3.1 学生包解压根目录**执行。请将示例信息替换为本人信息，学号按字符串填写并保留前导零。环境变量只在当前终端有效；填写学号和姓名不代表已完成身份认证。

<details>
<summary><strong>macOS / Linux · Bash</strong></summary>

```sh
export STUDENT_ID=2026123456
export STUDENT_NAME="Zhang San"
export STUDENT_NICKNAME="zs"
python3 -B -m student_tools config

# 实现算法后检查；--tasks 可选任意四项或五项
python3 -B -m student_tools.selftest --project starter --public-data data/public --tasks lr,svm,gbdt,kmeans,hmm
# 已有 clang++ 时可追加 --compiler clang++

python3 -B -m student_tools package --project starter --output submission.zip
```

</details>

<details>
<summary><strong>Windows · PowerShell 本地检查与打包</strong></summary>

```powershell
$env:STUDENT_ID = "2026123456"
$env:STUDENT_NAME = "Zhang San"
$env:STUDENT_NICKNAME = "zs"
python -B -m student_tools config
python -B -m student_tools.selftest --project starter --compiler g++ --public-data data/public --tasks lr,svm,gbdt,kmeans,hmm
python -B -m student_tools package --project starter --output submission.zip
```

若使用 Python Launcher，可将 `python` 替换为 `py -3`。

</details>

本地检查完成后，打包生成 `submission.zip`，作为本次作业的源码提交文件。再次打包时，请使用新的输出文件名。每次提交均须如实选择 `ai_assisted` 或 `independent`；`AI_USAGE.md` 不能替代提交标签。身份信息、每日提交次数、待处理任务限制及超时重试、幂等要求见 [学生指南](docs/student-guide.md)。

本地自测通过或打包成功不代表获得正式成绩。请勿向未公布的地址发送身份信息或源码。

## 排行榜与反馈

排行榜展示排名、昵称、各任务成绩和总成绩。页面提供“刷新”按钮；没有已公布成绩时，显示“成绩暂未公布。”。

对评测结果有疑问时，请保留提交编号与相关反馈信息，便于核查。
