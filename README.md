# ML Assignment 1 · C++17 算法实践

[下载配套 v0.3.1 学生包（v3；发行后可用）](https://github.com/ZeroGen-Lab/ML-Assignment-1/releases/tag/v0.3.1) · [评测与评分](#评测指标与评分方式) · [排行榜说明](#排行榜) · [学生指南](docs/student-guide.md)

> **截止时间**：正式课程发布后 **3 周**，具体 Asia/Shanghai 时刻以课程公告为准。排行榜访问地址尚未公布，不向猜测地址发送身份或源码。
>
> **成绩构成**：每项 25 分，任选四项；实现五项时取最高四项，作业满分 100 分，占课程总评 35%。新版候选已完成 Windows 原生客户端合成测试与可信模板编译；v0.3.1 尚未发行，真实服务器提交与线上评测仍未验，文档发布不代表正式提交已开放。

## 作业任务

使用 **C++ 标准库**从零实现机器学习算法，理解数据处理、训练、预测与评测流程。保留 `assignment` 命名空间、类名和公开方法签名；模板未实现算法，不提交第三方机器学习库、训练数据、模型或二进制。

|任务|学习与实现重点|数据来源|特征 / 结构|正式训练 / 测试|
|---|---|---|---|---|
|Logistic Regression|二分类概率建模、损失与优化|Spambase，与SVM共源共划分|57 维，0/1 分类|1194 / 606 条|
|Linear SVM|线性分类、间隔与正则化|Spambase，与LR共源共划分|57 维，0/1 分类|1194 / 606 条|
|GBDT Regression|树模型与逐轮残差拟合|CCPP|4 维，实数回归|1199 / 601 条|
|K-Means|聚类分配与中心更新|Rice Cammeo/Osmancik|7 维，K=2，无公开类别标签|1199 / 601 条|
|Discrete HMM|监督参数估计、Viterbi 路径解码|监督序列|3 状态、4 观测符号，每条 20 步|60 / 30 条完整序列|


## 数据字段

详细字段、目标列、取值范围与读取注意事项见 [数据列说明](docs/dataset-columns.md)。

|训练文件（学生包 `data/public/training/`）|输入与目标|
|---|---|
|`lr-train.csv`|`f0`…`f56`，目标 `y` 为 0/1；与SVM同一公开子集|
|`svm-train.csv`|`f0`…`f56`，目标 `y` 为 0/1|
|`gbdt-train.csv`|`f0`…`f3`，目标 `y` 为实数|
|`kmeans-train.csv`|`f0`…`f6`，没有类别标签列；K=2|
|`hmm-train.csv`|`sequence_id,step,observation,state_label`；每序列 20 步|

新版配套公开诊断位于 `data/public/`，包含输入、诊断答案与 `manifest.json`。诊断从同版公开训练子集中划分，不是正式留出测试；LR、SVM、GBDT、KMeans各160条fit/40条predict，HMM为8条完整fit序列/2条完整predict序列；具体内容与身份见包内同版manifest。HMM必须按整条序列划分。当前新版尚未发行；KMeans诊断不公开类别真值，不要将诊断数据或答案装入源码提交ZIP。

## 评测指标与评分方式

### 各任务效果指标

|任务|指标|中文解释与方向|代码中的效果质量 `q`|
|---|---|---|---|
|LR / SVM|Accuracy、Macro-F1|Accuracy 是分类正确比例；Macro-F1 对各类别 F1 等权平均，兼顾不同类别。均越高越好。|`0.5 × Accuracy + 0.5 × Macro-F1`|
|GBDT|MAE、RMSE|MAE 是平均绝对误差；RMSE 是均方误差的平方根，对较大误差更敏感。均越低越好。|`0.6 / (1 + NRMSE) + 0.4 / (1 + NMAE)`；分别按训练目标总体标准差、围绕训练均值的平均绝对偏差归一化，退化时使用代码规定的小正数下限|
|KMeans|ARI、NMI|ARI 衡量扣除随机一致性的聚类一致程度；NMI 衡量聚类与类别信息的一致程度。均越高越好；聚类编号置换不直接算错。|`0.6 × ARI + 0.4 × NMI`|
|HMM|逐位置状态准确率、整条路径准确率|前者统计状态位置预测正确比例；后者要求一条序列全部位置正确。均越高越好；解码要求 Viterbi 联合最优路径。|`0.8 × state_accuracy + 0.2 × exact_sequence_accuracy`|

这些效果指标不等于最终任务分。正式效果、正确性、鲁棒性与效率共同形成综合表现；本机 `SELFTEST PASSED` 不是算法质量合格线，也不是课程成绩。

### 先合综合原始表现，再校准任务分

|组成|原始表现满额|含义|
|---|---|---|
|效果|18|由上表质量 `q` 转换，质量越好得分越高|
|正确性|4|接口与算法行为的正确性诊断|
|鲁棒性|2|边界与异常输入等诊断|
|效率|1|按正式评测的 CPU 证据与参考标定计入，不使用本机自报时间|

当前实现先计算综合原始表现 `R`：

```text
L = -0.6（KMeans），其它任务 L = 0
Q = min(1, max(0, (q - L) / (1 - L)))
R = 18 × Q + 正确性分 + 鲁棒性分 + 效率分
```

因此不能将“效果单项的 p90”当作本年度最终校准。当前实现对**各任务的综合原始表现**校准：每位同学在每个任务最多贡献一次有有效效果的最佳综合原始表现，计算这些样本的第 90 百分位；正常校准时以此形成满档阈值 `T`。`B` 是同套评测协议下随机预测程序的综合原始表现参考档，不是多数类、训练均值等旧无信息诊断。各组成分来自同一次提交，不能拼接多次提交的子分。

|综合表现区间|任务最终分（满分 25）|
|---|---|
|`R ≥ T`|25 分；正常校准时为综合前 10% 满档，阈值处并列也进入满档|
|`B ≤ R < T`|`15 + 10 × (R - B) / (T - B)`，在基础档与满档之间线性变化|
|`R < B`（`B > 0`）|`15 × R / B`，低于基础档按比例递减|

随机基础档对应 **15/25**，不是效果部分另加 15 分。可信完成但无有效效果的失败、无可用提交的缺项按 0 处理；仍 pending 的有效记录使结算暂缓，不产生分数或排名，平台证据缺失不冒充学生 0 分。若 `B=0`，有效分数的低于基础档分支不会出现。若没有有效样本，或综合 p90 不高于随机基础档，实现采用 `T=25` 的回退，不自动给满分，也不保证前 10% 满档；公式要求 `0 ≤ B < T`。**真实基础档数值、CPU 参考标定及正式评测尚未完成，尚未开放**，这里解释当前实现，不承诺已经测得阈值或已经产生正式成绩。

五项全部完成时，例如最终任务分为 `22、21、20、19、15`，取最高四项得到作业 **82/100**；按 35% 权重计入课程总评，贡献 28.7 分。只选四项则直接相加。详细要求见 [评分规则](docs/assignment-scoring-rules-20261010.md)，正式开放、截止与反馈渠道以课程公告为准。

## 项目目录

```text
starter/               # 算法实现与源码提交说明
  include/assignment.hpp
  src/assignment.cpp
  README.md / AI_USAGE.md
student_tools/         # 同一套 Python 本地客户端
data/public/training/  # 学生包：五份训练 CSV、manifest 与来源许可
data/public/           # 学生包：公开诊断输入与 manifest.json
docs/                  # 字段、学生指南与评分规则
public/                # 静态排行榜页面
```

上面两个 `data/` 路径均相对于**学生包解压根目录**。直接阅读仓库文档不代替下载完整 Release。

## 接口与源码提交

- 保持 `assignment` 命名空间、类名、public 方法参数与返回类型。具体接口见 [模板头文件](starter/include/assignment.hpp)，实现写入 [assignment.cpp](starter/src/assignment.cpp)。
- 可调整 private 成员，并在 `include/`、`src/` 增加允许的标准库辅助源码。GBDT 不限定为模板树桩结构。
- 模板 driver 负责公开 API 适配，学生无需另造评测 CLI，不将 driver 或工具装入源码 ZIP。
- 填写 `starter/README.md` 与 `starter/AI_USAGE.md`，不能把待填模板当成最终说明。
- ZIP 根须含 `README.md`、`AI_USAGE.md`、`include/assignment.hpp`、`src/assignment.cpp`；其余仅允许 `include/` 中 `.h/.hpp` 与 `src/` 中 `.cpp`。

完整路径、文件数量、大小与格式限制见 [模板说明](starter/README.md) 和 [学生指南](docs/student-guide.md)。数据、答案、模型、可执行文件、凭据与第三方机器学习库不入源码提交包。

## 环境配置

需要 **Python 3.10+** 与兼容 **C++17** 的 `g++` 或 `clang++`；工具不会自动安装编译器。服务器 Linux / GCC 12 统一评分，同一 Python 客户端生成同一格式源码 ZIP，不需要三个平台各自的评测二进制。

- **macOS**：可用 Apple clang++。
- **Linux**：可用 g++。
- **Windows**：原生 Python、PowerShell 与 MinGW-w64 g++ / 兼容 clang++；MSVC `cl` 不兼容当前自测命令。WSL2 可选，不是必需。

本机自测会编译并执行 C++，**不是安全沙箱，只运行自己编写或已审查的代码**。

## 运行与提交

本节命令在配套学生包根执行。当前 Release 中的 `student-release-0.3.0.zip` 对应旧版 v1，不对应上表 v3；新版配套学生包尚未发行，发布后请按新版发行说明下载，不混用数据与工具。示例本人信息须替换；变量只在当前终端有效，学号按字符串保留前导零，自报信息不是认证。

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

若使用 Python Launcher，将 `python` 换为 `py -3`。v0.3.0 原生 Windows 仍仅支持本地检查与打包。新版候选已在 windows-2022（Python 3.10.11、MinGW-w64 g++ 14.2.0）完成原生客户端存储边界、合成提交链与可信占位模板 `.exe` 编译测试；网络为 mock，不是实际服务器提交，也不是算法质量通过。v0.3.1 尚未发行，请以新版发行说明使用原生 `submit` / `retry`。这不代表 MSVC `cl`、任意编译器版本或正式提交已经验证或开放。

</details>

打包生成 `submission.zip` 后，按课程公布的入口提交；重复打包换输出文件名。逐提交如实选择 `ai_assisted` 或 `independent`，`AI_USAGE.md` 不代替该标签。本人信息、每日次数、pending 限制、超时重试与幂等要求见学生指南；不要向未公告地址发送身份或源码。

## 排行榜

访问地址以课程公告为准，当前不提供猜测链接。学生包内 `public/leaderboard.html` 可先阅读；file 离线预览不联网、不生成成绩。未公布成绩时显示“成绩暂未公布。”，正式发布后可用标题右侧“刷新”读取榜单。

公开诊断、本机耗时、文档发布或打包成功均不等于正式评测通过。平台故障不冒充学生 0 分或默认满分；正式结果、评分公布与重评安排以课程公告为准。
