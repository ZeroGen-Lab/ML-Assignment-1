# ML Assignment 1 · C++17 算法实践

[下载 v0.3.0 学生包](https://github.com/ZeroGen-Lab/ML-Assignment-1/releases/tag/v0.3.0) · [学生指南](docs/student-guide.md) · [模板说明](starter/README.md) · [评分规则](docs/assignment-scoring-rules-20261010.md)

现在可以下载学生包开始实现算法。**正式在线提交服务尚未开放。**

## 作业目标

用 **C++ 标准库**从零实现五项机器学习算法。每项 **25 分**，任选四项；完成五项时取最高四项，总分 **100 分**，占课程总评 **35%**。

保持 `assignment` 命名空间、类名与签名。模板未实现算法；不提交第三方机器学习库、数据、模型或二进制。

## 任务与训练数据

|任务|需要实现什么|数据来源|特征 / 结构|正式训练 / 测试|
|---|---|---|---|---|
|Logistic Regression|判断细胞样本良性 / 恶性|WDBC|30 维，0/1 分类|454 / 115 条|
|Linear SVM|判断邮件是否为垃圾邮件|Spambase|57 维，0/1 分类|1194 / 606 条|
|GBDT Regression|预测发电功率|CCPP|4 维，实数回归|1199 / 601 条|
|K-Means|训练聚类中心，再给新样本分簇|Seeds|7 维，K=3，无公开类别标签|168 / 42 条|
|Discrete HMM|从带标签的序列学习，再预测状态路径|课程合成监督序列|3 状态、4 符号，每条 20 步|60 / 30 条完整序列|

数据版本：`course-uci-20261014-v1`。只发布训练部分；前四项特征已标准化，**无需再次标准化**。自划验证集时 HMM 按完整序列划分。原始标签可能被查回，不应硬编码答案。来源许可见 [DATA-SOURCES.md](data/public/training/DATA-SOURCES.md)。

### 训练、验证与正式测试

每项任务只有一份正式训练集和一份教师留出测试集。学生包发布完整训练部分：分类 / 回归包含 `y`，HMM 包含每一步的 `state_label`，K-Means 仅发布特征。五份文件均为 CSV，见 [数据列说明](docs/dataset-columns.md)。

开发时从公开训练数据里留一部分作为本地验证，用于检查效果；方案确定后用完整训练集训练。公开快速诊断使用训练数据中的小子集（前四项 120 条训练 / 30 条验证，HMM 8 / 2 条完整序列），不会另增正式测试集。

服务器先调用同一对象的 `fit`，再调用 `predict`。预测只接收测试特征或观测序列，真实答案由评分端保留。算法不能依赖固定样本数量或数据行顺序；HMM 状态编号为 0–2，观测符号编号为 0–3，两套编号不同，真实生成矩阵不直接提供。

前四项从同一来源数据池划分，并将完全重复及近似重复样本整组放在同一侧；分类尽量保持类别比例，回归兼顾目标值范围。标准化参数只从训练集计算。HMM 的训练和测试使用相同生成规则，按整条序列划分。

## 项目结构

```text
starter/               # 算法实现与提交说明
  include/assignment.hpp
  src/assignment.cpp
  README.md / AI_USAGE.md
student_tools/         # 本地工具
data/public/training/  # 五份训练 CSV 与来源说明
data/public/           # 公开诊断输入与 manifest.json
docs/                  # 学生指南与评分规则
public/                # 静态榜单页面
```

## 开始使用

下载 Release 中的 `student-release-0.3.0.zip`，解压后在包根执行命令。需要 **Python 3.10+** 和兼容 **C++17** 的 `g++` 或 `clang++`；工具不会自动安装编译器。

- **macOS**：可用 Apple clang++。
- **Linux**：可用 g++。
- **Windows**：使用原生 Python 3.10+、PowerShell 与 MinGW-w64 g++ / 兼容 clang++；MSVC `cl` 不兼容当前自测命令。WSL2 可选，不是必需。

同一 Python 客户端打包源码 ZIP，服务器 Linux 统一评分，无需三份原生评测二进制。

<details>
<summary><strong>macOS / Linux · Bash 命令</strong></summary>

示例信息请替换为本人信息：

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

若使用 Python Launcher，将 `python` 换为 `py -3`。v0.3.0 提供上述本地检查与打包入口，原生 Windows 的实际兼容性验证及 `submit` / `retry` 更新仍在进行；完整提交支持以新版发布记录为准。

</details>

公开自测是格式与诊断检查，`SELFTEST PASSED` 不表示算法质量达标或取得课程分数。**本机自测不是沙箱，只运行自己编写或已审查的代码。**

## 提交源码

1. 在 `starter/` 实现任务，并填写其中的 `README.md`、`AI_USAGE.md`。
2. 打包生成 `submission.zip`，根目录须含 `include/assignment.hpp`、`src/assignment.cpp`、`README.md`、`AI_USAGE.md`；训练数据和工具不要装入。
3. 按课程公布的入口提交，并逐提交如实选择 `ai_assisted` 或 `independent`。完整格式、大小限制、本人信息、次数与重试要求见 [学生指南](docs/student-guide.md)。

变量仅当前终端有效；自报信息不是认证。重复打包换文件名。截止为正式课程发布后 3 周，以公告锁定的 Asia/Shanghai 时刻为准；不要向猜测地址发送信息或源码。

## 评分与排行榜

效果为主，正确性、鲁棒性与效率为辅；公开诊断不是正式测试集或成绩。详细计分与同分规则见 [评分规则](docs/assignment-scoring-rules-20261010.md)。

排行榜地址以课程公告为准；未公布成绩时显示“成绩暂未公布。”，发布后用标题右侧“刷新”读取榜单。
