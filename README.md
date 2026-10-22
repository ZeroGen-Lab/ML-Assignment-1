# ML Assignment 1 · C++17 机器学习算法实践

本次作业要求使用 **C++17 和 C++ 标准库**，从零实现机器学习算法，并在课程服务器上进行统一评测。你需要完成数据读取、模型训练、预测和实现说明，理解并能够解释自己提交的代码。

五项任务中**任选四项或五项**：Logistic Regression、SVM、GBDT、K-Means、Discrete HMM。每项参考分最高25分；排行榜展示所选任务得分合计，选五项最高125分。课程计分取同一次提交中最高四项，满分100分，占课程总评的35%。

## 1. 获取课程材料

在终端执行：

```bash
git clone https://github.com/ZeroGen-Lab/ML-Assignment-1.git
cd ML-Assignment-1
```

不使用 Git 时，可以打开[课程仓库](https://github.com/ZeroGen-Lab/ML-Assignment-1)，点击 **Code → Download ZIP**，解压后进入仓库目录。整个作业过程都在包含 `starter/` 和 `student_tools/` 的仓库根目录下操作。

```text
ML-Assignment-1/
├── README.md                    # 完整课程说明
├── starter/
│   ├── include/assignment.hpp   # 算法接口
│   ├── src/assignment.cpp       # 算法实现
│   ├── tools/driver.cpp         # 本地测试调用程序
│   ├── README.md                # 你需要填写的实现说明
│   └── AI_USAGE.md              # 你需要填写的 AI 使用声明
├── data/training/               # 五项任务的训练数据
└── student_tools/               # 检查、源码打包、提交与查分工具
```

你需要修改的是自己的算法源码和两份说明文件。课程材料通过仓库获取；提交时，由工具另行生成只包含你自己源码的 ZIP。

## 2. 五项任务要完成什么

| 任务 | 问题类型 | 输入特征或序列结构 | 预测输出 |
| --- | --- | --- | --- |
| Logistic Regression（LR） | 二分类 | 每条记录57维特征 | 每条记录一个标签：0或1 |
| SVM | 二分类 | 每条记录57维特征 | 每条记录一个标签：0或1 |
| GBDT | 九类分类 | 每条记录856维特征 | 每条记录一个整数标签：0至8 |
| K-Means | 聚类 | 每条记录7维特征，K=2 | 每条记录一个簇编号：0或1 |
| Discrete HMM | 监督序列建模与状态预测 | 3个状态、4种观测符号，每条序列20步 | 每条序列的逐位置状态编号 |

GBDT 的 **856维均为真实特征**，标签不计入特征维度。不要补零凑维度，也不要因为部分训练记录的某些特征为零就删除特征列。

你需要自行设计并实现每项算法。模板中的占位返回值用于展示调用接口，不是参考算法；即使占位程序能编译、输出格式正确，也不代表具有有效的预测能力。

### 实现入口与接口

接口声明位于 `starter/include/assignment.hpp`，实现位于 `starter/src/assignment.cpp`。保留 `assignment` 命名空间和公开方法签名；可以调整 private 成员，也可以在 `include/`、`src/` 下增加标准库辅助源码。

| 类 | 训练方法 | 预测方法 |
| --- | --- | --- |
| `LogisticRegression` | `fit(x, y, seed)` | `predict(x)` 返回整数标签向量 |
| `SVM` | `fit(x, y, seed)` | `predict(x)` 返回整数标签向量 |
| `GBDTClassifier` | `fit(x, y, seed)` | `predict(x)` 返回整数类别向量 |
| `KMeans` | `fit(x, k, seed)` | `predict(x)` 返回整数簇编号向量 |
| `DiscreteHMM` | `fit(observations, state_labels, n_states, n_symbols, seed)` | `predict(observations)` 返回状态序列集合 |

`x` 使用 `Matrix` 表示二维特征；分类标签使用 `std::vector<int>`。先在同一模型对象上调用 `fit`，再调用 `predict`。预测结果的数量和顺序必须与输入一致，不能在结果中额外输出说明文字。`seed` 是训练随机种子，相同输入与种子应得到可复现的结果。

仅使用 C++ 标准库，不得调用现成机器学习库，也不得把答案、训练完成的模型或预测结果写死在源码中。不要修改配套调用程序来替代算法实现。

## 3. 提供给你的训练数据

每项任务提供200条训练记录，用于理解数据、开发实现和自行划分验证集。文件位于 `data/training/`：

| 任务 | 文件 | 字段及含义 |
| --- | --- | --- |
| LR | `lr/train.csv` | `f0` 至 `f56` 为57维特征，末列 `y` 为0或1标签 |
| SVM | `svm/train.csv` | 与 LR 使用相同的特征和标签记录 |
| GBDT | `gbdt/train.csv` | 首列 `label` 为0至8类别，后续 `x000` 至 `x855` 为856维特征 |
| K-Means | `kmeans/train.csv` | `f0` 至 `f6` 为7维特征，不提供类别标签列 |
| HMM | `hmm/train.csv` | `sequence_id,step,observation,state_label` |

LR/SVM 使用 Spambase 数据，K-Means 使用 Rice Cammeo and Osmancik 数据，GBDT 使用 CNAE-9 数据。具体来源、署名及数据处理说明随训练数据提供。

LR/SVM 和 K-Means 的特征已完成课程规定的标准化，不需要再次标准化。GBDT 使用提供的856列词频特征，按文件中的原列顺序读取。分类标签不能作为输入特征。

你可以自行从提供的数据中划分训练集和验证集，检查模型效果。不要将样本条数写死在算法中；接口应根据实际输入长度工作。课程服务器使用规定的训练、测试数据调用你的实现，测试标签不作为预测输入，也不随提交包上传。

### HMM 的训练输入、预测目标与评分真值

HMM 的200行表示 **10条完整序列，每条20步**，不是200条独立序列。

1. 按 `sequence_id` 分组，再按 `step` 从0至19排序，得到完整观测序列。
2. `observation` 取值为0至3，组成 `observations`；同一行的 `state_label` 取值为0至2，组成对应的 `state_labels`。
3. `fit` 同时接收这两组序列。状态标签在不同序列中的含义保持一致。
4. `predict` 只接收观测序列，返回每个位置对应的状态编号；每条输出序列必须与输入等长。

评分真值是数据生成时实际采样得到的状态路径，不是要求预测结果复现某个解码程序的输出。训练文件提供的是带状态标签的序列，不直接提供需要照抄的转移矩阵、发射矩阵或初始状态分布。

划分 HMM 训练集和验证集时，必须按完整序列划分，不能把同一条序列的不同时间步分到两侧。K-Means 的簇编号允许置换，不要求簇0、簇1直接等同于某个品种标签。

## 4. 本机环境

需要 **Python 3.10或以上**。本地编译自测还需要支持 **C++17** 的 `g++` 或 `clang++`；工具不会自动安装编译器。服务器使用统一的 Linux 编译运行环境进行评测。

| 系统 | 建议检查 |
| --- | --- |
| Windows | 在 PowerShell 检查 `python --version`、`g++ --version`；可使用 MinGW-w64 GCC，或在 WSL 中使用 GCC |
| macOS | 在终端检查 `python3 --version`、`clang++ --version` |
| Linux | 在终端检查 `python3 --version`、`g++ --version` |

以下示例用 `python` 表示 Python 3。macOS/Linux 如果命令为 `python3`，将示例中的 `python` 替换为 `python3`。不要将 MSVC 的 `cl` 当作具有相同命令行参数的 GCC 使用。

## 5. 开发与本地检查

建议先完成一个任务的接口和数据处理，再逐步实现、检查其余任务。修改代码后，在仓库根目录执行：

```bash
python -B -m student_tools selftest --project starter --tasks lr,svm,gbdt,kmeans
```

完成五项时，可以使用：

```bash
python -B -m student_tools selftest --project starter --tasks lr,svm,gbdt,kmeans,hmm
```

需要指定编译器时，例如：

```bash
python -B -m student_tools selftest --project starter --tasks lr,svm,gbdt,kmeans --compiler clang++
```

本地检查帮助发现编译错误、接口错误和不符合要求的输出。它会在你的电脑上执行你自己的代码；检查通过不等于服务器参考分，也不能替代对算法效果的验证。

GBDT 的本地格式检查使用提供的200条训练记录作为查询输入，检查输出数量及0至8整数类别范围。常量输出也可能满足格式检查，你仍需要自行检查算法是否学习到了有效规律。

## 6. 填写实现说明与 AI 声明

提交前填写 `starter/README.md`，说明你完成了哪些任务、主要代码文件、实现思路及验证方式。

填写 `starter/AI_USAGE.md`，如实说明是否使用了 AI、使用的工具、用途，以及你如何检查和修改生成的内容。独立完成可以写明未使用 AI。无论是否使用 AI，你都需要理解并能解释自己的实现；提交命令中的声明应与文件内容一致。

## 7. 生成源码提交包

在仓库根目录执行：

```bash
python -B -m student_tools package --project starter --output submission-v1.zip
```

工具会检查文件结构并生成 `submission-v1.zip`。ZIP 根目录应直接包含：

```text
README.md
AI_USAGE.md
include/assignment.hpp
src/assignment.cpp
```

可以包含 `include/` 下的 `.h/.hpp` 和 `src/` 下的 `.cpp` 辅助源码。不要额外套一层 `starter/` 或项目目录，不要上传整个仓库。

- 不得包含训练数据、答案、模型、可执行文件、第三方库或凭据。
- 最多64个文件；单文件最多1 MiB，解压总量及 ZIP 文件最多8 MiB。
- 文件使用 UTF-8，不能为空；不使用符号链接或越出目录的路径。
- 工具不覆盖已有 ZIP。修改代码后，请使用新文件名，例如 `submission-v2.zip`。

生成提交包只检查提交材料，不会把你本机的预测结果当作服务器成绩。

## 8. 配置提交信息

填写本人学号、姓名和昵称。学号按字符串使用，保留开头的0；排行榜只显示学号和昵称，不公开姓名。

macOS/Linux：

```bash
export STUDENT_ID="2026123456"
export STUDENT_NAME="张三"
export STUDENT_NICKNAME="我的昵称"
python -B -m student_tools config
```

Windows PowerShell：

```powershell
$env:STUDENT_ID="2026123456"
$env:STUDENT_NAME="张三"
$env:STUDENT_NICKNAME="我的昵称"
python -B -m student_tools config
```

请替换示例信息。这些设置只在当前终端中生效；不要把自己的姓名等信息提交到课程公共仓库。

## 9. 上传、重试与查分

课程公告将提供评测地址的配置方式。下列命令中的 `EVAL_ENDPOINT` 使用公告给出的实际地址；如公告要求路径前缀，也应按公告配置。

### 首次上传

以选择 LR、SVM、GBDT、K-Means，且使用了 AI 为例，在 macOS/Linux 执行：

```bash
python -B -m student_tools submit --archive submission-v1.zip --ai ai_assisted --tasks lr,svm,gbdt,kmeans --endpoint "$EVAL_ENDPOINT"
```

PowerShell 中对应为：

```powershell
python -B -m student_tools submit --archive submission-v1.zip --ai ai_assisted --tasks lr,svm,gbdt,kmeans --endpoint "$env:EVAL_ENDPOINT"
```

独立完成使用 `--ai independent`；完成五项时，将任务列表改为 `lr,svm,gbdt,kmeans,hmm`。一次必须选择四个或五个不同任务。

成功受理后，请保存工具返回的 **提交编号 `submission_id`** 和 **请求标识 `idempotency_key`**。受理表示服务器收到了源码，并不表示已经评测完成。此后由服务器排队、编译和运行；学生不需要 SSH 登录服务器，也不需要自己安装服务器端评测程序。

### 网络异常时重试原请求

如果网络中断或没有收到明确结果，使用返回的原请求标识进行 `retry`。下面用 `原请求标识` 表示工具返回的实际值：

```bash
python -B -m student_tools retry 原请求标识 --endpoint "$EVAL_ENDPOINT"
```

PowerShell 将地址参数改为 `"$env:EVAL_ENDPOINT"`。同一请求的重试沿用原源码、AI 声明和任务选择，不应再次执行 `submit` 制造另一笔提交。请保留工具生成的本地 `.student-intents` 目录。

如果修改了算法源码，应重新生成源码 ZIP，并作为一次新提交上传。

### 查询状态和结果

使用原提交编号查询：

```bash
python -B -m student_tools status 提交编号 --endpoint "$EVAL_ENDPOINT"
```

查看自己的提交记录：

```bash
python -B -m student_tools submissions --endpoint "$EVAL_ENDPOINT"
```

| 状态 | 含义 |
| --- | --- |
| `QUEUED` | 已受理，正在排队 |
| `RUNNING` | 正在评测 |
| `COMPLETED` | 本次评测完成，可以查看结果 |
| `STUDENT_FAILURE` | 编译、运行或输出不符合要求，需要查看错误信息并修改代码 |
| `INFRASTRUCTURE_FAILURE` | 服务器环境发生问题，不直接作为学生零分处理 |

每个学号同时只能有一笔未完成提交；按北京时间自然日，每天最多成功受理10次新提交。原请求的重试不应作为一笔新的源码提交重复受理。

## 10. 服务器如何评测

服务器接收源码后，统一编译并在隔离环境中执行。评测程序调用你实现的 `fit` 和 `predict`，使用规定数据计算真实指标，再保存到本次提交记录。提交内容是代码，分数由服务器计算。

每次选择的各任务按规定训练、预测，不接受本机自报分数或本机耗时。程序需要在分配的计算资源和时间内结束；请控制内存使用，避免无界循环、无限输出及依赖网络下载。

| 评测预算 | 限制 |
| --- | --- |
| CPU | 1核 |
| 内存 | 1 GiB |
| 编译时间 | 60秒 |
| 单项训练与预测 | 30秒 |
| 所选任务内部总时间 | 300秒 |
| 输出大小 | 2 MiB |

本机自测使用本机资源；正式比较以服务器实际评测结果为准。

本次提交编号同时用于状态和结果查询。排队、处理中或尚未测得的项目显示未完成，不用0分代替；如果确实完成评测并得到0分，则按实际结果显示。

## 11. 参考分、课程计分与排行榜

### 各项效果指标

| 任务 | 原始效果质量 Q |
| --- | --- |
| LR / SVM / GBDT 分类 | `0.5 × Accuracy + 0.5 × Macro-F1` |
| K-Means | `0.6 × ARI + 0.4 × NMI` |
| HMM | `0.5 × state_accuracy + 0.5 × exact_sequence_accuracy` |

Accuracy 是分类正确比例；Macro-F1 是对各类别 F1 的等权平均。GBDT 按固定九个类别评价，不能通过少预测一些类别缩小评价范围。

ARI 和 NMI 衡量聚类结果与类别划分的一致性，簇编号互换不直接算错。HMM 的 state accuracy 统计逐位置状态是否正确；exact sequence accuracy 只有在整条20步状态均正确时才计为该序列正确。

令 `q = min(1, max(0, Q))`，每项参考分为：

```text
任务参考分 = 25 × q
```

例如 Accuracy 为0.80、Macro-F1 为0.76，则 Q 为0.78，该项参考分为19.5分。缺测的指标不参与虚构计算，也不自动记成0分。

### 任务得分合计与课程值

- 所选四项均完成时，任务得分合计最高100分。
- 所选五项均完成时，任务得分合计最高125分。
- 课程计分取同一次提交中最高四项，最高100分；不把多次提交中各自最好的单项拼到一起。
- 任一所选任务还未测得结果时，所选任务合计显示未完成。已有四项完成时，可以计算该次提交的课程值。
- 作业占课程总评的35%；例如课程值为80分，对总评的贡献为28分。最终课程成绩由教师统一评定。

### 排行榜展示

排行榜显示名次、学号、昵称、五项任务得分和“任务得分合计”，不显示姓名。所有人都可以点击任务得分，查看相应的真实指标，例如 Accuracy、Macro-F1 或 HMM 两项准确率。

同一学号按同一评测规则选择一个完整提交展示，采用任务得分合计最高的一次；同分时保留较早提交。未选择的任务显示“未选”，未测得的任务显示“未完成”，不会伪填分数。原始指标与参考分保存在各自的提交记录中。

## 12. 提交前自查

1. 已选择四项或五项，类名和公开方法签名与模板一致。
2. 输入多少条记录，就返回多少个预测；HMM 保留每条序列长度与顺序。
3. 没有把标签当作预测输入，没有硬编码数据答案。
4. `starter/README.md` 和 `starter/AI_USAGE.md` 已按实际实现填写。
5. 已使用工具生成源码 ZIP，未夹带数据、模型、程序或整个仓库。
6. 学号、姓名、昵称及 AI 声明填写正确。
7. 已保存提交编号和请求标识，网络异常使用原请求重试。

遇到问题时，向课程助教提供提交编号、错误代码和简要复现步骤；不要公开上传个人资料、完整私有日志或账户凭据。
