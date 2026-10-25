# ML Assignment 1 · C++17 机器学习算法实践

[排行榜与网页提交](http://120.55.190.84:18080/leaderboard) · [课程仓库](https://github.com/ZeroGen-Lab/ML-Assignment-1)

本次作业要求使用 **C++17 和 C++ 标准库**，从零实现机器学习算法，并在课程服务器上进行统一评测。你需要完成数据读取、模型训练、预测和实现说明，理解并能够解释自己提交的代码。

提交流程：获取模板并实现算法后，使用仓库工具生成源码 ZIP，再用命令行工具（推荐）或评测网页提交，并主动查询结果。

课程服务器统一按 C++17 标准编译，代码须遵守标准库和课程接口要求。请勿依赖特定操作系统、编译器扩展或本地环境配置。

五项任务中**任选四项或五项**：Logistic Regression、SVM、GBDT、K-Means、Discrete HMM。每项得分最高25分；排行榜展示所选任务得分合计，选五项最高125分。课程计分取同一次提交中最高四项，满分100分，占课程总评的35%。

**提交截止：2026年11月12日23:59（北京时间）。**

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

每项任务提供200条训练数据，用于理解数据、开发实现和自行划分验证集。文件位于 `data/training/`：

| 任务 | 文件 | 字段及含义 |
| --- | --- | --- |
| LR | `lr/train.csv` | `f0` 至 `f56` 为57维特征，末列 `y` 为0或1标签 |
| SVM | `svm/train.csv` | 与 LR 使用相同的特征和标签记录 |
| GBDT | `gbdt/train.csv` | 首列 `label` 为0至8类别，后续 `x000` 至 `x855` 为856维特征 |
| K-Means | `kmeans/train.csv` | `f0` 至 `f6` 为7维特征，不提供类别标签列 |
| HMM | `hmm/train.csv` | `sequence_id,step,observation,state_label` |

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

本地检查帮助发现编译错误、接口错误和不符合要求的输出。它会在你的电脑上执行你自己的代码；检查通过不等于服务器得分，也不能替代对算法效果的验证。

GBDT 的本地格式检查使用提供的200条训练记录作为查询输入，检查输出数量及0至8整数类别范围。常量输出也可能满足格式检查，你仍需要自行检查算法是否学习到了有效规律。

## 6. 填写实现说明与 AI 声明

提交前填写 `starter/README.md`，说明你完成了哪些任务、主要代码文件、实现思路及验证方式。

填写 `starter/AI_USAGE.md`，如实说明是否使用了 AI、使用的工具、用途，以及你如何检查和修改生成的内容。独立完成可以写明未使用 AI。无论是否使用 AI，你都需要理解并能解释自己的实现；网页或命令行提交时的 AI 声明应与文件内容一致。

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

## 8. 提交身份

提交与查分有两种方式：**命令行（推荐，见第9节）** 和 [网页](http://120.55.190.84:18080/leaderboard)。两种方式都使用学号、姓名和昵称三项身份信息，不需要预先登记，也不需要登录服务器。网页分为个人查询、源码上传和成绩榜三个区域。

不要在页面填写密码、密钥或其他无关敏感信息；只提交作业要求的材料。

命令行通过环境变量 `STUDENT_ID`、`STUDENT_NAME`、`STUDENT_NICKNAME` 提供这三项（见第9节）；网页在同页上方填写，上传和查询都使用这三项。学号按字符串填写，保留开头的0；请使用本人信息，不要填写示例姓名、他人身份或多余空格。排行榜不公开姓名。

请如实填写本人信息；冒用他人学号提交属于学术不端。网页不保存身份信息，重新打开页面后需重新填写。不要把自己的姓名等信息提交到课程公共仓库。

## 9. 提交、查分与重试

日常流程：**clone 获取模板 → 实现算法与说明 → 用第7节工具生成源码 ZIP → 用命令行（推荐）或网页提交 → 主动查询结果**。

### 命令行提交与查分（推荐）

在仓库根目录执行。`<学号>`、`<姓名>`、`<昵称>` 换成你自己的信息；`submission-v1.zip` 是第7节生成的提交包。

macOS / Linux：

```bash
# 1. 配置身份（每次打开新终端都要设置，也可写入 shell 配置文件）
export STUDENT_ID=<学号>
export STUDENT_NAME=<姓名>
export STUDENT_NICKNAME=<昵称>

# 2. 确认身份配置生效
python -B -m student_tools config

# 3. 提交（保存输出里的 idempotency_key 和 submission_id）
python -B -m student_tools submit --archive submission-v1.zip \
    --tasks lr,svm,gbdt,kmeans --ai independent \
    --endpoint http://120.55.190.84:18080

# 4. 查询单笔状态与得分（用第3步输出的 submission_id）
python -B -m student_tools status <submission_id> \
    --endpoint http://120.55.190.84:18080

# 5. 列出自己的提交记录
python -B -m student_tools submissions \
    --endpoint http://120.55.190.84:18080
```

Windows PowerShell：

```powershell
$env:STUDENT_ID = "<学号>"
$env:STUDENT_NAME = "<姓名>"
$env:STUDENT_NICKNAME = "<昵称>"

python -B -m student_tools config

python -B -m student_tools submit --archive submission-v1.zip `
    --tasks lr,svm,gbdt,kmeans --ai independent `
    --endpoint http://120.55.190.84:18080

python -B -m student_tools status <submission_id> `
    --endpoint http://120.55.190.84:18080

python -B -m student_tools submissions `
    --endpoint http://120.55.190.84:18080
```

- `--tasks` 用逗号分隔，选四项或五项：`lr,svm,gbdt,kmeans,hmm`。
- `--ai` 填 `independent`（独立完成）或 `ai_assisted`（使用AI辅助），须与 `AI_USAGE.md` 一致。
- `submissions` 输出完整 JSON 提交记录列表；`status` 先输出完整 JSON，之后附 `summary:` 摘要行：状态、各任务结果、五项合计与最高四项得分（四位小数）。尚未测得的值显示 `absent`，不补0。
- 提交后不会自动轮询；等待一段时间后再次执行 `status` 查询。

**网络异常时的重试**：先用 `submissions` 查最近记录，确认原请求是否已受理。需要重试时，先用 `prepare` 固定本次请求，再用它输出的 key 重试；同一 key 重试不重复计次：

```bash
python -B -m student_tools prepare --archive submission-v1.zip \
    --tasks lr,svm,gbdt,kmeans --ai independent
python -B -m student_tools retry <idempotency_key> \
    --endpoint http://120.55.190.84:18080
```

以下是网页提交方式，与命令行等效，二选一即可。

### 网页首次提交

1. 按第8节填写三项身份。在“提交源码 ZIP”区域的“源码压缩包（ZIP）”中选择第7节生成的 `submission-v1.zip`，不要选整个仓库 ZIP。ZIP 文件必须非空，且不超过 **8 MiB（8,388,608 bytes）**；包内每个文件解压后不超过 **1 MiB（1,048,576 bytes）**，解压总量不超过 **8 MiB**，其余规则见第7节。
2. 上传区的“提交编号（选填）”是请求标识 `idempotency_key`，也称提交key；首次可留空自动生成。它与受理后返回的32位受理编号 `submission_id` 不同。恢复未确认的原请求时填写原key。
3. 在“AI 使用声明”中选择“独立完成”或“使用 AI 辅助完成”，与 `AI_USAGE.md` 一致。
4. 在“评测项目（请选择 4 项或 5 项）”中确认本次任务。页面默认全选；只交四项时取消未选任务，最终必须是四个或五个不同任务。
5. 点击“提交源码”。本次有效尝试会固定身份、ZIP、key、AI 声明和任务选择；请保存页面显示的 key。页面不会替你持久保存身份或源码，先不要关闭页面或删掉原 ZIP。
6. 看到“提交已受理”后，保存 **受理编号 `submission_id`** 和 **请求标识 `idempotency_key`（提交key）**。受理不等于评测完成，也不代表已经出分。

### 同页查询 / 刷新

受理成功后，页面会回填本次身份与提交编号，但**不会自动查询或自动轮询**。请点击同页“查询 / 刷新”，查看排队、执行状态及已测结果；等待后再次主动点击刷新。

查询仍需填写原三项身份。查询区“受理编号”填写服务器返回的32位小写十六进制 `submission_id`，可查单笔；留空可查最近20条。不要误填上传区的提交key。重新打开页面后，使用原身份和受理编号查分，不必再次上传源码。

页面显示各任务得分；最终课程成绩由教师按第11节规则统一评定。已测得的指标会先显示，尚未测得的项目显示“未完成”，不补0。因服务器故障导致的评测失败不计为学生零分。

### 网络异常与原请求重试

网络中断、超时或没有明确结果时，**先用原身份查询最近提交**，避免同一份源码被重复受理。未确认原请求结果时，不要换 key 或发起新的提交。

本页已固定尝试后，使用“重试本次提交（沿用原身份、原文件与原提交编号）”；它保留本次 AI 声明和任务选择，不自动更换 key，也不自动重提。修改上方身份不会改变已固定尝试的身份。若重新打开页面恢复请求，须重新选择原 ZIP、填写原三项身份和原 key，并保持原 AI 声明与任务选择完全一致；不要用修改后的源码冒充原请求。

已核对最近记录、确认确实要提交新版本时，才使用“我已确认最近的提交记录，发起一次新的提交（将占用当日提交次数）”。修改算法后先生成新文件名的 ZIP，再按新提交处理；同学号上一单未结束时仍不能发起下一单。

### 状态、次数与受理规则

| 状态 | 含义 |
| --- | --- |
| `QUEUED` | 已受理，正在排队 |
| `RUNNING` | 正在评测 |
| `COMPLETED` | 本次评测结束，可以查看各任务得分 |
| `STUDENT_FAILURE` | 编译、运行或输出不符合要求，需要查看错误信息并修改代码 |
| `INFRASTRUCTURE_FAILURE` | 服务器故障，不计为学生零分；请联系助教并提供受理编号 |

每个学号同时只能有一笔排队中（`QUEUED`）或评测中（`RUNNING`）的提交；上一单到达可查询的完成、编译错误、超时或明确平台错误终态前，不能提交下一单。更换昵称、任务选择或请求标识不能绕过同学号限制。

按北京时间（`Asia/Shanghai`）自然日，每个学号每天最多成功受理 **5 次新提交**。同一请求标识（`idempotency_key`）重试不重复计数；非法提交包或因上一单未结束而被拒绝的请求不耗次数。已受理的提交即使编译失败或超时，也计为一次。网络异常或技术恢复请保留原提交编号，沿原请求处理，不要重新提交来替代恢复。

## 10. 服务器如何评测

服务器接收源码后，统一编译和运行，并按执行时限评测。评测程序调用你实现的 `fit` 和 `predict`，使用规定数据计算真实指标，再保存到本次提交记录。提交内容是代码，分数由服务器计算。

每次选择的各任务按规定训练、预测，不接受本机自报分数或本机耗时。程序需要在分配的计算资源和时间内结束；请控制内存使用，避免无界循环、无限输出及依赖网络下载。

每次提交从开始执行起，总墙钟预算为 **180 秒（3 分钟）**，包含编译、所选任务的训练、预测和评测；**排队时间不计入**。超过时限会记录超时结果，未测得的指标不补 0。3 分钟是执行上限，不是提交后 3 分钟内一定出分；多人提交时需要排队。

请合理使用 CPU、内存、进程数及输出；只使用作业输入与工作目录；不要访问服务数据、身份信息或隐藏答案，不要依赖外网下载。

本机自测结果仅供自查，成绩以服务器评测结果为准。

本次提交编号同时用于状态和结果查询。排队、处理中或尚未测得的项目显示未完成，不用0分代替；如果确实完成评测并得到0分，则按实际结果显示。

## 11. 得分、课程计分与排行榜

### 各项效果指标

| 任务 | 原始效果质量 Q |
| --- | --- |
| LR / SVM / GBDT 分类 | `0.5 × Accuracy + 0.5 × Macro-F1` |
| K-Means | `0.6 × ARI + 0.4 × NMI` |
| HMM | `0.5 × state_accuracy + 0.5 × exact_sequence_accuracy` |

Accuracy 是分类正确比例；Macro-F1 是对各类别 F1 的等权平均。GBDT 按固定九个类别评价，不能通过少预测一些类别缩小评价范围。

ARI 和 NMI 衡量聚类结果与类别划分的一致性，簇编号互换不直接算错。HMM 的 state accuracy 统计逐位置状态是否正确；exact sequence accuracy 只有在整条20步状态均正确时才计为该序列正确。

令 `q = min(1, max(0, Q))`，每项得分为：

```text
任务得分 = 25 × q
```

例如 Accuracy 为0.80、Macro-F1 为0.76，则 Q 为0.78，该项得分为19.5分。缺测的指标不参与虚构计算，也不自动记成0分。

### 任务得分合计与课程值

- 所选四项均完成时，任务得分合计最高100分。
- 所选五项均完成时，任务得分合计最高125分。
- 课程计分取同一次提交中最高四项，最高100分；不把多次提交中各自最好的单项拼到一起。
- 任一所选任务还未测得结果时，所选任务合计显示未完成。已有四项完成时，可以计算该次提交的课程值。
- 作业占课程总评的35%；例如课程值为80分，对总评的贡献为28分。最终课程成绩由教师统一评定。

### 排行榜展示

排行榜地址：[http://120.55.190.84:18080/leaderboard](http://120.55.190.84:18080/leaderboard)（打开页面后在下方“成绩榜”区域查看，点击“刷新”获取最新排名）。

榜单按任务得分合计从高到低排名；同分并列同一名次，较早提交的排在前面。最终课程成绩由教师统一评定。

排行榜显示名次、学号、昵称、五项任务得分和“任务得分合计”，不显示姓名。所有人都可以点击任务得分，查看相应的真实指标，例如 Accuracy、Macro-F1 或 HMM 两项准确率。

每个学号只展示一次提交：任务得分合计最高的一次；同分时取较早的提交。未选择的任务显示“未选”，未测得的任务显示“未完成”，不会伪填分数。原始指标与得分保存在各自的提交记录中。

## 12. 提交前自查

1. 已选择四项或五项，类名和公开方法签名与模板一致。
2. 输入多少条记录，就返回多少个预测；HMM 保留每条序列长度与顺序。
3. 没有把标签当作预测输入，没有硬编码数据答案。
4. `starter/README.md` 和 `starter/AI_USAGE.md` 已按实际实现填写。
5. 已使用工具生成源码 ZIP，未夹带数据、模型、程序或整个仓库。
6. 学号、姓名、昵称及 AI 声明填写正确。
7. 已保存受理编号 `submission_id` 和提交key `idempotency_key`；提交后主动查询结果，网络异常先查最近记录，再以原身份、原ZIP和原key重试。

遇到问题时，向课程助教提供提交编号、错误代码和简要复现步骤；不要公开上传个人资料、完整私有日志或账户凭据。
