# ML-Assignment-1 学生使用指南

实验材料已开放。先[下载完整材料](https://github.com/ZeroGen-Lab/ML-Assignment-1/archive/refs/heads/main.zip)，完成算法实现、本地自测和源码打包。在线提交、成绩查询和排行榜暂未开放；本地自测不产生课程成绩。

## 1. 需要完成什么

用C++标准库从零实现Logistic Regression、Linear SVM、GBDT Regression、K-Means和Discrete HMM。保持`assignment`命名空间、类名及公开方法签名；可以修改private成员，在`starter/include/`和`starter/src/`增加辅助源码。模板只提供接口与占位输出，五种算法尚未实现；需要自行实现训练与预测，不能把占位输出当作参考算法。无需实现上传程序或正式评测器，不提交模型、二进制、第三方机器学习库或数据。

| 任务 | 方法 | 输出约定 |
|---|---|---|
| Logistic Regression | fit / predict | 二分类标签0/1，长度及顺序与输入一致 |
| Linear SVM | fit / predict | 二分类标签0/1，长度及顺序与输入一致 |
| GBDT Regression | fit / predict | 有限实数预测，不能含NaN/Infinity |
| K-Means | fit / predict | 先训练同一对象，再预测[0,K)簇编号；本课程Rice任务K=2；不是在predict中重新训练 |
| Discrete HMM | fit / predict | 使用观测及状态标签监督训练；预测Viterbi联合最优路径 |

HMM精确平局取最小前驱、最小末状态；空观测和无可行路径返回该序列的空路径。空预测矩阵返回空结果；K>N和空簇须正常结束并返回合法编号。正常训练输入为非空、有限、等维度数据。不要硬编码公开样例答案，隐藏测试使用不同数据。

## 2. 本机环境与公开自测

### 下载与目录

下载课程公告提供的完整学生包，或下载学生仓库源码 ZIP 后解压。保留 `starter/`、`student_tools/`、`docs/`、`public/` 与 `data/public/` 的目录关系，不要只下载单个源文件。先阅读[模板说明](../starter/README.md)，按本人实际情况填写[AI 使用说明模板](../starter/AI_USAGE.md)；模板不是学生本人的声明。课程说明见[学生页面](../public/index.html)。

本节适用 `course-uci-20261014-v3`，数据身份 `0f23098f6a76cf60424ca04bc1bfe1d7e96ad1ef1a59a632dd2b319ab162c671`。当前配套学生包为 v0.3.1；使用本仓库提供的同版模板、数据和工具。五份公开训练CSV位于 `data/public/training/`：LR/SVM共享同一Spambase来源、正式train/test划分、标准化及公开200条，文件分别对应两个算法；GBDT、Rice各公开200条。公开200是正式train代表性子集，不替代服务器正式训练。HMM公开10条完整20步序列，共200行；正式60/30条完整序列不变。

| 任务 | 数据来源与训练文件 | 正式训练 / 留出测试 | 输入约定 |
|---|---|---|---|
| Logistic Regression | Spambase，`lr-train.csv` | 1194 / 606 条 | 57 个特征；列 `f0` 至 `f56`，`y` 为0/1标签 |
| Linear SVM | Spambase，`svm-train.csv` | 1194 / 606 条 | 57 个特征；列 `f0` 至 `f56`，`y` 为0/1标签 |
| GBDT Regression | CCPP，`gbdt-train.csv` | 1199 / 601 条 | 4 个特征；列 `f0` 至 `f3`，`y` 为实数目标 |
| K-Means | Rice Cammeo/Osmancik，`kmeans-train.csv` | 1199 / 601 条 | 7 个特征；列 `f0` 至 `f6`，无类别标签列；K=2 |
| Discrete HMM | 监督序列，`hmm-train.csv` | 60 / 30 条完整序列 | 每序列20步，3个状态、4个观测符号 |

课程只发布上述正式训练代表子集，不发布完整正式train或留出test及其真值。前四项服务器实际正式训练/测试为LR/SVM1194/606条、GBDT和Rice各1199/601条，不以学生公开200替代。前四项数据来自 UCI，来源、许可与引用见包内 `DATA-SOURCES.md`；公开原始数据的标签可能被查回，不应将课程留出部分视为不可获得的秘密，也不要据此硬编码答案。

前四项 CSV 的特征已使用**正式训练集的统计量进行 z-score 标准化**，与服务器训练输入一致；**无需再次标准化**。分类标签及回归目标不做特征标准化，HMM的离散编号也不做标准化。请自行从公开200条（HMM10整条）划分本地训练/验证集，用于调参与检查泛化；确定方案后，可用全部公开训练子集重新训练。公开子集不单独计算标准化统计量。不要把留出测试或验证标签当作预测输入。

HMM CSV 的列为 `sequence_id,step,observation,state_label`，公开序列本地编号0..9，仅用于分组，不是特征或教师内部编号。按 `sequence_id` 分组、按 `step` 排序，还原完整观测序列及对应状态标签；训练/验证划分以整条序列为单位，不能拆散时间步。K-Means训练 CSV 不发布真实类别，不能把样本顺序当簇标签，也不能从这些公开数据计算依赖真实类别的ARI/NMI。

公开接口以[assignment.hpp](../starter/include/assignment.hpp)为准：同一对象先 `fit` 再 `predict`，预测只接收特征或观测，不接收标签。输入宽度分别为LR57、SVM57、GBDT4、K-Means7；不要硬编码样本数量。HMM使用 `fit(observations, state_labels, n_states, n_symbols, seed)` 监督训练；观测属于[0,4)，状态属于[0,3)，两种编号空间不同。训练数据不直接提供转移、发射或初始分布，由实现内部建模；`predict(observations)`输出各序列对应长度的状态路径，状态标签含义跨序列一致。

需要Python 3.10+及支持C++17的编译器。官方评测使用**GCC 12、C++17、`-std=c++17 -O2`**；本机可以用兼容编译器。

- Windows：MinGW-w64 GCC或WSL GCC；检查`python --version`和`g++ --version`。不能将MSVC cl直接当作GCC命令行编译器。
- macOS：Apple Command Line Tools / clang++；检查`python3 --version`、`clang++ --version`。g++可能是Apple clang。
- Linux：检查`python3 --version`、`g++ --version`。

工具不自动安装编译器。命令不存在时按平台官方说明安装，不运行来源不明的脚本。在包含`starter/`和`student_tools/`的目录操作。

公开诊断用于快速检查接口与基本效果，不等于正式训练/测试。新版诊断从对应公开训练子集中划分，LR、SVM、GBDT、KMeans各160条fit/40条predict，HMM为8条完整fit序列/2条完整predict序列；具体内容与身份见包内同版manifest；HMM按整条序列划分，诊断predict部分不是课程留出测试。勿将公开200或诊断规模写死到算法中。

新版公开诊断文件位于学生包的`data/public/`，配套五份训练CSV；不把旧版工具、旧版JSON或旧维度混用。在学生包根目录执行：

```sh
python3 -B -m student_tools.selftest --project starter --public-data data/public --tasks lr,svm,gbdt,kmeans,hmm
# 指定已有兼容编译器：
python3 -B -m student_tools.selftest --project starter --compiler clang++ --public-data data/public --tasks lr,svm,kmeans,hmm
```

Windows命令名为python时将python3换为python。`--tasks`须选择四项或五项已实现任务，GBDT也可参与诊断。公开数据模式显示`SELFTEST PASSED`或`5/5 passed`只表示编译、数据读取和输出格式检查成功；工具报告诊断指标，不判定算法质量达到合格线。请结合指标和自划验证结果判断算法效果。占位模板尚未实现算法，输出不代表算法质量；成功编译不等于算法正确，公开自测不产生正式成绩。

**安全提示：本机自测会编译并执行C++代码，不是安全沙箱。只运行自己编写或已审查的代码，不替陌生人运行提交包。**

## 3. 提交包规范

ZIP根目录必须包含：

- `include/assignment.hpp`和`src/assignment.cpp`。
- `README.md`：简述项目结构、各任务实现文件及必要编译注意事项。
- `AI_USAGE.md`：简述AI工具、用途及本人检查/修改/验证；未使用AI写`No AI assistance used.`。

其余仅允许include/内的.h/.hpp及src/内的.cpp；不额外嵌套项目根目录。**五份训练CSV、公开诊断文件和来源说明不放进源码ZIP**；评测时由服务器提供训练与预测输入。两份Markdown必须为非空UTF-8文本。文件最多64个（含Markdown），每文件最多1MiB，解压总量和ZIP各最多8MiB，每成员压缩比最多200。仅接受未加密stored/deflate ZIP；禁止空文件、NUL、非UTF-8、符号链接、路径逃逸、重复或规范化冲突路径、凭证、数据、答案、模型、二进制和其他未列明文件。

请使用课程公告提供的工具版本，按该版本说明打包；打包命令：

```sh
python3 -B -m student_tools package --project starter --output submission.zip
python3 -B -m student_tools package --help
```

已有输出不覆盖，重复打包使用新文件名。检查包内四个必需文件齐全，不因本机工具命令返回成功就假定包满足所有提交规范。

## 4. 学号信息与AI声明

请填写本人学号、姓名和昵称，提交及结果按学号归属。学号按字符串处理，保留前导零；无需学生SSO、密码、个人token或一次本人绑定。学号自报不是本人认证，教师管理权限独立。

macOS / Linux：

```sh
export STUDENT_ID=2026123456        # 虚构示例，换为本人信息
export STUDENT_NAME="Zhang San"
export STUDENT_NICKNAME="zs"
python3 -B -m student_tools config
```

Windows PowerShell环境变量语法：

```powershell
$env:STUDENT_ID = "2026123456"
$env:STUDENT_NAME = "Zhang San"
$env:STUDENT_NICKNAME = "zs"
python -B -m student_tools config
```

变量只在当前终端有效。config只做本机格式检查、不上传、不回显原值；通过不代表已登录、本人认证或可提交。

鼓励AI辅助；AI辅助和自主完成均可提交，须理解、验证并简述最终实现。逐提交声明`ai_assisted`或`independent`，不直接加扣分，不能事后改写旧声明；AI标签是学生声明，不是自动检测结论。提交包同时包含AI_USAGE.md。

在线提交暂未开放。开放后使用提供的 HTTPS 入口；不要向自行猜测的地址发送身份或源码。结果以服务端收据为准，客户端不上传自报分数。遇到超时或结果未知，保留原意图/幂等标识查询或重试，不盲目创建新意图。幂等标识不是认证凭据。

本地意图保存支持 macOS、Linux 和 Windows：macOS/Linux 使用 POSIX 文件后端，Windows 使用 SQLite 后端。准备意图时必须用 `--tasks` 选择四个或五个不同任务，AI 声明须与实际使用一致；Windows PowerShell 将下面的 `python3` 换为 `python`。

```sh
python3 -B -m student_tools prepare --archive submission.zip --ai independent --tasks lr,svm,gbdt,kmeans,hmm
python3 -B -m student_tools intent <idempotency_key>
```

## 5. 评分规则

五项任务各25分，可任选四项完成；完成五项时取最高四项合计100分，占课程总评35%。课程总评60分规定不变。

效果为主，正确性、鲁棒性与效率为辅；综合表现前10%为满分档，随机/无信息基线为基础档。这是概要，不承诺单一指标或公开诊断通过就取得任务25分。分类关注准确率与F1，回归关注误差，聚类关注划分质量，HMM关注状态和整条路径；同时满足接口、合法边界、数值稳定及效率要求。

每任务选取完整有效提交，不拼不同提交的子分；AI本身不直接加扣分，须如实声明。正式计算、同分与截止政策见[课程题面](https://dasellmg.yuque.com/org-wiki-dasellmg-vsl5wg/ml2026/qckpnr5fvr269d83)及课程公告。[评分与提交概要](assignment-scoring-rules-20261010.md)不修改评测程序，不新增精确权重。

可打开包内[成绩榜单](../public/leaderboard.html)；离线打开不请求网络、不生成排名，成绩未公布时显示“成绩暂未公布。”。在线排行榜暂未开放。本机公开诊断不是课程成绩；基础设施故障或缺评测证据不记学生0分，暂缓后按同规则重评。

截止以公告带时区时刻及服务端完成收包/校验后的接收时刻为准。每学号每天（Asia/Shanghai）最多10个新提交意图、最多1个未完成评测任务；原意图重试不重复计次。

提交限额拒绝的HTTP状态为 **429**；响应正文的 `error.code` 区分原因，客户端按该错误码处理：

- `DAILY_SUBMISSION_LIMIT`：当天已接收至少10个新提交意图。等待 **Asia/Shanghai次日零点** 后再尝试；任务完成或失败不会退还当日名额。等待不延长截止时间，新提交的服务端接收时刻仍须不晚于截止时刻。
- `PENDING_SUBMISSION_LIMIT`：已有 `QUEUED` 或 `RUNNING` 的评测任务。先查询已有提交，等待它进入 `COMPLETED`、`STUDENT_FAILURE` 或 `INFRASTRUCTURE_FAILURE` 后再尝试；未完成任务跨日仍计入限制，不因零点自动释放。

这两种429拒绝不会创建新的服务端提交记录或消耗新的当日次数，也不会排队等待；当前响应没有 `Retry-After`，客户端不自动等待或重试，不要持续发送新请求。两种限制同时命中时先返回每日限额码。历史当天已接受超过10次的记录不删除或改写，但不能继续新增；已接受原意图的合法幂等回放仍保留。

请求超时不等于提交失败。重试原意图须保留原幂等键（`idempotency_key`）、完全相同的ZIP字节和AI声明；已接收的原意图会返回原回执，不新增次数。不要为重试换新键，也不要用同一键提交修改后的内容。若先前明确收到限额429，服务端尚未接收该意图；等待限制解除后沿用原请求再次尝试，仍须通过当时的限额与截止检查，不是已接收提交的回放。

## 6. 常见误解与反馈

- 编译通过不等于算法正确，公开样例通过不等于隐藏测试满分。
- 修改昵称不改变学号或成绩归属。
- 本机打包时间不是服务端接收时间；请求超时不表示一定失败。
- 公开自测不产生正式课程分数；截止前暂估榜单不是最终结算。

反馈附操作系统、Python/编译器版本、命令和简短错误。不公开凭证、真实姓名/完整学号或未公开数据。在线入口开放后将在仓库首页补充。
