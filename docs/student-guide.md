# 学生指南：当前课程合同与同步状态

更新日期：2026-10-20。本文与仓库首页说明同版；文档更新不代表新版代码、数据、在线系统已经交付。

> **新版模板、数据、工具同步中。** 当前公开 main 的模板仍可能采用旧 GBDT 回归接口；新版要求九类 `GBDTClassifier`。不要混用旧回归模板、旧数据和新评分合同，不要把本页当作当前仓库已能执行新版示例的证明。正式 HTTPS URL 尚未确认，在线提交、查分、自动榜单尚未开放；Windows 实机 **NOT_RUN**。

## 1. 获取源码与确认版本

打开 [ZeroGen-Lab/ML-Assignment-1](https://github.com/ZeroGen-Lab/ML-Assignment-1)，使用 **Code → Download ZIP**，或在自己的机器上执行：

```sh
git clone https://github.com/ZeroGen-Lab/ML-Assignment-1.git
cd ML-Assignment-1
```

GitHub Download ZIP 下载的是仓库快照，不是教师自制发行 ZIP。新版不再要求教师另外制作发行 ZIP；旧 `v0.3.1` 包仅留作历史。学生上传自己的源码仍有 `package` 步骤，不因此取消。

先核对本次课程公告和仓库实际文件。拟采用公开数据布局为 `data/training/{task}/train.csv`，任务名为 `lr`、`svm`、`gbdt`、`kmeans`、`hmm`；合并 manifest 与来源说明拟位于 `data/training/manifest.json`、`data/training/DATA-SOURCES.md`。GBDT 单项元数据拟保留 `data/training/gbdt/manifest.json`。尚未进入 main 的文件不能仅凭这里列出路径就视为存在。旧 `data/public/training` 不作为新版推荐路径。

只使用成组同步后的接口、数据、工具及说明；不自行将旧总 manifest 改名成新版，旧总 manifest 含回归 GBDT。

## 2. 实现任务

从 LR、SVM、GBDT、KMeans、HMM 中选择四项或五项。使用 C++17 与 C++ 标准库，不使用现成机器学习库；保留 `Matrix` 接口，不要求改为纯 C 或连续存储。声明、实现入口为 `starter/include/assignment.hpp`、`starter/src/assignment.cpp`，以同版实际文件为准。

- **LR**：二分类，公开训练 200 行、57 维特征和 `0/1` 标签。
- **SVM**：题面名称为 SVM，保留 `LinearSVM` C++ 符号；公开训练与 LR 共享原记录，不据此要求核函数。
- **GBDT**：九类分类，真实 **856** 维词频特征，整数标签 **0..8**。`GBDTClassifier::fit` 接收 `Matrix`、`vector<int>` 标签及 `std::uint64_t seed`；`predict` 返回每输入行一个整数标签。旧 `GBDTRegressor`、浮点回归预测及 CCPP 数据不适用。公开训练 200 行，表头 `label,x000,...,x855`，标签在首列；保留全部 856 特征列，不补维或降到仅非零列。
- **KMeans**：公开训练 200 行、7 维，无类别标签列；`k=2` 沿公开合同，不从文件伪造评估 truth。
- **HMM**：公开训练 200 行，表示 **10 条完整序列 × 20 步**。字段为 `sequence_id,step,observation,state_label`，观测 `0..3`，监督状态 `0..2`；step 为 `0..19`。监督 fit 同时接收真实 observations 与 state_labels，状态含义跨序列一致；不任意拼接、拆散或重新生成序列。

数据细节见[数据列说明](dataset-columns.md)。Spambase、Rice、CNAE-9 来源与许可按配套来源文档记录；课程生成 HMM 的独立许可证据未明确，状态保持 PARTIAL，不自动套用 CC BY。

## 3. 本地检查：不是正式分

本地工具只检查接口、输出格式及有限 sanity 条件，不证明服务器正式效果。未执行指标为 `NOT_RUN`，缺值为 `None/null`，不补 0。

配套新版公开 selftest 的 GBDT 查询仅为 200 条公开训练自身：检查输出数量 200、规范整数和 `0..8` 标签域。**常量合法输出也能通过，不能证明学习有效、泛化能力或参考分。** 其余四任务公开 micro-fixture 也不是正式评估集，不冒充课程提供的 200 条训练记录。

本地 selftest 会在自己的机器上编译、执行源码，**不是沙箱**。只执行自己理解并愿意本地运行的代码。当前模板、工具和数据未同步到同一版本前，不运行混合版本命令。新版不沿用旧 `--public-data` 回归验证用法；不能靠改版本名、复制教师数据或放宽身份校验绕过。

本页不提供假定新版工具已经进 main 的可执行自测命令。实际工具与依赖成组同步后，以工具帮助和同版说明为准。Windows 尚未实机验证。

## 4. 学生源码打包、提交与重试

当前配套工具仍要求 `STUDENT_ID`、`STUDENT_NAME`、`STUDENT_NICKNAME`：分别填写学号、姓名、昵称。**字段不是身份认证**，不能用改学号或昵称冒充他人；榜单不公开姓名不等于工具已经去掉 STUDENT_NAME。不要把个人身份资料写入公开仓库、示例或公开日志。

流程保持：

1. **package**：将自己的实现源码按同版学生工具规则打包。不要上传整个课程仓库或教师材料。这是学生提交源码包，不是教师发行包。
2. **submit**：仅在正式受信地址公布、入口开放后提交；保留本次提交的 key、submission_id 和工具返回的实际状态，不把受理当作得分。
3. **同 key retry**：网络异常或结果不明确时，沿原 key 和原提交意图重试；不生成新 key 制造第二提交，不把不同源码或身份拼到同一次重试。
4. **status / submissions**：用原身份上下文与同版工具查单次状态及自己的提交列表；`QUEUED`、`RUNNING`、受理成功或列表有记录均不是成绩。真实结果尚未生成时保持未完成，不反推分数。

### 配套 CLI 的现有参数

实际环境变量名为 `STUDENT_ID`、`STUDENT_NAME`、`STUDENT_NICKNAME`，学号作为字符串保留前导零。不要另设 `NAME` 或 `NICKNAME` 冒充上述变量。完成本机环境配置后，`python3 -B -m student_tools config` 只检查本机字段格式，不上传、不回显原值，不证明已登录。

在仓库根目录，学生源码打包的现有命令为：

```sh
python3 -B -m student_tools package --project starter --output my-source.zip
```

目标必须不存在；已有文件不覆盖。这条命令仅说明当前源码工具的打包接口，不表示旧回归实现已经符合新版九分类合同。

在线子命令的参数关系如下，仅作语法说明，**当前不执行在线命令**：

- `submit` 接收 `--archive`、`--ai independent` 或 `--ai ai_assisted`、`--tasks` 四或五个不同任务；提交意图默认保存在本机私有 `.student-intents`。
- `retry` 接收原 key，沿用原意图，不重新传另一份 archive/ai/tasks。key 不是登录凭据。
- `status` 接收原 submission_id；`submissions` 支持 `--limit`、`--offset`。
- 这些在线命令均需正式公告给出的 `--endpoint`，需要路径前缀时使用同公告的 `--path-prefix`；前缀为绝对路径且不以 `/` 结尾。未给地址时不自动连接服务，不猜测地址。

**在线提交、查分、自动榜尚未开放。** 这里保留流程，不提供占位提交 URL 或猜测地址。不要将公开仓库地址当 API 地址，不关闭 TLS 验证、不借用不匹配证书，也不在网络失败时复制、删除或替换服务端活动存储。

## 5. 参考分与课程值

当前 policy：`trial-quality25-advisor-20261020-v1`。每项参考分为 `25 × q`，原始 Q 按任务真实指标计算，q 夹到 `[0,1]`。分类任务为 `0.5 × accuracy + 0.5 × macro_f1`；HMM 为 `0.5 × state_accuracy + 0.5 × exact_sequence_accuracy`；KMeans 为 `0.6 × ari + 0.4 × nmi`。

HMM 整条 20 步状态全部正确才算一次 exact sequence 正确，该项权重 50%；状态准确率高不保证整序列分高，不据此改权重。

- 所选四项全部 READY，所选合计最高 **100**；所选五项全部 READY，最高 **125**。
- 所选任一任务缺测，所选合计为未完成/null；未选显示“未选”，不补 0。真实测得 0 分有效。
- 同一提交至少四项 READY 才有课程值，取该提交最高四项，最高 **100**。选五但仅四项 READY 时可有课程值，所选全部任务合计仍未完成。
- 不跨提交拼分、不混新旧评分规则或评估版本；教师最终课程给分另定，系统参考分不代替人工最终分。

完整定义见[当前试运行评分说明](current-trial-scoring.md)。baseline 后置：没有真实 baseline 就写未测，不拿本地自测、教师合成例或新文档充实测。

## 6. 榜单公开边界与当前状态

榜单要求显示学号、昵称，**不显示姓名**；所有人都可点击查看相应提交允许公开的真实聚合指标及参考分。不得从分数反推指标，不公开逐行预测、隐藏标签、私密日志或身份凭据。

这些是最新课程要求，**不证明 API、在线查分、点击指标或自动榜已经接通**。正式地址、受信 HTTPS、实际开放状态、同版模板/工具/数据同步结果应由真实发布公告确认。旧评分文档与旧发行只作历史，不推荐混用。
