# ML Assignment 1

C++17 机器学习课程作业：从 LR、SVM、GBDT、KMeans、HMM 中选择四项或五项，提交自己的实现。

> **2026-10-20 课程说明更新；新版代码、数据和工具同步中。** 当前公开模板仍可能含旧 GBDT 回归接口，不能据本说明认定仓库已经提供可运行的九分类模板。必须使用同一版本的接口、数据、工具与文档，不能把旧版回归文件和新版分类合同混用。在线提交、查分、自动榜单尚未开放；正式 HTTPS 地址未确认。Windows 实机验证为 **NOT_RUN**。

## 获取仓库

公开仓库：[ZeroGen-Lab/ML-Assignment-1](https://github.com/ZeroGen-Lab/ML-Assignment-1)。

- 使用 GitHub 的 **Code → Download ZIP** 下载仓库源码；这是 GitHub 仓库快照，不是教师另行制作的发行 ZIP。
- 或在自己的机器上执行：

```sh
git clone https://github.com/ZeroGen-Lab/ML-Assignment-1.git
cd ML-Assignment-1
```

不再以教师自制发行 ZIP 作为新版获取入口。旧 `v0.3.1` 发行和旧 `data/public/training` 文件仅为历史，不推荐与新版混用。学生提交自己的源码时仍需使用配套工具的 `package` 操作；它与获取课程仓库是两回事。

## 从哪里开始

1. 阅读[学生指南](docs/student-guide.md)，核对同步状态与本地执行风险。
2. 阅读[接口与实现入口](starter/README.md)，使用 C++17 和 C++ 标准库，保留 `Matrix` 接口，不调用现成机器学习库。
3. 阅读[数据列说明](docs/dataset-columns.md)。拟采用公开训练路径为 `data/training/{task}/train.csv`；路径出现在说明中不代表该文件已经进入当前 main。
4. 阅读[当前试运行评分说明](docs/current-trial-scoring.md)。本地自测不是服务器正式分；baseline 后置，不编造缺测成绩。

## 最新课程合同

- **GBDT 是九类分类**：真实 856 维词频特征，整数标签 `0..8`，接口为 `GBDTClassifier`，不再使用 `GBDTRegressor` 或旧回归数据。公开训练子集中非零列较少不表示降维，不补维、不裁列。
- 题面称 **SVM**，C++ 符号沿用 `LinearSVM`，不据此要求核函数。
- 各任务提供 200 条公开训练记录；HMM 的 200 行为 **10 条完整序列 × 20 步**，不是 200 条独立序列。HMM 为监督训练，保留序列边界与一致的状态标签含义。
- 每项参考分为 `25 × q`；分类质量为 `0.5 × accuracy + 0.5 × macro_f1`；HMM 为 `0.5 × state_accuracy + 0.5 × exact_sequence_accuracy`；KMeans 为 `0.6 × ari + 0.4 × nmi`。原始 Q 保留，q 夹到 `[0,1]`。
- 所选四项全部完成时合计最高 **100**；所选五项全部完成时最高 **125**。课程值取同一提交中已完成的最高四项，最高 **100**，不跨提交拼分；教师最终课程给分另定。
- 缺测保留 `None/null`，未执行为 `NOT_RUN`，不补 0；真实测得 0 分与未测不同。

## 提交流程与身份信息

配套学生源码工具保留 **package → submit → 同 key retry → status / submissions** 流程。当前工具仍要求 `STUDENT_ID`、`STUDENT_NAME`、`STUDENT_NICKNAME`，不得因榜单不显示姓名就自行省略 `STUDENT_NAME`；这些字段不构成身份认证。

在线入口尚未开放时，不填占位地址、不向猜测的 URL 发请求、不关闭 TLS 证书校验。正式地址和配套版本公布后，再按[学生指南](docs/student-guide.md)操作。网络失败不新建重复提交意图，同一提交重试必须沿用原 key。

榜单目标是显示学号与昵称、**不显示姓名**；所有人均可点击查看该提交允许公开的真实聚合指标。这里只说明最新课程要求，不证明服务、查分或榜单已经接通，不公开逐行预测、隐藏标签或私密日志。

## 本地检查不等于正式成绩

本地自测只检查接口、输出格式或有限 sanity 条件，不生成正式参考分。新版 GBDT 公开检查以 200 条训练自身作查询；合法常量输出也可通过数量/整数/标签域检查，不能据此证明学习有效或泛化能力。自测会在自己的机器上编译并执行源码，不是沙箱；只运行自己理解并愿意执行的代码。

新版模板、数据与工具尚未成组同步时，不执行混合版本示例。历史评分文档仅作记录，以[当前试运行评分说明](docs/current-trial-scoring.md)为准。
