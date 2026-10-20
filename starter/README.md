# Starter：实现入口与新版同步状态

2026-10-20。**新版模板、训练数据、工具同步中；当前公开模板仍可能是旧 GBDT 回归接口。** 本页描述最新课程要求，不证明当前 main 已提供可执行九分类示例。不要按新文档运行混合旧版模板、数据和工具。

## 源码与约束

- 课程仓库：[ZeroGen-Lab/ML-Assignment-1](https://github.com/ZeroGen-Lab/ML-Assignment-1)。使用 GitHub **Code → Download ZIP** 或 clone 获取；不是教师自制发行 ZIP。旧 `v0.3.1` 仅作历史。
- 声明位于 `starter/include/assignment.hpp`，实现位于 `starter/src/assignment.cpp`；相对本目录为 `include/assignment.hpp` 和 `src/assignment.cpp`。
- 使用 **C++17 与 C++ 标准库**，不调用现成机器学习库。保留 `Matrix` 接口，不要求改为纯 C 或连续存储。
- 从 LR、SVM、GBDT、KMeans、HMM 选择四项或五项。题面称 SVM，保留 `LinearSVM` 符号，不据此要求核函数。

## 新版 GBDT 是分类，不是回归

最新合同为 **GBDTClassifier、856 维真实特征、整数九类 0..8**。

- `fit` 接收 `Matrix`、`vector<int>` 训练标签和 `std::uint64_t seed`。
- `predict` 返回与输入行数相同的 `vector<int>`，每值 `0..8`。
- 公开训练 200 行，表头 `label,x000,...,x855`，首列标签，其余 856 列依次为真实词频。
- 保留完整维度；子集中某列全零不表示可删除或补造特征。
- 旧 `GBDTRegressor`、浮点回归输出和 CCPP 数据不适用。若实际头文件仍为旧接口，说明配套同步尚未完成，不要把其签成新版可运行模板。

脚手架中的常量输出仅示例接口，不是训练模型。接口、工具和数据必须按最终同版文件成组使用。

## 数据与 HMM 序列

拟采用公开训练路径为 `data/training/{task}/train.csv`；合并元数据为 `data/training/manifest.json`、`data/training/DATA-SOURCES.md`，GBDT 单项元数据为 `data/training/gbdt/manifest.json`。这些是仓库根路径，**不保证已进入当前 main**。旧 `data/public/training` 仅历史，不推荐混用。

各任务 200 条公开训练记录。HMM 的 200 行表示 **10 条完整序列 × 20 步**，字段 `sequence_id,step,observation,state_label`；监督训练同时使用 observations 和 state_labels，保持边界、步顺序及跨序列一致的标签含义，不拆散或拼接成伪序列。

详细字段、预处理沿革、来源和 HMM 许可 PARTIAL 状态见[数据列说明](../docs/dataset-columns.md)。

## 本地检查与源码提交

本地自测只核接口、格式或有限 sanity 条件，**不是正式分**。新版 GBDT 公开查询为 200 条训练自身，仅检查数量、整数及标签域；合法常量预测也可过门，不证明泛化。自测会在自己的机器编译执行源码，不是沙箱，只执行自己理解并愿意运行的代码。同步完成前不提供混合版本的编译或自测命令。Windows 实机 NOT_RUN。

学生源码提交流程保留 **package → submit → 同 key retry → status / submissions**；package 是打包自己的实现，不是获取课程仓库。当前工具仍要求 `STUDENT_ID`、`STUDENT_NAME`、`STUDENT_NICKNAME`，字段不构成身份认证。网络不明不新建重复提交意图，保留原 key。

在线提交、查分、自动榜单尚未开放，正式 HTTPS URL 未确认；不要使用猜测或占位 URL，不关闭 TLS 校验。

## 分数边界

每项参考分 `25 × q`；HMM 为 `0.5 × state_accuracy + 0.5 × exact_sequence_accuracy`，完整 20 步全对才算整序列正确。所选四项全部 READY 最高 100，所选五项全部 READY 最高 125；课程值取同一提交最高四项，最高 100，教师最终课程分另定。缺测为 None/null、不补 0，未执行指标 NOT_RUN，baseline 后置。

榜单目标显示学号、昵称、不显示姓名；所有人可点击查看允许公开的真实聚合指标，不能据此宣称线上功能已接通。

阅读[学生指南](../docs/student-guide.md)、[当前试运行评分说明](../docs/current-trial-scoring.md)，或返回[仓库首页](../README.md)。
