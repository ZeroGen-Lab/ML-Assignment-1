# ML Assignment 1

本课程五项 C++17 算法作业：逻辑回归、SVM、GBDT、KMeans、监督 HMM。每项 25 分，选五取四计 100 分，占课程总评 35%。只提交指定源码与 AI 使用声明；客户端自报分数不作为正式评分。

## 先看什么

阅读 [学生指南](docs/student-guide.md)、[评分规则](docs/assignment-scoring-rules-20261010.md) 和 [模板说明](starter/README.md)。实现接口位于 `starter/include/assignment.hpp`，模板位于 `starter/src/assignment.cpp`；模板能编译不代表算法已完成。

## 数据与公开诊断

数据版本 `course-uci-20261014-v1`；公开训练 CSV、来源许可与精确 SHA 位于 `data/public/training/`。

|任务|数据|维度/结构|正式训练/测试|
|---|---|---|---|
|LR|WDBC|30 维，0/1|454 / 115|
|SVM|Spambase|57 维，0/1|1194 / 606|
|GBDT|CCPP|4 维回归|1199 / 601|
|KMeans|Seeds|7 维，k=3，无公开类别标签|168 / 42|
|HMM|监督序列|3 状态、4 符号、每条 20 步|60 / 30 条|

数值特征已用正式训练集拟合后标准化，不要重复标准化。HMM 保留完整序列。课程不发布正式测试集；UCI 原始数据公开，原始标签可能从公开来源恢复，因此不能声称正式测试标签不可恢复。

`data/public/` 根目录的十个诊断成员及 `manifest.json` 是从公开训练部分划出的本地诊断，不是正式 holdout 或正式分数。连续任务使用 120/30 行，HMM 使用 8/2 条完整序列；KMeans 只做无标签诊断，不报告有真值的 ARI/NMI。

## 本地运行

```sh
python3 -B -m student_tools.identity
python3 -B -m student_tools.selftest --project starter --public-data data/public --tasks lr,svm,gbdt,kmeans,hmm
python3 -B -m student_tools.package --project starter --output submission.zip
```

新版公开诊断完整命令见学生指南随包说明。空模板或占位输出可能失败，不能据此宣称算法完成。源码 ZIP 不带训练 CSV；本地诊断分数不作为正式成绩。

身份使用自报学号、姓名与昵称，不设置个人 token/密码；自报身份不证明本人。服务地址与开放状态以课程另行通知为准，本 README 不提供虚构生产 URL。

## 安全边界

本机自测由你主动运行自己的程序，**不是沙箱**。静态网页/GitHub Pages 不运行正式学生提交。教师评分物料、正式 holdout、truth、DB、名册与凭据不属于学生发行包。
