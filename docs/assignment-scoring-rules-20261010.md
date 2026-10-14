# 作业一评分与提交概要

概要日期：2026-10-13。五项任务每项25分，可任选四项完成；全部完成时取最高四项合计100分，占课程总评35%。未完成任务计0分，第五项只替换较低分。课程总评60分规定不变。

## 1. 评分概要

- 效果为主，正确性、鲁棒性和效率为辅。综合表现前10%为满分档，随机/无信息基线为基础档；这只是课程概要，不承诺仅凭一个指标或公开样例通过就得到任务25分。
- LR/SVM关注分类准确率与F1，GBDT关注回归误差，K-Means关注聚类质量，HMM关注状态与整条路径正确性。正确性检查接口、顺序及输出语义；鲁棒性关注合法边界、数值稳定和复现；效率评价受控环境中的计算成本。
- 每任务从完整有效提交中选取成绩，不拼不同提交的子分；完成五项时取最高四项。AI辅助和自主完成均可提交，须如实声明并理解、验证实现；AI使用本身不直接加扣分。
- 本机公开诊断仅供开发练习，不产生课程成绩。正式计算、同分处理、截止安排与结算以[课程题面](https://dasellmg.yuque.com/org-wiki-dasellmg-vsl5wg/ml2026/qckpnr5fvr269d83)及公告为准；本概要不修改评测程序或新增精确权重。

## 2. 官方环境与提交规范

- 官方评测使用GCC 12、C++17、`-std=c++17 -O2`，仅依赖C++标准库。学生可用兼容编译器本地开发。
- 保持`assignment`命名空间及公开类名、方法签名不变；可以修改private成员并增加辅助源码。
- ZIP根目录必须包含`README.md`、`AI_USAGE.md`、`include/assignment.hpp`、`src/assignment.cpp`。除此之外仅允许`include/`内的`.h/.hpp`和`src/`内的`.cpp`，不允许嵌套一层项目根目录。
- 两份Markdown均为非空UTF-8文本。README简述结构、任务实现文件和编译注意事项；AI_USAGE简述工具、用途和本人验证，未使用AI时写`No AI assistance used.`。
- 最多64个文件（包含两份Markdown）；每文件最多1,048,576字节，解压总量最多8,388,608字节，ZIP最多8,388,608字节，每成员解压/压缩比最多200（压缩大小为0而内容非空时拒绝）。只接受未加密的stored/deflate ZIP。
- 禁止空文件、NUL、非UTF-8、符号链接、绝对路径、反斜杠、`.`/`..`路径段、重复或NFC+casefold后冲突的路径、Windows设备名、冒号及路径分量尾部点/空格。不得夹带teacher/private、凭证、数据、答案、模型、可执行文件或其他二进制。
- 请填写本人学号、姓名和昵称；设置`export STUDENT_ID=...`、`export STUDENT_NAME="..."`、`export STUDENT_NICKNAME="..."`。学号按字符串处理，保留前导零；提交及结果按学号归属。配置检查不是本人认证，教师权限独立。

## 3. 公开练习与本机诊断

完整学生包内的 `data/public/training/` 包含五份训练CSV：`lr-train.csv`（LR，WDBC，30维）、`svm-train.csv`（SVM，Spambase，57维）、`gbdt-train.csv`（GBDT，CCPP，4维）、`kmeans-train.csv`（KMeans，Seeds，7维，K=3，无公开类别标签）、`hmm-train.csv`（监督HMM，3状态、4观测符号，每条20步）。`data/public/` 包含五任务各自的公开诊断输入、诊断答案和 `manifest.json`；KMeans诊断无类别真值。来源与许可见解压根目录 `data/public/training/DATA-SOURCES.md`。公开数据不是隐藏正式测试，不能加入作业提交ZIP。

在解压根目录选择已实现任务诊断，例如：

```sh
python3 -B -m student_tools selftest --project starter --tasks lr,svm,kmeans,hmm --public-data data/public
python3 -B -m student_tools package --project starter --output submission.zip
```

GBDT可加入任务列表。空模板未实现时失败是预期结果，不要硬编码公开答案。**本机自测会编译并执行C++，不是安全沙箱；只运行自己编写或已审查的代码。**

## 4. 提交、时间与榜单

提交与成绩开放以课程公告为准；公告给出HTTPS入口、开始/截止时刻和反馈渠道。不要向猜测的地址发送身份或源码。包内[成绩榜单](../public/leaderboard.html)可先阅读；file离线预览显示“暂无成绩”，不请求网络、不生成排名或分数。在线页面只读取当前站点的公开榜单，不把本机诊断当作可信成绩。

以服务端完成收包及校验后的接收时刻判断截止；本机打包或请求开始时刻不是接收时刻。每学号每天（Asia/Shanghai）最多10个新提交意图，最多1个未完成评测任务；原幂等意图重试不重复占次数。超时不等于失败，重试保留原幂等键、相同ZIP字节与AI声明；限额拒绝按反馈处理，不持续创建新请求。

基础设施故障、损坏数据或无法验证的评测证据不补学生0分，也不给默认满分；评测暂缓，修复后按相同规则重评。未完成评测不能伪装成正式成绩；服务中断、统一顺延和重评版本按课程公告处理，保留原记录与原因。

使用说明见[学生指南](student-guide.md)。首版材料能用于开始实现，不表示正式评测服务已经开放。
