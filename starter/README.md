# ML-Assignment-1 学生模板

五项任务：Logistic Regression、Linear SVM、GBDT Regression、K-Means、Discrete HMM。每项25分，可任选四项；完成五项时取最高四项，作业100分，占课程总评35%。效果为主，正确性、鲁棒性和效率为辅；综合表现前10%为满分档，随机/无信息基线为基础档。评分概要见解压根目录 `docs/assignment-scoring-rules-20261010.md`；正式细则、提交与成绩开放及截止时刻以课程题面和公告为准。

## 开始

在同时包含 `starter/`、`student_tools/`、`docs/` 的解压根目录执行：

```sh
python3 -B -m student_tools selftest --project starter --tasks lr,svm,kmeans,hmm --public-data data/public
python3 -B -m student_tools package --project starter --output submission.zip
```

Windows若使用`python`命令，将`python3`替换为`python`。需要Python3.10+与支持C++17的g++/clang++；本工具不自动安装编译器。官方评测使用GCC 12、C++17、`-std=c++17 -O2`，学生可用兼容编译器本地开发。

空模板全部方法未实现，公开自测失败是预期结果。先完成算法、填写两份文档，再测试和打包。**本机自测会执行你的C++代码，不是安全沙箱；只运行自己编写或已经审查的代码。** 公开自测不产生正式课程成绩；打包成功不等于算法正确，本机结果不能作为可信成绩上传。

## 作业区域

- `starter/include/assignment.hpp`：保留`assignment`命名空间、类名、public方法参数/返回类型。
- `starter/src/assignment.cpp`：填写算法实现。
- 可修改private成员，可在include/src中增加标准库辅助头文件和C++源码；GBDT不限定使用模板的树桩结构。
- `starter/tools/driver.cpp`为公开API适配层，学生不需要实现CLI，不随源码提交。
- HMM要求Viterbi联合最优路径，不是逐位置后验最大状态。

## 源码提交包

ZIP根目录必须包含四个文件：`README.md`、`AI_USAGE.md`、`include/assignment.hpp`、`src/assignment.cpp`。除此之外仅允许`include/`内的`.h/.hpp`与`src/`内的`.cpp`。不要在ZIP里再套一层`starter/`目录。打包器只收这两份根文档和允许的源码，不收tools、teacher/private、其他根文件。

两份Markdown必须为非空UTF-8文本，不可只填空白。将本README替换为自己项目的结构、实现文件和编译说明；`AI_USAGE.md`按实际情况填写。未使用AI时明确写`No AI assistance used.`；不要把模板中的待填内容当最终声明。

限额：64文件（含两份Markdown）、单文件1MiB、解压总量8MiB、ZIP 8MiB、每成员解压/压缩比最多200。只接受未加密的stored/deflate ZIP；拒绝空文件、符号链接、路径逃逸、重复/NFC+大小写冲突、Windows不安全路径、二进制/NUL和非UTF-8文本。已有ZIP不会覆盖。不要提交数据、答案、模型、可执行文件、凭证或第三方机器学习库。

## README填写示例

下列内容须按自己的实现修改；模板不证明已经完成或验证算法。

```text
项目结构：include/assignment.hpp定义公开接口；src/assignment.cpp实现算法。
任务实现：LR：[填写文件和方法]；SVM：[填写]；GBDT：[填写]；KMeans：[填写]；HMM：[填写]。
编译注意：C++17，仅依赖标准库；官方使用GCC 12和-O2。[填写自己的其他注意事项]
本地验证：[填写实际执行的公开自测及结果；未通过项如实列出]
```

## 提交身份与AI声明

使用工具前填写本人信息，示例值必须替换：

```sh
export STUDENT_ID="你的学号"
export STUDENT_NAME="你的姓名"
export STUDENT_NICKNAME="你的昵称"
```

学号按字符串处理，保留前导零；提交及结果按学号归属，配置检查不是本人认证。教师权限独立。AI辅助和自主完成均可提交，不直接加扣分；AI使用说明随ZIP提交，工具的`ai_assisted`/`independent`标签仍须逐提交按实际情况声明，文档不替代该标签。

公开训练数据在解压根目录 `data/public/training/`：`lr-train.csv`（LR，WDBC，30维）、`svm-train.csv`（SVM，Spambase，57维）、`gbdt-train.csv`（GBDT，CCPP，4维）、`kmeans-train.csv`（KMeans，Seeds，7维，K=3，无公开类别标签）、`hmm-train.csv`（监督HMM，3状态、4观测符号，每条20步）。`data/public/`含五任务公开诊断输入、诊断答案及 `manifest.json`；KMeans诊断无类别真值。来源与许可见解压根目录 `data/public/training/DATA-SOURCES.md`。选择已实现任务自测，GBDT可加入 `--tasks` 列表；公开数据不随作业源码提交。

完整使用说明见解压根目录 `docs/student-guide.md`。包内 `public/leaderboard.html` 可直接打开；file离线预览显示“暂无成绩”，不请求网络、不生成排名。真实提交、查询与成绩开放以课程公告为准，本机诊断不是课程成绩。
