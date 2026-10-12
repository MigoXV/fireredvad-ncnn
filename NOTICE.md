# 代码来源

服务协议、异步引擎、音频处理、后处理和事件水位逻辑迁移自本地项目
`/workspace/opusi/fireredvad`，基线 revision 为
`e476b153af3f44407a355b6de43731bd25c974eb`。
迁移范围在这些模块的 Git 首次提交中可追溯；本项目将模型执行替换为 NCNN，
将运行期张量边界替换为 NumPy，保留 gRPC wire schema 和语义。

模型结构和特征/后处理代码源于 FireRedTeam/FireRedVAD：
https://github.com/FireRedTeam/FireRedVAD

原文件中的版权声明予以保留。模型来源目录中的模型卡声明 Apache-2.0。
转换产物的元数据记录源权重 SHA-256；本项目未重新发布原模型权重或改变其许可。
