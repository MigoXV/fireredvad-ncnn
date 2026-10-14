# 代码来源

服务协议、异步引擎、音频处理、后处理和事件水位逻辑迁移自本地项目
`/workspace/opusi/fireredvad`，基线 revision 为
`e476b153af3f44407a355b6de43731bd25c974eb`。
迁移范围在这些模块的 Git 首次提交中可追溯；本项目将模型执行替换为 NCNN，
将运行期张量边界替换为 NumPy，保留 gRPC wire schema 和语义。
本次原生迁移将上述音频、后处理与水位算法移植为 C++，保留 Python 实现在
`tests/reference` 作为验证参考；运行包不再依赖通用 NCNN Python 绑定。

模型结构和特征/后处理代码源于 FireRedTeam/FireRedVAD：
https://github.com/FireRedTeam/FireRedVAD

原文件中的版权声明予以保留。模型来源目录中的模型卡声明 Apache-2.0。
转换产物的元数据记录源权重 SHA-256；本项目未重新发布原模型权重或改变其许可。

## 原生依赖

源码版本和归档 SHA-256 固定在顶层 CMakeLists.txt，构建与 wheel 安装对应许可证：

- Tencent/NCNN：20260526，BSD-3-Clause。
- csukuangfj/kaldi-native-fbank：b09e686fe2084732ddd30d1ef80acfc0f13eaf01，Apache-2.0。
- mborgerding/KissFFT：febd4caeed32e33ad8b2e0bb5ea77542c40f18ec，BSD-3-Clause。
- nlohmann/json：v3.12.0，MIT。
- okdshin/PicoSHA2：161cb3fc4170fa7a3eca9e582cebd27cc4d1fe29，MIT。
- chirlu/soxr：945b592b70470e29f917f4de89b4281fbbd540c0，LGPL-2.1。

libsoxr 默认动态链接并随 wheel 提供可替换的共享库；对应源码来自上述固定 revision。
启用静态 soxr 时，安装其完整源码及原生引擎对象；分发时一并提供依赖归档和
构建配置，允许使用方重建 libsoxr 并重新链接。依赖内含第三方代码的许可说明
保留在其源码和许可证材料中。
