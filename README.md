# FireRedVAD NCNN

通过 NCNN Python 绑定提供 CPU FP32 流式 VAD，兼容原 FireRedVAD gRPC 协议。
使用 Poetry 管理环境；模型转换依赖单独放在 `export` 组。

```bash
poetry env use python3.10
poetry install
# 转换和开发环境
poetry install --with export
```

实现、模型转换和验证命令随对应功能提交补充。
