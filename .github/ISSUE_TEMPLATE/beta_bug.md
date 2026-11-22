---
name: 公测 Bug / 画质问题
about: 报告花屏、噪点、报错、崩溃等公测期问题
title: "[Beta] "
labels: bug
---

**问题类型**
<!-- 勾选一个 -->
- [ ] 花屏 / 噪点 / 色块
- [ ] 画面全黑、全白或纯色
- [ ] 形状/维度报错（RuntimeError）
- [ ] 内存或显存溢出
- [ ] 其他

**复现信息**

| 项 | 值 |
| --- | --- |
| 版本 | v2.0.2-beta |
| 使用的权重 | weights/vae_latest_96px.pt + dit_latest_96px.pt |
| prompt |  |
| seed |  |
| 采样步数 `--steps` |  |
| 尺寸 `--size` |  |
| cfg `--cfg-scale` |  |
| 模型参数 `--vae-dim/--dit-dim/--dit-depth/--dit-heads` |  |
| 运行设备 | CPU / GPU（型号） |
| Python / torch 版本 |  |

**完整命令**

```
把实际执行的命令贴在这里
```

**报错日志或截图**

```
把完整 traceback 贴在这里（不要只贴最后一行）
```

**期望结果**
<!-- 你原本期待看到什么样的画面 -->
