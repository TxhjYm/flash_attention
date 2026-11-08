# Flash Attention (CUDA)

从零实现 FlashAttention 的练习项目：先写朴素 attention 和 online softmax，
再逐步实现 FlashAttention-1 / FlashAttention-2 的 forward、backward 和 causal
版本，与 PyTorch 参考实现对比正确性，并记录性能与显存数据。

## 目录结构

```
flash-attention/
├── README.md
├── CMakeLists.txt                       # 构建配置（默认 sm_86）
├── build.sh                             # ./build.sh [clean|make]
├── run.sh                               # ./run.sh [list|<test-filter> ...]
├── setup.sh                             # 初始化 googletest 子模块
├── sync_up.sh                           # 同步到远程 RTX 3060 机器
├── sync_down.sh                         # 从远程拉回本地
├── .gitignore
├── .rsyncignore
├── docs/
│   ├── design.md                        # 整体设计与分阶段思路
│   ├── memory_analysis.md               # score 矩阵的显存开销分析
│   ├── online_softmax_derivation.md     # online softmax 数学推导
│   ├── backward_derivation.md           # attention backward 的完整数学推导
│   ├── flash2_optimizations.md          # FlashAttention-2 三个优化点的说明
│   ├── benchmark_notes.md               # 性能与显存对比数据
│   └── transformer_mapping.md           # 与 Transformer 架构的对应说明
├── src/
│   ├── naive_attention_forward.cu       # 朴素三步 forward
│   ├── naive_attention_backward.cu      # 朴素 backward
│   ├── softmax_standard.cu              # 标准两次 pass softmax
│   ├── softmax_online.cu                # online softmax 独立实现
│   ├── flash_forward_v1.cu              # FlashAttention-1 forward
│   ├── flash_backward_v1.cu             # FlashAttention-1 backward
│   ├── flash_forward_v2.cu              # FlashAttention-2 forward
│   ├── flash_backward_v2.cu             # FlashAttention-2 backward
│   ├── flash_causal.cu                  # causal mask 版本
│   └── utils.cu                         # 公共工具
├── include/
│   ├── flash_attention.h
│   └── utils.h                          # 公共工具（错误检查等）
├── scripts/
│   ├── verify_forward.py                # forward 正确性验证
│   ├── verify_backward.py               # backward 正确性验证（与 PyTorch autograd 对比）
│   └── benchmark.py                     # 性能与显存 benchmark
├── tests/
│   └── test_flash_attention.cu          # gtest 测试入口
├── third_party/
│   └── googletest/                      # git submodule
└── results/
    ├── correctness_report.md
    ├── benchmark.csv
    └── summary.md
```

## 分阶段计划

1. **参考实现**：朴素 forward（QK^T -> softmax -> PV）与朴素 backward，作为正确性基准。
2. **Softmax**：标准两次 pass 版本与 online softmax 版本，单独验证数值。
3. **FlashAttention-1**：tile 化 + online softmax，forward 后在 backward 中重算中间量。
4. **FlashAttention-2**：调整 tile 划分与线程分工，减少非矩阵乘运算和 shared memory 往返。
5. **Causal mask**：只计算下三角区域。
6. **Benchmark**：不同序列长度、head dimension 下的时间和显存对比。

## 状态

- [ ] 朴素 forward / backward
- [ ] softmax 标准版 / online 版
- [ ] FlashAttention-1 forward / backward
- [ ] FlashAttention-2 forward / backward
- [ ] causal mask
- [ ] 正确性报告与 benchmark

## 构建与运行

首次 clone 后拉取 googletest 子模块：

```bash
./setup.sh
```

构建并运行测试：

```bash
./build.sh make        # 配置并编译，产物在 build/flash_attention_tests
./run.sh               # 运行全部测试
./run.sh list          # 列出测试
./run.sh SoftmaxTest   # 按名称过滤（多个参数用 : 拼接 filter）
./build.sh clean       # 删除 build/
```

默认按 `sm_86`（RTX 3060）编译。换其他 GPU 时重新配置：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build --parallel
```

## 同步到远程 GPU 机器

本地编辑，远程运行（远程目录 `~/work/flash-attention`）：

```bash
./sync_up.sh      # 本地 -> 远程，--delete 会删除远程多余文件
./sync_down.sh    # 远程 -> 本地，不删除本地文件
```

`build/`、`results/`、`.git/` 不参与同步，详见 `.rsyncignore`。
因此远程生成的 benchmark 结果需要手动 `scp` 取回。

## 说明

当前只有项目骨架、构建配置和占位文件，各 kernel 尚未实现；
测试框架已就绪，目前只包含一个 CUDA 设备可用性检查，
实现 kernel 后按 `tests/test_flash_attention.cu` 中的 TODO 补充正确性测试。
