# 根因分析与证据记录手册（Root-Cause Analysis）

本文件是 `SKILL.md` 的按需参考：只有在 SKILL.md 正文明确指向、且确实需要规范的证据记录格式或深入案例时才加载。它不替代四阶段工作流，只是把每个阶段的“怎么记录、怎么定位”讲清楚。所有示例命令保持原始语言（shell / Python），说明文字用中文。

## 目录

1. [证据记录格式](#证据记录格式)
   - [stack trace 片段](#stack-trace-片段)
   - [最小复现（minimal repro）记录字段](#最小复现minimal-repro记录字段)
   - [二分定位（bisection）](#二分定位bisection)
   - [依赖假设（dependency assumptions）清单](#依赖假设dependency-assumptions清单)
   - [证据表格模板](#证据表格模板)
2. [worked 案例](#worked-案例)
3. [反模式（anti-patterns）](#反模式anti-patterns)

---

## 证据记录格式

调试的价值取决于证据是否可复核。原则：**把观察写成别人能重跑的东西**，而不是把结论写成断言。

### stack trace 片段

traceback 有两种读法：从头部往下读会先看到入口帧（往往是被截断的、无关的调用栈），从尾部往上读才会先命中真正的抛错点。正确顺序：

1. **从最后一行 `Error` / `Exception` 开始读**：它给出异常类型与消息，这是症状的锚点。
2. **从抛错帧向上（向调用者方向）走**，直到遇到**最早的业务代码帧**（属于本项目仓库、非标准库、非第三方框架内部的帧）。这一帧以上的调用链通常与根因无关，不必粘贴。
3. 只粘贴 **5–8 个最相关的帧**：抛错帧、第一个业务帧、以及中间用于说明数据如何从调用者流到抛错点的那几帧。省略 `site-packages/`、`importlib`、框架调度循环这类噪声帧，但要保留其存在的一行说明（例如“中间有 3 层框架调度帧，已省略”）。
4. 记录时标注 **每个帧的职责**：谁构造了出错的数据、谁把非法值传了下来。

```text
traceback 片段（从底向上截取）：
  Frame 1  app/service/order.py:88  in submit          <- 抛错帧 KeyError: 'amount'
  Frame 2  app/service/order.py:41  in build_payload
  Frame 3  app/api/routes.py:27     in create_order    <- 最早业务帧（入口层）
  （以上为框架调度帧，省略）
记录要点：amount 由 Frame 3 的请求体经 Frame 2 归一化后传入，Frame 2 在缺失键时未补默认值。
```

### 最小复现（minimal repro）记录字段

复现是后续所有验证的地基。稳定复现 = 阶段 1 完成、可以进入假设验证。按以下字段记录：

| 字段 | 说明 |
| --- | --- |
| 触发命令 | 可直接复制粘贴的完整命令，例如 `python -m pytest tests/test_order.py::test_missing_amount -x` |
| 输入 | 触发所用的最小输入/数据/fixture，能贴文本就贴，能贴文件路径就贴 |
| 复现率 | 如 `10/10`（确定性）或 `1/20`（间歇性）；注明统计口径 |
| 环境 | Python 版本、关键依赖版本、OS、配置开关、外部服务是否在跑 |
| 期望 vs 实际 | 期望行为与实际行为的对照 |
| 缩小过程 | 从原始场景到最小复现，去掉了哪些无关变量 |

“缩小过程”常被忽略却很关键：它证明剩余变量才是嫌疑点。若无法稳定复现，明确写“当前不可稳定复现”，并进入二分定位或加大日志的思路，而不是直接猜。

### 二分定位（bisection）

当问题“以前是好的”或“改坏了一半”时，二分是最省的定位法。

**git 历史二分**（找到引入 bug 的那个 commit）：

```bash
git bisect start
git bisect bad HEAD              # 当前是坏的
git bisect good v1.4.0           # 这个 tag/commit 是好的
# git 会自动 checkout 中间 commit，你运行复现命令后标记：
git bisect good   # 或 git bisect bad
git bisect reset  # 定位结束后清理
```

自动化版本（用测试退出码驱动，适合确定性 bug）：

```bash
git bisect run python -m pytest tests/test_order.py::test_missing_amount -x
```

**代码内部二分**（不知道哪段逻辑坏、但能局部禁用时）：把可疑函数体“注释掉一半”，跑复现，观察症状是否还在，逐步把范围收敛到某几行。示例：

```python
def normalize(payload):
    # --- 先注释掉上半段，验证下半段是否仍能触发 ---
    # payload["amount"] = int(payload.get("amount", 0))
    # payload["currency"] = payload.get("currency", "CNY").upper()
    payload.setdefault("status", "new")
    return payload
```

注意：内部二分是**临时诊断手段**，验证完必须还原，不能把注释状态当成修复。

### 依赖假设（dependency assumptions）清单

很多“代码没问题但就是坏了”源于未声明的依赖假设。逐项核对并记录实际值：

- [ ] 运行时版本：Python、关键第三方包（`pip freeze | grep <pkg>`）。
- [ ] 导入路径/工作目录：是否依赖相对路径、`sys.path`、CWD。
- [ ] 环境变量与配置：默认值从哪来、是否被覆盖。
- [ ] 外部服务：数据库、缓存、HTTP 依赖是否可达、返回结构是否变化。
- [ ] 数据形态：输入是 dict / list / None / 空集合，边界值是否被覆盖。
- [ ] 并发模型：进程 / 线程 / 协程，共享状态与初始化顺序。
- [ ] 初始化顺序：单例、连接池、缓存的建立时机是否早于首次使用。

每条假设都要写成“若它不成立，症状会怎样变化”，这样它才能变成阶段 3 的可证伪假设。

### 证据表格模板

把零散观察整理成一张可复核的表。**结论**必须能由同一行的“命令 + 结果”推出；不确定就写低置信度，不要粉饰。

```markdown
| 观测 | 命令 | 结果 | 结论 | 置信度 |
| --- | --- | --- | --- | --- |
| 单元测试在缺失 amount 时抛 KeyError | `python -m pytest tests/test_order.py -x` | KeyError: 'amount'，10/10 复现 | 缺失键路径未兜底 | 高 |
| 加上默认值后测试通过 | 临时改 `payload.get("amount", 0)` | 测试通过，回归全绿 | 该分支即根因所在 | 中 |
| 生产日志中偶发同样报错 | `grep KeyError app.log` | 3 次记录，均在 12:00 前后 | 可能与定时任务输入相关 | 低 |
```

## worked 案例

### 案例 A：确定性异常（KeyError）

**症状**：`POST /orders` 传入不含 `amount` 的 JSON 时，稳定返回 500，日志末尾为 `KeyError: 'amount'`。

**记录**：repro 命令 `curl -X POST /orders -d '{"currency":"CNY"}'`，复现率 10/10。stack trace 从 `KeyError` 行向上截取：抛错帧在 `order.py:88`，最早业务帧在 `routes.py:27`。

**定位**：沿调用链向上，`build_payload` 直接把 `payload["amount"]` 读出，而入口层允许字段缺省。根因是**入口校验与业务读取之间的契约不一致**，不是“curl 用错了”。

**验证与修复**：先加失败测试 `test_missing_amount`（Red），确认修复前失败；再做最小修复——在归一化层补默认值或返回明确的 400（Green）。回归全绿，剩余风险：需要确认下游是否允许 0 作为默认金额。

### 案例 B：pytest 回归（原本绿的测试变红）

**症状**：`tests/test_pricing.py::test_discount` 三天前还通过，现在失败，报 `AssertionError: 90 != 100`。

**记录**：先跑 `git log --oneline -- tests/pricing.py app/pricing.py`，看到近期有一次“重构折扣计算”的提交。repro 命令 `python -m pytest tests/test_pricing.py::test_discount -x`，复现率 10/10。

**定位**：用 git bisect 收敛到那次重构提交：

```bash
git bisect start && git bisect bad HEAD && git bisect good <known-good>
git bisect run python -m pytest tests/test_pricing.py::test_discount -x
```

**模式对比**：在同仓库找到仍正确的 `compute_tax`，它的百分比以小数（0.1）而非整数（10）参与运算；重构后的 `discount` 误用整数百分比。根因是**单位约定在重构中被破坏**。

**验证与修复**：新增一个覆盖单位约定的失败测试（Red），修回小数单位并集中一处转换（Green）。剩余风险：其他调用 `discount` 的地方是否也传了整数，需要补扫一遍。

### 案例 C：间歇性 / 并发 flaky

**症状**：`tests/test_cache.py::test_concurrent_write` 约 1/20 失败，报 `RuntimeError: dictionary changed size during iteration`，本地难复现。

**记录**：repro 命令带 `-p no:randomly --count=50 -x` 反复跑，复现率约 1/20。环境：单机多线程。

**定位**：不要靠加 `time.sleep` 掩盖。改为构造确定性条件——在读写之间插入 `threading.Barrier` 或事件，让竞态必然发生：

```python
barrier = threading.Barrier(2)
# writer / reader 各自在关键点 barrier.wait()，使交错确定化
```

用二分确认是迭代期间被并发修改；模式对比同仓库其他缓存实现，发现它们对外暴露的是快照副本而非活字典。

**验证与修复**：写一个在 Barrier 下必然触发的失败测试（Red），修复为迭代前取副本或加锁（Green）。运行回归并重复 50 次确认不再偶发。剩余风险：其他直接暴露内部字典的接口是否存在同类竞态。

## 反模式（anti-patterns）

- **未调查即改**：看到报错就改代码，跳过复现与证据收集。结果是“改了但不知道为什么好了”，代价是问题复发。
- **同时改多处**：一次提交改 5 个地方，破坏了因果可归因性——即使修好了也无法判断是哪一处起效，回归时更无从定位。
- **用 sleep 掩盖竞态**：用 `time.sleep(1)` 让 flaky 测试“变绿”，只是把概率调到很低，竞态依然存在。应等待**实际条件**（事件、future、条件变量、就绪探针）。
- **把症状当根因**：`None` 导致 `AttributeError` 时，只加 `if x is not None` 而不追查 `x` 为何是 `None`，等于把根因挪到了别处。
- **忽略环境层校验**：只改业务代码，不核对依赖版本、配置、外部服务可用性；很多“代码 bug”其实是环境假设失效。
