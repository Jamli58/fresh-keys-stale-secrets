# AutoDL 运行手册（M2：R011 / R012 / R014 / R031）

论文数字全部在 AutoDL 上用 vLLM 跑；Mac 上的 Ollama 只用来调试。

| key | 模型（ModelScope） | vLLM 服务名 | 显存 | 结果编号 |
|---|---|---|---|---|
| `llama8b` | LLM-Research/Meta-Llama-3.1-8B-Instruct（bf16，约 16 GB） | `llama3.1-8b-instruct-bf16` | 24 GB 够 | R011 |
| `qwen32b` | Qwen/Qwen2.5-32B-Instruct-AWQ（约 19 GB） | `qwen2.5-32b-instruct-awq` | 24 GB 够（上下文 4096） | R012 |

## 0. 租实例

- GPU：1 × RTX 4090（24 GB）。镜像选 PyTorch + CUDA 12.x 的任意版本即可（Python 由 uv 另装 3.12）。
- 数据盘（`/root/autodl-tmp`）至少留 60 GB：两个模型约 36 GB，外加 venv。

## 1. 上传代码（在 Mac 上）

```bash
export AUTODL_HOST=connect.xxx.seetacloud.com AUTODL_PORT=12345   # 控制台「登录指令」里的主机和端口
bash cloud/sync.sh push
```

## 2. 装环境并下载模型（在 AutoDL 上；可以先用「无卡模式」开机，更便宜）

```bash
cd /root/autodl-tmp/numpriv/pilot
bash cloud/setup.sh             # uv、vLLM venv、项目 venv，并跑一遍测试
bash cloud/download_models.sh   # 下载后逐个文件核对 sha256 → results/model_provenance_<key>.json
```

## 3. 先跑 dev（换到有卡模式）

```bash
nohup bash cloud/run_all.sh dev > /root/autodl-tmp/run_all_dev.log 2>&1 &
tail -f /root/autodl-tmp/run_all_dev.log
```

每个模型的流程：启动 vLLM → R031 冒烟测试（`results/LLM_SMOKE_autodl_<key>.json`）→ 50 家 dev 公司 × 全部防御 × A2/A3/A4 → 停止 vLLM。

跑完拉回 Mac（`bash cloud/sync.sh pull`，结果放在 `results/autodl/`），检查失败率、A3 截断和吞吐。如需改 prompt，只能在 dev 上改，每改一次都要升 `PROMPT_VERSION`（每个攻击者最多 10 个变体）。

## 4. 测试集（只跑一次）

```bash
CONFIRM_TEST=1 SHUTDOWN_AFTER=1 nohup bash cloud/run_all.sh test > /root/autodl-tmp/run_all_test.log 2>&1 &
```

- 3 个防御 seed，覆盖 R014；输出 `results/R011_test_llama8b.json` 和 `results/R012_test_qwen32b.json`，以及对应的 `_rows.jsonl`。
- `SHUTDOWN_AFTER=1`：跑完自动关机，停止计费，数据盘保留。

## 注意事项

- **中断后可以续跑**：每个成功的回答都缓存在 `results/llm_cache/<server>__<model>.jsonl`，重新执行同一命令即可，已缓存的请求不会重复调用。
- **可复现性**：vLLM 在并发批处理下，即使 temperature 为 0，结果也可能随批次组成出现微小差异。以缓存文件作为本次运行的原始记录，归档时连同 `vllm_env_<key>.txt` 和 `model_provenance_<key>.json` 一起保存。
- **单独操作**：`bash cloud/serve.sh qwen32b` 启动服务，`bash cloud/stop.sh` 停止，日志在 `results/logs/`。
- **只跑一部分**：给 `run_m2.sh` 追加参数，例如 `bash cloud/run_m2.sh llama8b dev --max-firms 10 --attackers A4`。
