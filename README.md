# Aethelis

Aethelis 是一个面向玩家的 AI-native 互动式动态世界项目。当前版本提供一个可在本地浏览器中运行的单人 2D/2.5D 叙事与社会模拟切片：玩家可以在 Mistgate 城中移动、调查、与角色交谈、表达自然语言意图，并通过受治理的行动影响角色认知、关系、资源和世界状态。

项目的核心不是让大语言模型直接编写世界真相，而是让角色 Agent 和 AI 在明确的世界规则内提出行动与表达，再由 Aethelis World Engine 验证、提交并持久化合法后果。

## 当前体验

- 第三人称斜俯视 2D/2.5D 场景，支持玩家移动、镜头跟随和空间交互。
- 地图、背包、任务、日志及世界状态覆盖层。
- 传统预设选项与自然语言自由表达结合的多轮对话。
- NPC 局部认知、记忆、关系和来源受限的玩家信息学习。
- 有限的 Agent / Multi-Agent 行动、信息传播和世界时间推进。
- 资源、承诺、压力、修复和结局均经过确定性治理与持久化。
- 保存、恢复、刷新和时间线分叉。
- Mistgate 非固定顺序推进：零件、校准钥匙和闸门透镜可以按不同顺序调查，履约与失约会保留不同的社会后果。

当前内容版本为 `mistgate_product_v1_10_0`。

## World Engine 治理原则

所有能够改变世界的行为都遵循以下链路：

```text
PlayerCommand / ActionProposal
-> EventCandidate
-> VerificationResult
-> CommittedEvent
-> StateDiff
-> WorldState
```

- Canon、角色认知和玩家主张彼此分离。
- 玩家、Agent 和 LLM Provider 都不能直接修改 `WorldState` 或 Canon。
- Provider 负责受约束的理解与表达，不拥有资源、权限、承诺、修复或结局决定权。
- 只有通过验证并提交的事件才能产生 `StateDiff`。
- 世界变化、角色认知、记忆、关系和因果结果可以追踪并持久化。

## 技术栈

- 后端：Python 3.11+、FastAPI、Pydantic、SQLAlchemy、Alembic
- 数据库：PostgreSQL、pgvector
- 前端：React 19、TypeScript、Vite、PixiJS 8
- 包管理：`uv`、pnpm 11（可通过 Corepack 使用）
- 测试：pytest、Vitest

## 主要目录

```text
src/aethelis/   World Engine、Agent、治理、持久化和 Product API
frontend/       React / PixiJS 玩家客户端与 Mistgate 游戏资产
content/        不可变、版本化的产品世界内容包
alembic/        PostgreSQL 数据库迁移
configs/        运行时和机制配置
scripts/        本地启动与验证脚本
seeds/          Mistgate 等世界种子
tests/          单元、集成和 smoke 测试
```

## 本地运行要求

建议使用 Windows PowerShell。开始前需要：

- Python 3.11 或更高版本
- [uv](https://docs.astral.sh/uv/)
- Node.js 24.11.1 或更高版本
- pnpm 11.8.0，或已启用 Corepack
- PostgreSQL，并安装可用的 pgvector 扩展

本地单用户模式不需要 Auth0、OIDC 租户或登录界面。自然语言 Provider 是可选配置；没有 Provider 凭据时，相关能力会使用受控回退，但数据库仍是产品持久化真相。

## 安装

克隆仓库并进入项目目录：

```powershell
git clone https://github.com/Lelov093/Aethelis.git
Set-Location Aethelis
```

创建本地环境文件：

```powershell
Copy-Item .env.example .env
```

至少需要在 `.env` 中配置可访问的 PostgreSQL `DATABASE_URL`。如需真实 LLM 或 Embedding Provider，再填写相应的 API 地址、模型和密钥。不要提交 `.env`。

安装 Python 和前端依赖：

```powershell
uv sync --extra dev
corepack pnpm --dir frontend install --frozen-lockfile
```

应用数据库迁移：

```powershell
uv run alembic upgrade head
```

## 启动完整本地游戏

在仓库根目录执行：

```powershell
.\scripts\dev.ps1
```

脚本会分别启动：

- Product API：`http://127.0.0.1:8000`
- Governance Worker
- 玩家客户端：`http://127.0.0.1:5173`

脚本不会自动安装或删除 PostgreSQL。进程日志保存在 `tmp/dev/`，该目录不会提交到 Git。

只检查三个服务能否启动并在就绪后退出：

```powershell
.\scripts\dev.ps1 -ExitAfterReady
```

如果只想分别启动后端进程：

```powershell
uv run aethelis-api
uv run aethelis-worker
```

## 验证

运行后端测试：

```powershell
uv run pytest
```

运行前端检查和测试：

```powershell
corepack pnpm --dir frontend lint
corepack pnpm --dir frontend test
corepack pnpm --dir frontend build
```

部分 PostgreSQL 集成测试默认跳过，需要显式设置测试门控，并要求本机数据库可用。

## 当前边界

Aethelis 当前是一个已经连接真实持久化、World Engine 治理、有限 Agent 活动和玩家界面的产品垂直切片，不代表：

- 无限或连续运行的自主世界模拟；
- 通用开放世界或完整商业游戏；
- LLM 可以绕过世界规则直接改变事实；
- 已完成 Tauri 桌面打包、云端托管或生产运营体系。

## 安全说明

- 不要提交 `.env`、API 密钥、Authorization Header 或数据库凭据。
- 不要将包含私有角色认知、隐藏 Canon 或原始 Provider 输出的本地运行记录公开。
- `runs/`、临时日志、缓存、虚拟环境和前端依赖目录默认被忽略。

## License

除另有说明外，Aethelis 源代码采用 [Apache License 2.0](LICENSE)。

`frontend/public/assets/mistgate/` 下的 AI 辅助视觉资产保留独立的来源记录和分发审核要求；当前将其包含在私有仓库中，不代表已经完成公开再分发审核。
