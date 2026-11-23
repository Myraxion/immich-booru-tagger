# Immich Booru Tagger

面向 **Immich** 的本地自动化图片打标 CLI 工具，采用**单机直连、主动拉取、无本地状态**的批处理架构。通过本地 ONNX 视觉模型与原图 EXIF 规则为二次元（Anime）与三次元（Real）图片自动生成遵循 Booru 层级规范的标签并批量写回 Immich。

## 核心特性

- **无中间服务、无本地数据库**：直接直连 Immich REST API，完全依托写回资产上的 `meta/booru_tagged` 标签实现增量检索与重复执行幂等性，不维护本地 SQLite 或状态文件。
- **双域智能分流（`auto` / `anime` / `real`）**：
  - `auto` 模式通过轻量 `anime_real` 分类器自动判别二次元/三次元并路由到对应专用管线；
  - 支持在相册级别强制指定 `anime` 或 `real` 跳过分类。
- **二次元轻量精准组合（Anime Pipeline）**：
  - `anime_pixai` (`v0.9`) 提取 `general`、`character` 并自动推导作品版权 `copyright`（不通过视觉模型猜测画师 `artist`）；
  - `animetimm/mobilenetv3_large_150d.dbv4-full` 提取 Danbooru 原生四级分级（`general` / `sensitive` / `questionable` / `explicit`）；
  - `anime_classify` 与 `anime_completeness` 提取媒介形式（`illustration` / `bangumi` / `comic` / `3d`，自动忽略 `not_painting`）与完成度（`polished` / `rough` / `monochrome`）。
- **三次元专用管线（Real Pipeline）**：
  - 使用 `real_idolsankaku` 提取三次元 `character`、`general` 及 Sankaku 原生三级分级（`safe` / `questionable` / `explicit`）。
- **EXIF 真实尺寸规则**：直接读取 Immich EXIF 原图真实宽高生成 `meta/absurdres`、`meta/highres`、`meta/lowres`、`meta/tall_image`、`meta/wide_image`，避免缩略图推理导致分辨率误判。
- **微批次流转与安全重跑**：逐张拉取缩略图到内存推理后即时释放图像内存，按 `batch.size` 分批落库；支持 `--force` 在清理六大受管前缀旧标签的同时完整保留用户手动添加的个人标签。

---

## 环境要求与安装

- **Python**：`>= 3.11, < 3.13`（推荐 Python 3.12）
- **包管理器**：[uv](https://docs.astral.sh/uv/)

克隆仓库后，根据硬件选择 **GPU（NVIDIA CUDA 12）** 或 **CPU** 依赖组进行安装：

```bash
# GPU 模式（推荐，纯 uv 托管 CUDA 12 / cuDNN 9 运行时，无需系统全局安装 cuDNN）
uv sync --extra gpu

# 或 CPU 模式
uv sync --extra cpu
```

### Hugging Face 授权准备（必须）

二次元四级分级模型 [`animetimm/mobilenetv3_large_150d.dbv4-full`](https://huggingface.co/animetimm/mobilenetv3_large_150d.dbv4-full) 为 Hugging Face 受限仓库（Gated Repo）。首次运行前请完成：

1. 登录 Hugging Face 并前往该模型页面点击同意访问协议；
2. 在本地登录 Hugging Face CLI（或设置环境变量 `HF_TOKEN`）：
   ```bash
   uv run --extra gpu hf auth login
   ```
   *(程序会自动将本地 `hf auth login` 缓存的 Token 透传给底层推理客户端。)*

---

## Immich API Key 权限配置

在 Immich **Account Settings → API Keys → New API Key** 中创建密钥时，按最小权限原则勾选以下 **5 项权限**：

| 权限 | 用途 |
| :--- | :--- |
| `album.read` | 读取目标相册列表与相册资产 |
| `asset.read` | 读取资产详情以补齐已有标签与 EXIF 信息 |
| `asset.view` | 拉取资产 `preview` / `thumbnail` 缩略图流用于推理 |
| `tag.create` | 批量创建尚不存在的 Booru 层级标签（部分版本可一并勾选 `tag.read`） |
| `tag.asset` | 为资产绑定新标签，以及 `--force` 时解绑旧受管标签 |

> 无需授予 `all`、`asset.download`、`asset.delete` 或 `tag.delete` 等高危权限。

---

## 配置文件

复制模板文件 [config.example.toml](./config.example.toml) 为 `config.toml`（已加入 `.gitignore`）并修改：

```toml
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "your-immich-api-key"
thumbnail_size = "preview"        # "preview"（高清预览图）或 "thumbnail"（小尺寸缩略图）
timeout = 30.0

[batch]
size = 20                         # 微批次大小，每满该数量资产立即向 Immich 写回一次

[pipeline.anime]
general_threshold = 0.35
character_threshold = 0.85

[pipeline.real]
general_threshold = 0.35
character_threshold = 0.85

[[albums]]
name = "Mixed Album"              # 每个相册必须且只能指定 name 或 id（UUID）之一
mode = "auto"                     # "auto" | "anime" | "real"
# artist_patterns = [             # 可选：按顺序从文件名（不含扩展名）提取 artist/* 标签，命中第一条有效规则即止
#   '^\[(?P<artist>[^\]]+)\]',    # 优先读取 (?P<artist>...) 命名捕获组，否则取第 1 个捕获组
#   '^pixiv_\d+_(.+)$',
# ]
```

- 若希望避免在文件中保存明文密钥，可通过环境变量覆盖：
  - `IMMICH_BASE_URL`
  - `IMMICH_API_KEY`
- 若使用 `name` 匹配到多个同名相册，程序会在启动阶段立即报错并列出所有同名相册的 UUID，以便改用 `id` 精确指定。

---

## 使用方法

运行命令时请带上安装时对应的 `--extra gpu` 或 `--extra cpu`：

```bash
# 1. 空跑预览（仅本地推理并打印结果，不向 Immich 发送任何写请求）
uv run --extra gpu immich-booru-tagger --config config.toml --dry-run --limit 5

# 2. 正式执行批处理打标（自动跳过已携带 meta/booru_tagged 的图片）
uv run --extra gpu immich-booru-tagger --config config.toml

# 3. 强制重打标（忽略 meta/booru_tagged，清理旧受管标签后重新生成，保留个人自定义标签）
uv run --extra gpu immich-booru-tagger --config config.toml --force
```

### CLI 参数说明

| 参数 | 说明 |
| :--- | :--- |
| `-c`, `--config <path>` | 指定配置文件路径（默认读取当前目录 `config.toml`） |
| `--dry-run` | 空跑模式，仅在终端输出预测标签，不修改 Immich 远端数据 |
| `--limit <N>` | 限制本次最多处理的待打标图片数量（须为正整数） |
| `--force` | 强制重打标，先解绑资产上受管命名空间的旧标签再重新生成写入 |

---

## 标签体系与规范

所有产出的标签均统一格式化为二级扁平层级 `<category>/<tag_name>`（全小写，空白字符与标签内原有的 `/` 统一转换为 `_`，保留 `()` 与 `:` 等 Booru 标准符号）：

- **受管六大分类前缀（Managed Tag Namespace）**：
  - `rating/*`：内容分级（二次元四级 / 三次元三级，每张图唯一）
  - `copyright/*`：作品版权 / IP（由角色自动推导）
  - `character/*`：角色 / 人物名称
  - `general/*`：通用画面特征
  - `meta/*`：媒介形式、完成度、原图尺寸属性及幂等标记 `meta/booru_tagged`
  - `artist/*`：画师 / 作者命名空间（通过相册 `artist_patterns` 正则从文件名提取）
- **原图尺寸与比例规则（`meta/*`）**：
  - 分辨率互斥分档：
    - `meta/absurdres`：原图宽 `>= 3200` 或高 `>= 2400`
    - `meta/highres`：未达 `absurdres`，且宽 `>= 1600` 或高 `>= 1200`
    - `meta/lowres`：宽 `<= 500` 且高 `<= 500`
  - 极端长宽比（独立叠加）：
    - `meta/tall_image`：高宽比 `>= 2.5`
    - `meta/wide_image`：宽高比 `>= 2.5`

---

## 开发与测试

```bash
# 运行全部单元测试与 Mock 集成测试
uv run --extra gpu pytest -v

# 运行严格静态类型检查
uv run --extra gpu mypy src tests
```
