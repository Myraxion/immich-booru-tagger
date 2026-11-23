# Immich Booru Tagger

面向 Immich 的无状态本地自动化图片打标批处理系统，通过视觉模型与元数据规则提取图片的 Booru 体系层级标签并写回 Immich。

## Language

### 核心实体与范围

**Target Album（目标相册）**:
配置中指定的待处理 Immich 相册，可通过相册 UUID 或相册名称定位，并绑定一个独立的 Routing Mode。
_Avoid_: Folder, bucket, directory

**Eligible Asset（待处理资产）**:
Target Album 中类型为图片（`IMAGE`）且尚未携带 Processed Marker（或在强制重打标模式下被选中）的 Immich 资产。
_Avoid_: Photo, file, item

**Processed Marker（远端幂等标记）**:
写回在 Immich 资产上的特殊元数据标签（固定为 `meta/booru_tagged`），作为无本地状态架构下判断资产是否已完成打标的唯一依据。
_Avoid_: Status flag, local state, lock

**Micro-batch（微批次）**:
单次批处理流转与写回的最小资产集合单元：批次内逐张拉取缩略图推理并即时释放图像内存，批次结束时一次性将产出的标签与 Processed Marker 写回 Immich（在强制重打标模式下，写回前先移除该资产已有的受管 Booru 分类标签）。
_Avoid_: Chunk, page, job

### 路由与打标管线

**Routing Mode（路由模式）**:
绑定在 Target Album 上的分流策略，决定资产进入哪条打标管线，取值为 `auto`（通过域分类器自动判别）、`anime`（强制进入二次元打标管线）或 `real`（强制进入三次元打标管线）。
_Avoid_: Pipeline type, album type

**Anime Pipeline（二次元打标管线）**:
针对二次元图像（插画、漫画、番剧截图、3D 渲染等）的串联特征提取流程，组合输出通用特征、角色、推导版权、Danbooru 四级分级以及画面媒介与完成度标签。
_Avoid_: 2D processor, anime model

**Real Pipeline（三次元打标管线）**:
针对三次元真实世界照片的综合特征提取流程，输出人物角色、通用特征以及 Sankaku 三级分级标签。
_Avoid_: 3D pipeline, photo tagger

### 标签体系

**Managed Tag Namespace（受管标签命名空间）**:
由本系统全权维护的六大顶级标签分类前缀集合（`rating/`, `copyright/`, `character/`, `general/`, `meta/`, `artist/`）。在强制重打标（`--force`）时，仅清理属于该命名空间下的旧标签，绝不触碰资产上的用户自定义标签。
_Avoid_: System tags, all tags

**Booru Hierarchical Tag（Booru 层级标签）**:
遵循 `<category>/<tag_name>` 二级扁平层级结构的标签文本，其中 `<category>` 属于 Managed Tag Namespace，`<tag_name>` 统一采用全小写与下划线连接（原有空格与斜杠 `/` 均规范化为 `_`，保留括号 `()` 与冒号 `:` 等 Booru 标准符号）。
_Avoid_: Flat tag, nested path

**Dimension Meta Tag（尺寸元数据标签）**:
基于资产 EXIF 原图真实宽高确定性计算的 `meta/` 标签，包含互斥分辨率分档（`meta/absurdres`, `meta/highres`, `meta/lowres`）与极端长宽比标记（`meta/tall_image`, `meta/wide_image`）。
_Avoid_: Thumbnail size tag, resolution category
