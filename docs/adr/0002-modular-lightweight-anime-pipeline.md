# 二次元管线采用主力语义模型与极轻量专用模型加确定性规则的组合架构

二次元打标管线不采用单一模型输出全部维度，而是将职责拆分为三层组合：
1. 主力语义模型 `anime_pixai` 负责提取通用特征（`general`）、角色（`character`）并基于角色关联推导作品版权（`copyright`），且不通过视觉模型猜测画师（`artist`）。
2. 三个极轻量 MobileNetV3 模型分别专职负责 Danbooru 四级内容分级（`animetimm/mobilenetv3_large_150d.dbv4-full` 取分级最高置信度项）、载体形式（`anime_classify`）与完成度识别（`anime_completeness`）。
3. 图像尺寸与比例标签（`absurdres`、`highres`、`lowres`、`tall_image`、`wide_image`）不依赖视觉模型对缩略图的推理，而是直接基于 Immich 资产 EXIF 中的原图真实宽高进行确定性计算。

这种组合规避了使用缩略图推理导致模型误判原图分辨率的问题，消除了视觉模型盲猜画师带来的高误报率，并以极低的额外算力开销补齐了 `anime_pixai` 缺失的 Danbooru 标准四级分级与媒介形式元数据。
