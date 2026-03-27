# Word2Vec Demo (Python 3.11 + Conda)

一个可开箱即跑的词向量训练与可视化示例项目。

## 项目目标

- 使用 `gensim` 训练 Word2Vec 词向量模型
- 保存训练后的模型到 `models/`
- 运行主程序后，自动绘制 10 个词向量在二维空间中的分布图

## 项目结构

- `main.py`：主入口（训练 + 可视化）
- `src/train_word2vec.py`：训练与保存模型
- `src/visualize_10_words.py`：读取模型并绘制 10 个词向量
- `data/corpus.txt`：示例语料（内置，离线可运行）
- `models/`：训练后模型文件
- `outputs/`：可视化图像输出目录
- `requirements.txt`：pip 依赖
- `environment.yml`：Conda 环境定义

## 1) Conda 环境（Python 3.11）

你已创建好环境 `word2vec`，可跳过创建步骤。

```bash
conda env create -f environment.yml
conda activate word2vec
```

## 2) 安装依赖

```bash
pip install -r requirements.txt
```

## 3) 运行

```bash
python main.py
```

运行完成后：
- 模型输出：`models/word2vec.model`、`models/word2vec.kv`
- 图片输出：`outputs/word_vectors.png`

## 4) 预期效果

程序会在终端输出训练与保存日志，并在 `outputs/word_vectors.png` 里展示 10 个词向量二维分布（PCA）。

## 5) 数据说明

本项目使用 `data/corpus.txt` 的小型语料，包含“王室/职业/食物/动物/交通”等主题词，便于快速演示词向量聚类。

## 6) 复现建议

- 固定 Python 版本为 3.11
- 优先使用 `environment.yml` 创建环境
- 若只用 pip，确保先激活目标 Conda 环境后再执行 `pip install -r requirements.txt`

## 7) GitHub 发布说明

1. 创建远程仓库（例如：`word2vec-demo`）
2. 本地执行：

```bash
git init
git add .
git commit -m "feat: initial word2vec demo"
git branch -M main
git remote add origin <你的仓库URL>
git push -u origin main
```

> 若模型文件后续变大（>50MB），建议改为运行时生成或使用 Git LFS。
