# AutoCLP

---

## Installation



---

## 项目结构

```
autoclp/
├── config/               # 基础配置
├── core/                 # 核心代码
├── data/                 # 导入的对话文件目录
├── exp_result/           # 一些实验结果的存储目录
├── memory/               # faiss数据库的目录
├── rag/                  # RAG的实现
├── raw/                  # 原始数据集对话数据
├── test/                 # 一些测试脚本
├── test_data/            # 一些测试脚本用的数据
├── test_store/           # 测试用目录
├── utils/                # 辅助工具
├── main.py               # 主流程入口
├── requirements.txt
└── README.md
```

---

## Quick Start

```
python main.py --dialogue=data/1.txt
```

将`1.txt`中的对话传入进行处理