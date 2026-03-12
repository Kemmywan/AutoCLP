# test_rag.py
# 逻辑：从测试数据文件读取三元组，冷启动构建RAG，并验证检索效果

import json
import os
from rag import rag_core

# ────────────────────────────────────────
# 配置项
# ────────────────────────────────────────
TEST_DATA_PATH = 'test_data/test_triples.jsonl'
TEST_STORE_INDEX = 'test_store/vectors.faiss'
TEST_STORE_TRIPLE = 'test_store/triples.jsonl'

# 测试用query（模拟真实诊断输入）
TEST_QUERIES = [
    {
        'query': '患者胸痛，心电图异常，肌钙蛋白升高',
        'expected_keyword': 'STEMI'
    },
    {
        'query': '女性患者咳嗽发热，CT显示肺部磨玻璃影',
        'expected_keyword': '肺炎'
    },
    {
        'query': '老年男性血糖极高，糖尿病史，视力模糊',
        'expected_keyword': '糖尿病'
    },
]


# ────────────────────────────────────────
# Step 1: 读取测试三元组数据文件
# ────────────────────────────────────────
def load_test_triples(filepath: str) -> list:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'测试数据文件不存在：{filepath}，主人请先建好数据文件(╬￣皿￣)')
    triples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                triples.append(json.loads(line))
    print(f'[数据加载] 成功读取 {len(triples)} 条三元组测试数据。')
    return triples


# ────────────────────────────────────────
# Step 2: 初始化RAG并批量导入测试数据
# ────────────────────────────────────────
def build_test_rag(triples: list) -> rag_core.VectorMemory:
    # 使用独立的测试路径，防止污染主库（别把测试数据搞进生产库= =）
    store = rag_core.VectorMemory(
        index_file=TEST_STORE_INDEX,
        triple_file=TEST_STORE_TRIPLE
    )
    store.batch_import(triples)
    store.save_index()
    print(f'[RAG构建] {len(triples)} 条三元组已成功导入向量数据库并持久化。')
    return store


# ────────────────────────────────────────
# Step 3: 检索验证
# ────────────────────────────────────────
def test_search(store: rag_core.VectorMemory, queries: list):
    print('\n' + '=' * 40)
    print('[检索验证] 开始测试RAG检索效果...')
    print('=' * 40)

    pass_count = 0
    fail_count = 0

    for i, test_case in enumerate(queries):
        query = test_case['query']
        expected_keyword = test_case['expected_keyword']

        results = store.search(query, top_k=3)

        print(f'\n[Query {i+1}] {query}')
        print(f'[期望关键词] {expected_keyword}')
        print(f'[召回结果]')

        hit = False
        for j, r in enumerate(results):
            diagnostic = r.get('diagnostic', '')
            print(f'  Top{j+1}: {diagnostic[:60]}...' if len(diagnostic) > 60 else f'  Top{j+1}: {diagnostic}')
            if expected_keyword in diagnostic:
                hit = True

        if hit:
            print(f'  ✓ PASS — 关键词命中')
            pass_count += 1
        else:
            print(f'  ✗ FAIL — 未检索到期望关键词，可能embedding效果不足或数据不够覆盖')
            fail_count += 1

    print('\n' + '=' * 40)
    print(f'[测试结果] PASS: {pass_count} / {pass_count + fail_count}，FAIL: {fail_count}')
    if fail_count == 0:
        print('[结论] 全部通过，RAG系统检索逻辑正常。(•̀ω•́)ง')
    else:
        print('[结论] 存在失败项，建议检查embedding模型或扩充测试数据。(ˉ▽ˉ；)')
    print('=' * 40)


# ────────────────────────────────────────
# 主流程入口
# ────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 40)
    print('[test_rag.py] RAG冷启动构建与检索验证')
    print('=' * 40)

    # 读取数据
    triples = load_test_triples(TEST_DATA_PATH)

    # 构建RAG
    store = build_test_rag(triples)

    # 检索验证
    test_search(store, TEST_QUERIES)
