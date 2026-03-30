"""
verify_week3.py — Chạy tất cả 8 test blocks để xác nhận code tuần 3 hoạt động đúng.

Usage (từ project root):
    python code/week3/verify_week3.py
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'week1'))

PASS = 0
FAIL = 0


def run_test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"✅ {name} PASSED")
        PASS += 1
    except Exception as e:
        print(f"❌ {name} FAILED: {e}")
        FAIL += 1


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Imports và constants
# ──────────────────────────────────────────────────────────────────────────────
def test1():
    from utils.constants import ASPECT_COLUMNS, RARE_ASPECTS, LABEL_TO_IDX, IDX_TO_LABEL
    assert len(ASPECT_COLUMNS) == 34, f"Cần 34 aspects, có {len(ASPECT_COLUMNS)}"
    assert set(RARE_ASPECTS).issubset(set(ASPECT_COLUMNS)), "RARE_ASPECTS phải nằm trong ASPECT_COLUMNS"
    assert LABEL_TO_IDX == {"absent": 0, "positive": 1, "negative": 2, "neutral": 3}
    assert IDX_TO_LABEL == {0: "absent", 1: "positive", 2: "negative", 3: "neutral"}

run_test("Test 1: Imports & constants", test1)


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: parse_llm_output
# ──────────────────────────────────────────────────────────────────────────────
def test2():
    from prompts import parse_llm_output

    assert parse_llm_output('{"SERVICE#GENERAL": "positive"}') == {"SERVICE#GENERAL": "positive"}
    assert parse_llm_output('```json\n{"HOTEL#GENERAL": "negative"}\n```') == {"HOTEL#GENERAL": "negative"}
    assert parse_llm_output('{}') == {}
    assert parse_llm_output('Tôi không hiểu yêu cầu.') == {}

    result = parse_llm_output('{"INVALID#ASPECT": "positive", "SERVICE#GENERAL": "positive"}')
    assert result == {"SERVICE#GENERAL": "positive"}, f"Phải filter invalid aspect, got: {result}"

    result = parse_llm_output('{"SERVICE#GENERAL": "very_positive"}')
    assert result == {}, f"Phải filter invalid sentiment, got: {result}"

    cot_output = 'Phân tích:\n- SERVICE#GENERAL: tốt → positive\nOutput: {"SERVICE#GENERAL": "positive"}'
    assert parse_llm_output(cot_output) == {"SERVICE#GENERAL": "positive"}

    # Case 8: Nested JSON safeguard
    nested = '{"analysis": "tốt", "result": {"SERVICE#GENERAL": "positive"}}'
    result = parse_llm_output(nested)
    assert result == {"SERVICE#GENERAL": "positive"}, f"Nested JSON failed: {result}"

run_test("Test 2: parse_llm_output", test2)


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: labels_dict_to_array
# ──────────────────────────────────────────────────────────────────────────────
def test3():
    from prompts import labels_dict_to_array
    from utils.constants import ASPECT_COLUMNS

    arr = labels_dict_to_array({})
    assert arr == [0] * 34, "Dict rỗng phải cho array toàn 0"

    asp = ASPECT_COLUMNS[0]
    arr = labels_dict_to_array({asp: "positive"})
    assert arr[0] == 1, f"positive phải map sang 1, got {arr[0]}"
    assert sum(arr) == 1, "Chỉ có 1 aspect được set"

    last_asp = ASPECT_COLUMNS[-1]
    arr = labels_dict_to_array({last_asp: "negative"})
    assert arr[-1] == 2, f"negative phải map sang 2, got {arr[-1]}"

run_test("Test 3: labels_dict_to_array", test3)


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: build_prompt structure
# ──────────────────────────────────────────────────────────────────────────────
def test4():
    from prompts import build_prompt

    examples = [
        {"review": "Phòng sạch, view đẹp", "labels": {"ROOMS#CLEANLINESS": "positive"}},
        {"review": "Giá hơi cao", "labels": {"ROOMS#PRICES": "negative"}},
    ]
    messages = build_prompt("Khách sạn ổn nhưng service chậm", examples)

    assert messages[0]["role"] == "system", "Message đầu phải là system"
    assert messages[-1]["role"] == "user", "Message cuối phải là user"

    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant_msgs) == len(examples), \
        f"Phải có {len(examples)} assistant messages, có {len(assistant_msgs)}"

run_test("Test 4: build_prompt structure", test4)


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: LLMClient cache
# ──────────────────────────────────────────────────────────────────────────────
def test5():
    import tempfile
    from llm_client import LLMClient

    class MockClient(LLMClient):
        def __init__(self):
            self.cache_dir = tempfile.mkdtemp()
            self.provider = "openai"
            self.model = "gpt-4o-mini"
            self.max_retries = 3
            self.retry_delay = 2.0
            self.total_tokens = 0
            os.makedirs(self.cache_dir, exist_ok=True)

    mock = MockClient()
    msgs1 = [{"role": "user", "content": "Hello"}, {"role": "system", "content": "Be helpful"}]
    msgs2 = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Hello"}]

    key1 = mock._cache_key(msgs1)
    key2 = mock._cache_key(msgs2)
    assert key1 != key2, "Messages với thứ tự khác phải có cache key khác"

    mock._save_cache("testkey123", "test response")
    loaded = mock._load_cache("testkey123")
    assert loaded == "test response", f"Cache load/save thất bại: {loaded}"

run_test("Test 5: LLMClient cache", test5)


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: ABSARetriever
# ──────────────────────────────────────────────────────────────────────────────
def test6():
    try:
        import sentence_transformers  # noqa
    except ImportError:
        print("  [SKIP] sentence_transformers không có — cài bằng: pip install sentence-transformers")
        return
    import tempfile
    import pandas as pd
    import numpy as np
    from rag_retriever import ABSARetriever
    from utils.constants import ASPECT_COLUMNS

    n = 20
    dummy_data = {"processed_review": [f"review số {i}" for i in range(n)]}
    for asp in ASPECT_COLUMNS:
        dummy_data[asp] = [0] * n
    dummy_data[ASPECT_COLUMNS[0]][0] = 1
    dummy_data[ASPECT_COLUMNS[1]][1] = 2
    dummy_data[ASPECT_COLUMNS[2]][2] = 3

    train_df = pd.DataFrame(dummy_data)
    cache_path = os.path.join(tempfile.mkdtemp(), "test_cache.npy")

    retriever = ABSARetriever(cache_path=cache_path)
    retriever.fit(train_df)

    indices = retriever.retrieve("phòng sạch đẹp", k=3)
    assert len(indices) == 3, f"retrieve(k=3) phải trả về 3 indices, got {len(indices)}"
    assert all(0 <= i < n for i in indices), "Indices phải nằm trong range train_df"
    assert len(set(indices)) == 3, "Không được trùng lặp indices"
    assert os.path.exists(cache_path), "Cache file phải được tạo sau fit()"

    retriever2 = ABSARetriever(cache_path=cache_path)
    retriever2.fit(train_df)
    assert retriever2.train_embeds is not None

run_test("Test 6: ABSARetriever", test6)


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: df_to_examples
# ──────────────────────────────────────────────────────────────────────────────
def test7():
    import pandas as pd
    from icl_predictor import df_to_examples
    from utils.constants import ASPECT_COLUMNS

    dummy_data = {"processed_review": ["review A", "review B"]}
    for asp in ASPECT_COLUMNS:
        dummy_data[asp] = [0, 0]
    dummy_data[ASPECT_COLUMNS[0]][0] = 1
    dummy_data[ASPECT_COLUMNS[1]][0] = 2

    train_df = pd.DataFrame(dummy_data)
    examples = df_to_examples(train_df, [0, 1])

    assert len(examples[0]["labels"]) == 2, \
        f"Row 0 phải có 2 aspects, got {len(examples[0]['labels'])}"
    assert examples[0]["labels"][ASPECT_COLUMNS[0]] == "positive"
    assert examples[0]["labels"][ASPECT_COLUMNS[1]] == "negative"
    assert examples[1]["labels"] == {}, \
        f"Row 1 phải có labels rỗng, got {examples[1]['labels']}"

run_test("Test 7: df_to_examples", test7)


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: End-to-end mock
# ──────────────────────────────────────────────────────────────────────────────
def test8():
    from unittest.mock import MagicMock
    import pandas as pd
    import numpy as np
    from icl_predictor import predict_icl
    from llm_client import LLMClient
    from utils.constants import ASPECT_COLUMNS

    mock_response = '{"SERVICE#GENERAL": "positive"}'

    n_test, n_train = 5, 20
    test_data = {"processed_review": [f"test {i}" for i in range(n_test)]}
    train_data = {"processed_review": [f"train {i}" for i in range(n_train)]}
    for asp in ASPECT_COLUMNS:
        test_data[asp] = [0] * n_test
        train_data[asp] = [0] * n_train
    test_data[ASPECT_COLUMNS[-1]] = [1] * n_test  # SERVICE#GENERAL positive

    test_df = pd.DataFrame(test_data)
    train_df = pd.DataFrame(train_data)

    mock_client = MagicMock(spec=LLMClient)
    mock_client.complete.return_value = mock_response

    y_true, y_pred = predict_icl(
        test_df=test_df,
        train_df=train_df,
        client=mock_client,
        k=2,
        max_samples=5,
    )

    assert y_true.shape == (5, 34), f"y_true shape sai: {y_true.shape}"
    assert y_pred.shape == (5, 34), f"y_pred shape sai: {y_pred.shape}"
    assert mock_client.complete.call_count == 5, \
        f"complete() phải được gọi 5 lần, gọi {mock_client.complete.call_count}"

run_test("Test 8: End-to-end mock (no API)", test8)


# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*50}")
print(f"Kết quả: {PASS}/{PASS+FAIL} tests PASSED")
if FAIL:
    print(f"❌ {FAIL} test(s) FAILED — xem lỗi ở trên")
    sys.exit(1)
else:
    print("✅ Tất cả tests PASSED — code tuần 3 sẵn sàng chạy!")
