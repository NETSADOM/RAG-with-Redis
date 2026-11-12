from threading import Barrier
from unittest.mock import MagicMock
import pytest
from backend.rag.retrieval import HybridRetriever, KBChunk


def chunk(name, doc='allowed', page=1):
    return KBChunk(name, doc, name, 'Delivery', 'Page 1', 'Policy evidence.', page_number=page)


def test_parallel_branches_filter_before_fusion_and_preserve_semantic_only_match():
    excluded = chunk('excluded', 'other')
    a, b, semantic_only = chunk('a'), chunk('b'), chunk('semantic-only')
    barrier = Barrier(2)
    def keyword(query, top_k):
        barrier.wait(timeout=5)
        return [(excluded, 10), (a, 3), (b, 1)]
    def semantic(query, top_k):
        barrier.wait(timeout=5)
        return [(semantic_only, .9), (excluded, .8), (b, .7)]
    bm25, dense = MagicMock(), MagicMock()
    bm25.search.side_effect = keyword
    dense.search.side_effect = semantic
    results = HybridRetriever(bm25, dense).search('paraphrase', metadata_filters={'doc_id': 'allowed'})
    by_id = {item['chunk'].chunk_id: item for item in results}
    assert set(by_id) == {'a', 'b', 'semantic-only'}
    assert results[0]['chunk'].chunk_id == 'b'
    assert by_id['a']['keyword_rank'] == 1
    assert by_id['semantic-only']['semantic_rank'] == 1
    assert 'keyword_rank' not in by_id['semantic-only']
    assert by_id['b']['rrf_score'] == pytest.approx(2 / 62)
    bm25.search.assert_called_once_with('paraphrase', top_k=50)
    dense.search.assert_called_once_with('paraphrase', top_k=50)


def test_metadata_constraints_apply_to_both_branches_and_can_return_empty():
    bm25, dense = MagicMock(), MagicMock()
    bm25.search.return_value = [(chunk('a', page=1), 2)]
    dense.search.return_value = [(chunk('b', page=2), .9)]
    hybrid = HybridRetriever(bm25, dense)
    assert [r['chunk'].chunk_id for r in hybrid.search('delivery', metadata_filters={'page_number': 2})] == ['b']
    assert hybrid.search('delivery', metadata_filters={'doc_id': 'absent'}) == []
    with pytest.raises(ValueError):
        hybrid.search('delivery', metadata_filters={'unknown': 'value'})


def test_keyword_retrieval_handles_obvious_typo_without_rewriting_numbers():
    from backend.rag.retrieval import BM25Retriever, KBChunk
    chunks = [KBChunk("c1", "returns", "Returns", "Window", "Page 1", "Refunds within 30 days."),
              KBChunk("c2", "shipping", "Shipping", "Delivery", "Page 2", "Delivery takes 5 days.")]
    retriever = BM25Retriever(chunks)
    assert retriever.search("refudns")[0][0].chunk_id == "c1"
    assert retriever.search("refudns")[0][1] > 0
    assert "refunds" in retriever._query_tokens("refnds")
    assert retriever._query_tokens("90") == ["90"]
