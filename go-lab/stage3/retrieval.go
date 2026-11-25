// stage3 环节③：检索——纯向量 / BM25 / 混合 RRF，对标 Python 版 retrieval.py。
package stage3

import (
	"context"
	"math"
	"sort"
	"strings"
	"unicode"

	"github.com/cloudwego/eino/components/embedding"
)

// Scored 带分数的块。
type Scored struct {
	Chunk
	Score float64
}

func topK(in []Scored, k int) []Scored {
	sort.Slice(in, func(i, j int) bool { return in[i].Score > in[j].Score })
	if k > 0 && k < len(in) {
		return in[:k]
	}
	return in
}

// VectorRetrieve 纯向量检索。
func VectorRetrieve(ctx context.Context, emb embedding.Embedder, store *InMemoryStore, query string, k int) ([]Scored, error) {
	vecs, err := emb.EmbedStrings(ctx, []string{query})
	if err != nil || len(vecs) == 0 {
		return nil, err
	}
	return store.Search(vecs[0], k), nil
}

// BM25Retriever 纯 Go BM25（中文单字 + 相邻两字组合，不依赖分词库）。
type BM25Retriever struct {
	docs   []bm25Doc
	idf    map[string]float64
	avgLen float64
	k1     float64
	b      float64
}

type bm25Doc struct {
	chunk Chunk
	terms map[string]int // term -> 词频
	len   int
}

// NewBM25 建索引并预计算 IDF（与 Python 版一致）。
func NewBM25(chunks []Chunk) *BM25Retriever {
	r := &BM25Retriever{k1: 1.5, b: 0.75, idf: map[string]float64{}}
	var total int
	df := map[string]int{}
	for _, c := range chunks {
		terms := bigrams(c.Text)
		l := 0
		for _, n := range terms {
			l += n
		}
		r.docs = append(r.docs, bm25Doc{chunk: c, terms: terms, len: l})
		total += l
		for t := range terms {
			df[t]++
		}
	}
	n := float64(len(chunks))
	for t, freq := range df {
		r.idf[t] = math.Log(1 + (n-float64(freq)+0.5)/(float64(freq)+0.5))
	}
	if len(chunks) > 0 {
		r.avgLen = float64(total) / float64(len(chunks))
	}
	return r
}

// Search 返回 Top-K。
func (r *BM25Retriever) Search(query string, k int) []Scored {
	qterms := bigrams(query)
	scored := make([]Scored, 0, len(r.docs))
	for _, d := range r.docs {
		var score float64
		for q := range qterms {
			f := d.terms[q]
			if f == 0 {
				continue
			}
			denom := float64(f) + r.k1*(1-r.b+r.b*float64(d.len)/maxFloat(r.avgLen, 1))
			score += r.idf[q] * float64(f) * (r.k1 + 1) / denom
		}
		if score > 0 {
			scored = append(scored, Scored{Chunk: d.chunk, Score: score})
		}
	}
	return topK(scored, k)
}

// HybridRetrieve 混合检索：向量与 BM25 的结果用 RRF（倒数排序融合）合并。
func HybridRetrieve(ctx context.Context, emb embedding.Embedder, store *InMemoryStore, bm25 *BM25Retriever, query string, k int) ([]Scored, error) {
	vvecs, err := emb.EmbedStrings(ctx, []string{query})
	if err != nil || len(vvecs) == 0 {
		return nil, err
	}
	vecResults := store.Search(vvecs[0], k*3)
	bmResults := bm25.Search(query, k*3)

	const rrfK = 60.0
	fused := map[string]Scored{}
	add := func(list []Scored) {
		for rank, item := range list {
			score := 1.0 / (rrfK + float64(rank+1))
			if prev, ok := fused[item.ID]; ok {
				prev.Score += score
				fused[item.ID] = prev
			} else {
				item.Score = score
				fused[item.ID] = item
			}
		}
	}
	add(vecResults)
	add(bmResults)
	out := make([]Scored, 0, len(fused))
	for _, v := range fused {
		out = append(out, v)
	}
	return topK(out, k), nil
}

// bigrams 中文按「单字 + 相邻两字组合」切分，英文/数字按连续串。
// 与 Python 版 _bigrams 完全一致：中文"防水等级"→"防","水","等","级","防水","水等","等级"。
func bigrams(text string) map[string]int {
	out := map[string]int{}
	tokens := []string{}
	var buf []rune
	flush := func() {
		if len(buf) > 0 {
			tokens = append(tokens, string(buf))
			buf = nil
		}
	}
	for _, r := range strings.ToLower(strings.TrimSpace(text)) {
		isCJK := r >= '\u4e00' && r <= '\u9fff'
		switch {
		case isCJK:
			flush()
			tokens = append(tokens, string(r))
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			buf = append(buf, r)
		default:
			flush()
		}
	}
	flush()
	for i, t := range tokens {
		out[t]++
		if isCJKString(t) && i+1 < len(tokens) && isCJKString(tokens[i+1]) {
			out[t+tokens[i+1]]++
		}
	}
	return out
}

func isCJKString(s string) bool {
	r := []rune(s)
	return len(r) == 1 && r[0] >= '\u4e00' && r[0] <= '\u9fff'
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
