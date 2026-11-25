// stage3 环节②：内存向量库，纯 Go 实现余弦相似度。
//
// 为什么自己写：Eino 没有内置内存 VectorStore（后端都在 eino-ext，需要 chroma/qdrant）。
// 自己实现只需 40 行，余弦相似度透明可见，教学价值更高，也零额外依赖。
package stage3

import (
	"context"
	"math"

	"github.com/cloudwego/eino/components/embedding"
)

// VecChunk 带向量的块。
type VecChunk struct {
	Chunk
	Vector []float64
}

// InMemoryStore 内存向量库。
type InMemoryStore struct {
	Items []VecChunk
	Dim   int
}

// BuildStore 用 Eino Embedder 把块向量化并建库。
// 注意：百炼 embedding 单次批量 ≤10 条，这里自动分批（Python 版踩过的坑）。
func BuildStore(ctx context.Context, chunks []Chunk, emb embedding.Embedder, batchSize int) (*InMemoryStore, error) {
	if batchSize <= 0 || batchSize > 10 {
		batchSize = 10
	}
	store := &InMemoryStore{}
	for i := 0; i < len(chunks); i += batchSize {
		end := i + batchSize
		if end > len(chunks) {
			end = len(chunks)
		}
		texts := make([]string, 0, end-i)
		for _, c := range chunks[i:end] {
			texts = append(texts, c.Text)
		}
		vecs, err := emb.EmbedStrings(ctx, texts)
		if err != nil {
			return nil, err
		}
		for j, v := range vecs {
			if store.Dim == 0 {
				store.Dim = len(v)
			}
			store.Items = append(store.Items, VecChunk{Chunk: chunks[i+j], Vector: v})
		}
	}
	return store, nil
}

// Search 按余弦相似度返回 Top-K。
func (s *InMemoryStore) Search(queryVec []float64, k int) []Scored {
	scored := make([]Scored, 0, len(s.Items))
	for _, item := range s.Items {
		scored = append(scored, Scored{Chunk: item.Chunk, Score: cosine(queryVec, item.Vector)})
	}
	return topK(scored, k)
}

func cosine(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var dot, na, nb float64
	for i := 0; i < n; i++ {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
