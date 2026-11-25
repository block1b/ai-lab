package stage3

import (
	"context"
	"testing"

	"github.com/cloudwego/eino/components/embedding"
)

// hashEmbed 纯本地字符哈希 embedding（无语义能力），用于离线验证检索链路。
type hashEmbed struct{}

func (hashEmbed) EmbedStrings(_ context.Context, texts []string, _ ...embedding.Option) ([][]float64, error) {
	out := make([][]float64, len(texts))
	for i, t := range texts {
		v := make([]float64, 64)
		for _, r := range t {
			v[int(r)%64]++
		}
		out[i] = v
	}
	return out, nil
}

func TestBM25Recall(t *testing.T) {
	chunks := ChunkSemantic(LoadDocs(), 160)
	bm25 := NewBM25(chunks)
	qs := EvalQuestions()
	hits := 0
	for _, q := range qs {
		got := bm25.Search(q.Question, 5)
		set := map[string]bool{}
		for _, g := range got {
			set[g.Source] = true
		}
		for _, rel := range q.Relevant {
			if set[rel] {
				hits++
				break
			}
		}
	}
	rate := float64(hits) / float64(len(qs)) * 100
	t.Logf("BM25 Recall@5: %d/%d = %.1f%%", hits, len(qs), rate)
	if rate < 70 {
		t.Errorf("BM25 召回过低: %.1f%%", rate)
	}
}

func TestHybridRecall(t *testing.T) {
	ctx := context.Background()
	emb := hashEmbed{}
	chunks := ChunkSemantic(LoadDocs(), 160)
	store, err := BuildStore(ctx, chunks, emb, 10)
	if err != nil {
		t.Fatal(err)
	}
	bm25 := NewBM25(chunks)
	qs := EvalQuestions()
	hits := 0
	for _, q := range qs {
		got, err := HybridRetrieve(ctx, emb, store, bm25, q.Question, 5)
		if err != nil {
			t.Fatal(err)
		}
		set := map[string]bool{}
		for _, g := range got {
			set[g.Source] = true
		}
		for _, rel := range q.Relevant {
			if set[rel] {
				hits++
				break
			}
		}
	}
	rate := float64(hits) / float64(len(qs)) * 100
	t.Logf("混合检索 Recall@5（哈希假向量）: %d/%d = %.1f%%", hits, len(qs), rate)
	if rate < 50 {
		t.Errorf("混合检索召回过低: %.1f%%", rate)
	}
}

func TestChunking(t *testing.T) {
	docs := LoadDocs()
	fixed := ChunkFixed(docs, 120, 20)
	semantic := ChunkSemantic(docs, 160)
	if len(fixed) <= len(semantic) {
		t.Errorf("固定 120 应比语义 160 切得更碎: fixed=%d semantic=%d", len(fixed), len(semantic))
	}
	t.Logf("分块数: 固定120=%d, 语义160=%d", len(fixed), len(semantic))
}
