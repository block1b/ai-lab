// stage3 CLI + 评估，对标 Python 版 cli.py / evaluate.py。
//
//   go run ./stage3/cmd "手环的防水等级是多少？"          # 问答（需 API）
//   go run ./stage3/cmd --recall                          # Recall@5 评估（需 API）
//   go run ./stage3/cmd --chunk                           # 分块对比（零成本）
package stage3

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"golab/pkg/config"
)

var (
	flagRecall = flag.Bool("recall", false, "跑 Recall@5 评估")
	flagChunk  = flag.Bool("chunk", false, "分块对比，零成本")
	flagMode   = flag.String("mode", "hybrid", "检索模式：vector / bm25 / hybrid")
	flagRerank = flag.Bool("rerank", false, "启用 LLM 重排")
)

// Main 是 stage3 的 CLI 入口。
func Main() {
	flag.Parse()
	if *flagChunk {
		chunkCompare()
		return
	}
	if *flagRecall {
		recallEval()
		return
	}
	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "用法: go run ./stage3/cmd \"问题\" | --recall | --chunk")
		os.Exit(1)
	}
	answer(strings.Join(args, " "))
}

func chunkCompare() {
	docs := LoadDocs()
	fmt.Println("=== 分块对比（零成本）===")
	for _, cfg := range []struct {
		name string
		fn   func() []Chunk
	}{
		{"固定 120/20", func() []Chunk { return ChunkFixed(docs, 120, 20) }},
		{"固定 200/40", func() []Chunk { return ChunkFixed(docs, 200, 40) }},
		{"语义 160", func() []Chunk { return ChunkSemantic(docs, 160) }},
	} {
		chunks := cfg.fn()
		minLen, maxLen, total := 1<<30, 0, 0
		for _, c := range chunks {
			l := len([]rune(c.Text))
			if l < minLen {
				minLen = l
			}
			if l > maxLen {
				maxLen = l
			}
			total += l
		}
		fmt.Printf("%-12s: %d 块 | 平均 %d 字 | 最短 %d | 最长 %d\n",
			cfg.name, len(chunks), total/len(chunks), minLen, maxLen)
	}
	fmt.Println("\n提示：固定 200 接近语料长度，多数文档一块装下；固定 120 才真正切成多块。")
}

func recallEval() {
	ctx := context.Background()
	emb, err := config.NewEmbeddings(ctx)
	if err != nil {
		fmt.Fprintln(os.Stderr, "初始化 embedding 失败:", err)
		return
	}
	chunks := ChunkSemantic(LoadDocs(), 160)
	store, err := BuildStore(ctx, chunks, emb, 10)
	if err != nil {
		fmt.Fprintln(os.Stderr, "建库失败:", err)
		return
	}
	bm25 := NewBM25(chunks)
	qs := EvalQuestions()

	for _, mode := range []string{"vector", "bm25", "hybrid"} {
		hits := 0
		for _, q := range qs {
			var got []Scored
			switch mode {
			case "vector":
				got, _ = VectorRetrieve(ctx, emb, store, q.Question, 5)
			case "bm25":
				got = bm25.Search(q.Question, 5)
			case "hybrid":
				got, _ = HybridRetrieve(ctx, emb, store, bm25, q.Question, 5)
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
		fmt.Printf("%s Recall@5: %d/%d = %.1f%%\n", mode, hits, len(qs), float64(hits)/float64(len(qs))*100)
	}
}

func answer(question string) {
	ctx := context.Background()
	emb, err := config.NewEmbeddings(ctx)
	if err != nil {
		fmt.Fprintln(os.Stderr, "初始化 embedding 失败:", err)
		return
	}
	chunks := ChunkSemantic(LoadDocs(), 160)
	store, err := BuildStore(ctx, chunks, emb, 10)
	if err != nil {
		fmt.Fprintln(os.Stderr, "建库失败:", err)
		return
	}
	bm25 := NewBM25(chunks)

	var candidates []Scored
	switch *flagMode {
	case "vector":
		candidates, err = VectorRetrieve(ctx, emb, store, question, 6)
	case "bm25":
		candidates = bm25.Search(question, 6)
	default:
		candidates, err = HybridRetrieve(ctx, emb, store, bm25, question, 6)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "检索失败:", err)
		return
	}
	fmt.Println("检索 Top-6:")
	for i, c := range candidates {
		fmt.Printf("  %d. [%s] score=%.4f %s\n", i+1, c.Source, c.Score, trunc(c.Text, 40))
	}

	if *flagRerank {
		candidates, err = Rerank(ctx, question, candidates, 3)
		if err != nil {
			fmt.Fprintln(os.Stderr, "重排失败:", err)
			return
		}
		fmt.Println("重排 Top-3:")
		for i, c := range candidates {
			fmt.Printf("  %d. [%s] score=%.1f\n", i+1, c.Source, c.Score)
		}
	}
	if len(candidates) > 3 {
		candidates = candidates[:3]
	}
	ans, err := Answer(ctx, question, candidates)
	if err != nil {
		fmt.Fprintln(os.Stderr, "生成失败:", err)
		return
	}
	fmt.Println("\n答案:", ans)
	fmt.Printf("引用数=%d 拒答=%v\n", CitationCount(ans), IsRefusal(ans))
}

func trunc(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "..."
}
