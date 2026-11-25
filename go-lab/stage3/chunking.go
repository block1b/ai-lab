// stage3 环节①：分块，对标 Python 版 chunking.py。
package stage3

import "strings"

// Chunk 一个文本块，保留来源用于 Recall 评估和引用。
type Chunk struct {
	ID     string
	Source string
	Text   string
}

// ChunkFixed 固定长度分块（按 rune 切，避免中文半字）。
// overlap 只在切第二刀时回退，第一个块不会被截短。
func ChunkFixed(docs []Doc, size, overlap int) []Chunk {
	if size <= 0 {
		size = 200
	}
	if overlap < 0 || overlap >= size {
		overlap = size / 5
	}
	var out []Chunk
	for _, d := range docs {
		runes := []rune(d.Content)
		start := 0
		idx := 0
		for start < len(runes) {
			end := start + size
			if end > len(runes) {
				end = len(runes)
			}
			out = append(out, Chunk{
				ID:     d.Source + "#" + itoa(idx),
				Source: d.Source,
				Text:   string(runes[start:end]),
			})
			idx++
			if end == len(runes) {
				break
			}
			start = end - overlap
		}
	}
	return out
}

// ChunkSemantic 语义分块：按句子边界聚合，尽量接近 target 长度而不切断句子。
func ChunkSemantic(docs []Doc, target int) []Chunk {
	if target <= 0 {
		target = 160
	}
	var out []Chunk
	for _, d := range docs {
		sents := splitSentences(d.Content)
		var buf strings.Builder
		idx := 0
		for _, s := range sents {
			cur := len([]rune(buf.String()))
			if cur > 0 && cur+len([]rune(s)) > target {
				out = append(out, Chunk{
					ID: d.Source + "#" + itoa(idx), Source: d.Source,
					Text: strings.TrimSpace(buf.String()),
				})
				idx++
				buf.Reset()
			}
			buf.WriteString(s)
		}
		if strings.TrimSpace(buf.String()) != "" {
			out = append(out, Chunk{
				ID: d.Source + "#" + itoa(idx), Source: d.Source,
				Text: strings.TrimSpace(buf.String()),
			})
		}
	}
	return out
}

// splitSentences 按中文句末标点切句，保留标点。
func splitSentences(text string) []string {
	var out []string
	var buf strings.Builder
	for _, r := range text {
		buf.WriteRune(r)
		switch r {
		case '。', '！', '？', '；', '!', '?', ';':
			s := strings.TrimSpace(buf.String())
			if s != "" {
				out = append(out, s)
			}
			buf.Reset()
		}
	}
	if strings.TrimSpace(buf.String()) != "" {
		out = append(out, strings.TrimSpace(buf.String()))
	}
	return out
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	pos := len(b)
	for i > 0 {
		pos--
		b[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(b[pos:])
}
