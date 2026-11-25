// Package config 为 go-lab 各阶段提供共享配置。
//
// 设计原则：复用 Python 版 stage1/.env，不引入 dotenv 依赖——
// 自己解析 KEY=VALUE 只需 30 行，教学上更透明，也避免 Go 侧的依赖膨胀。
package config

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	openaimodel "github.com/cloudwego/eino-ext/components/model/openai"
	openaiemb "github.com/cloudwego/eino-ext/components/embedding/openai"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/embedding"
)

// EnvPath 返回仓库里 Python 版写好的 .env 路径（Go 版复用它，不重复填 Key）。
func EnvPath() string {
	// config.go 位于 go-lab/pkg/config，往上三级是 ai-lab
	_, thisFile, _, _ := runtime.Caller(0)
	dir := filepath.Dir(thisFile)
	return filepath.Clean(filepath.Join(dir, "..", "..", "..", "stage1", ".env"))
}

// LoadEnv 读取 KEY=VALUE 格式的 .env，已存在的环境变量优先（不覆盖）。
func LoadEnv(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("打开 .env 失败 (%s): %w", path, err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key, val := strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
		val = strings.Trim(val, "\"'")
		if _, exists := os.LookupEnv(key); !exists {
			_ = os.Setenv(key, val)
		}
	}
	return scanner.Err()
}

// LoadDefault 加载仓库默认的 stage1/.env，找不到文件时返回错误让调用方决定。
func LoadDefault() error { return LoadEnv(EnvPath()) }

// Get 读取配置，缺失时返回 defaultVal。
func Get(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}

// GetFloat 读取浮点配置。
func GetFloat(key string, defaultVal float32) float32 {
	v := os.Getenv(key)
	if v == "" {
		return defaultVal
	}
	f, err := strconv.ParseFloat(v, 32)
	if err != nil {
		return defaultVal
	}
	return float32(f)
}

// GetInt 读取整数配置。
func GetInt(key string, defaultVal int) int {
	v := os.Getenv(key)
	if v == "" {
		return defaultVal
	}
	i, err := strconv.Atoi(v)
	if err != nil {
		return defaultVal
	}
	return i
}

// APIInfo 汇总当前生效的模型配置（用于 CLI 打印，Key 脱敏）。
type APIInfo struct {
	BaseURL   string
	Model     string
	EmbedModel string
	KeyPrefix string
}

func Info() APIInfo {
	key := Get("DASHSCOPE_API_KEY", "")
	prefix := ""
	if len(key) > 8 {
		prefix = key[:8] + "..."
	} else if key != "" {
		prefix = "***"
	}
	return APIInfo{
		BaseURL:    Get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
		Model:      Get("QWEN_MODEL", "qwen-plus"),
		EmbedModel: Get("QWEN_EMBED_MODEL", "text-embedding-v3"),
		KeyPrefix:  prefix,
	}
}

// NewChatModel 用 Eino 的 OpenAI 兼容组件接千问。
// temperature 传 nil 表示让服务端用默认值；TopP 同理。
func NewChatModel(ctx context.Context, temperature *float32) (model.ChatModel, error) {
	if err := LoadDefault(); err != nil {
		// .env 不存在时仍继续，允许调用方用系统环境变量
		_ = err
	}
	info := Info()
	if Get("DASHSCOPE_API_KEY", "") == "" {
		return nil, fmt.Errorf("未找到 DASHSCOPE_API_KEY，请检查 %s 或系统环境变量", EnvPath())
	}
	cfg := &openaimodel.ChatModelConfig{
		APIKey:      Get("DASHSCOPE_API_KEY", ""),
		BaseURL:     info.BaseURL,
		Model:       info.Model,
		Temperature: temperature,
	}
	return openaimodel.NewChatModel(ctx, cfg)
}

// NewEmbeddings 用 Eino 的 OpenAI 兼容 embedding 组件接千问。
func NewEmbeddings(ctx context.Context) (embedding.Embedder, error) {
	if err := LoadDefault(); err != nil {
		_ = err
	}
	info := Info()
	return openaiemb.NewEmbedder(ctx, &openaiemb.EmbeddingConfig{
		APIKey:  Get("DASHSCOPE_API_KEY", ""),
		BaseURL: info.BaseURL,
		Model:   info.EmbedModel,
	})
}
