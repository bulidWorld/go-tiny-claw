// internal/provider/deepseek.go
package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
	"github.com/yourname/go-tiny-claw/internal/schema"
)

// DeepSeekProvider 实现 LLMProvider 接口，对接 DeepSeek API（兼容 OpenAI 协议）
type DeepSeekProvider struct {
	client          openai.Client
	model           string
	reasoningEffort shared.ReasoningEffort // "high" / "max"
	thinkingEnabled bool                   // 是否在 extra_body 中注入 thinking.type="enabled"
}

// NewDeepSeekProvider 创建 DeepSeek provider
// reasoningEffort: "high" 或 "max"（DeepSeek 仅支持这两个值）
// enableThinking: 是否开启 DeepSeek thinking mode，开启后响应会包含 reasoning_content
func NewDeepSeekProvider(model string, reasoningEffort string, enableThinking bool) *DeepSeekProvider {
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		panic("请设置 DEEPSEEK_API_KEY 环境变量")
	}
	return &DeepSeekProvider{
		client:          openai.NewClient(option.WithAPIKey(apiKey), option.WithBaseURL("https://api.deepseek.com")),
		model:           model,
		reasoningEffort: shared.ReasoningEffort(reasoningEffort),
		thinkingEnabled: enableThinking,
	}
}

func (p *DeepSeekProvider) Generate(ctx context.Context, msgs []schema.Message, availableTools []schema.ToolDefinition) (*schema.Message, error) {
	openaiMsgs := toOpenAIMessages(msgs)
	openaiTools := toOpenAITools(availableTools)

	params := openai.ChatCompletionNewParams{
		Model:    p.model,
		Messages: openaiMsgs,
	}
	if len(openaiTools) > 0 {
		params.Tools = openaiTools
	}

	// DeepSeek 特有参数：推理强度
	if p.reasoningEffort != "" {
		params.ReasoningEffort = p.reasoningEffort
	}

	// DeepSeek 特有参数：thinking mode 通过 extra_body 注入
	if p.thinkingEnabled {
		params.SetExtraFields(map[string]any{
			"thinking": map[string]string{"type": "enabled"},
		})
	}

	resp, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("DeepSeek API 请求失败: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("DeepSeek API 返回了空的 Choices")
	}

	resultMsg := fromOpenAIResponse(resp)

	// 提取 DeepSeek thinking mode 返回的 reasoning_content
	if p.thinkingEnabled {
		choice := resp.Choices[0].Message
		if f, ok := choice.JSON.ExtraFields["reasoning_content"]; ok {
			var reasoning string
			if err := json.Unmarshal([]byte(f.Raw()), &reasoning); err == nil {
				resultMsg.ReasoningContent = reasoning
			}
		}
	}

	return resultMsg, nil
}
