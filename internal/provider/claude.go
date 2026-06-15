// internal/provider/claude.go
package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/yourname/go-tiny-claw/internal/schema"
)

type ClaudeProvider struct {
	client anthropic.Client
	model  string
	effort string
}

const defaultConfigFile = "claw.config.json"

type clawConfig struct {
	DeepSeekClaude deepSeekClaudeConfig `json:"deepseek_claude"`
}

type deepSeekClaudeConfig struct {
	AnthropicBaseURL            string `json:"anthropic_base_url"`
	AnthropicAuthToken          string `json:"anthropic_auth_token"`
	AnthropicModel              string `json:"anthropic_model"`
	AnthropicDefaultOpusModel   string `json:"anthropic_default_opus_model"`
	AnthropicDefaultSonnetModel string `json:"anthropic_default_sonnet_model"`
	AnthropicDefaultHaikuModel  string `json:"anthropic_default_haiku_model"`
	ClaudeCodeSubagentModel     string `json:"claude_code_subagent_model"`
	ClaudeCodeEffortLevel       string `json:"claude_code_effort_level"`
}

func NewZhipuClaudeProvider(model string) *ClaudeProvider {
	apiKey := os.Getenv("ZHIPU_API_KEY")
	if apiKey == "" {
		panic("请设置 ZHIPU_API_KEY 环境变量")
	}
	baseURL := "https://open.bigmodel.cn/api/paas/v4/"
	return &ClaudeProvider{
		client: anthropic.NewClient(option.WithAPIKey(apiKey), option.WithBaseURL(baseURL)),
		model:  model,
	}
}

// NewDeepSeekClaudeProvider 创建 Anthropic 协议兼容的 DeepSeek provider。
//
// 优先读取 claw.config.json，也可用 CLAW_CONFIG 指定配置文件路径。
// 配置缺失时回退读取环境变量：
//   - ANTHROPIC_BASE_URL，默认 https://api.deepseek.com/anthropic
//   - ANTHROPIC_AUTH_TOKEN，DeepSeek API Key
//   - ANTHROPIC_MODEL，model 参数为空时使用
//   - CLAUDE_CODE_EFFORT_LEVEL，可选：low / medium / high / xhigh / max
func NewDeepSeekClaudeProvider(model string) *ClaudeProvider {
	cfg := loadDeepSeekClaudeConfig()

	authToken := firstNonEmpty(cfg.AnthropicAuthToken, os.Getenv("ANTHROPIC_AUTH_TOKEN"))
	if authToken == "" {
		panic("请在 claw.config.json 中配置 anthropic_auth_token，或设置 ANTHROPIC_AUTH_TOKEN 环境变量")
	}

	if model == "" {
		model = firstNonEmpty(
			cfg.AnthropicModel,
			cfg.AnthropicDefaultSonnetModel,
			cfg.AnthropicDefaultOpusModel,
			cfg.AnthropicDefaultHaikuModel,
			os.Getenv("ANTHROPIC_MODEL"),
			os.Getenv("ANTHROPIC_DEFAULT_SONNET_MODEL"),
			os.Getenv("ANTHROPIC_DEFAULT_OPUS_MODEL"),
			os.Getenv("ANTHROPIC_DEFAULT_HAIKU_MODEL"),
		)
	}
	if model == "" {
		model = "deepseek-v4-pro[1m]"
	}

	baseURL := firstNonEmpty(cfg.AnthropicBaseURL, os.Getenv("ANTHROPIC_BASE_URL"))
	if baseURL == "" {
		baseURL = "https://api.deepseek.com/anthropic"
	}

	return &ClaudeProvider{
		client: anthropic.NewClient(option.WithAuthToken(authToken), option.WithBaseURL(baseURL)),
		model:  model,
		effort: firstNonEmpty(cfg.ClaudeCodeEffortLevel, os.Getenv("CLAUDE_CODE_EFFORT_LEVEL")),
	}
}

func (p *ClaudeProvider) Generate(ctx context.Context, msgs []schema.Message, availableTools []schema.ToolDefinition) (*schema.Message, error) {
	var anthropicMsgs []anthropic.MessageParam
	var systemPrompt string

	// 1. 消息翻译
	for _, msg := range msgs {
		switch msg.Role {
		case schema.RoleSystem:
			systemPrompt = msg.Content
		case schema.RoleUser:
			if len(msg.ToolResults) > 0 {
				// 关键：多个 tool_result 必须放在同一条 user 消息里
				blocks := make([]anthropic.ContentBlockParamUnion, 0, len(msg.ToolResults))
				for _, tr := range msg.ToolResults {
					blocks = append(blocks, anthropic.NewToolResultBlock(tr.ToolCallID, tr.Output, tr.IsError))
				}
				anthropicMsgs = append(anthropicMsgs, anthropic.NewUserMessage(blocks...))
			} else if msg.ToolCallID != "" {
				anthropicMsgs = append(anthropicMsgs, anthropic.NewUserMessage(
					anthropic.NewToolResultBlock(msg.ToolCallID, msg.Content, false),
				))
			} else {
				anthropicMsgs = append(anthropicMsgs, anthropic.NewUserMessage(
					anthropic.NewTextBlock(msg.Content),
				))
			}
		case schema.RoleAssistant:
			var blocks []anthropic.ContentBlockParamUnion
			if msg.Content != "" {
				blocks = append(blocks, anthropic.NewTextBlock(msg.Content))
			}

			// 将历史工具调用转回 Claude 特有的 ToolUseBlockParam
			for _, tc := range msg.ToolCalls {
				var inputMap map[string]interface{}
				_ = json.Unmarshal(tc.Arguments, &inputMap)
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfToolUse: &anthropic.ToolUseBlockParam{
						ID:    tc.ID,
						Name:  tc.Name,
						Input: inputMap,
					},
				})
			}
			if len(blocks) > 0 {
				anthropicMsgs = append(anthropicMsgs, anthropic.NewAssistantMessage(blocks...))
			}
		}
	}
	// 打印anthropicMsgs日志
	// if data, err := json.MarshalIndent(anthropicMsgs, "", "  "); err == nil {
	// 	fmt.Printf("[claude provider] anthropicMsgs: %s\n", string(data))
	// }

	// 2. 工具 Schema 翻译
	var anthropicTools []anthropic.ToolUnionParam
	for _, toolDef := range availableTools {
		// ToolInputSchemaParam 是结构体，需要通过 Properties 字段精准填充
		var properties map[string]any
		var required []string

		if m, ok := toolDef.InputSchema.(map[string]interface{}); ok {
			if p, ok := m["properties"].(map[string]interface{}); ok {
				properties = p
			}
			if r, ok := m["required"].([]string); ok {
				required = r
			}
		}

		tp := anthropic.ToolParam{
			Name:        toolDef.Name,
			Description: anthropic.String(toolDef.Description),
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: properties,
				Required:   required,
			},
		}
		anthropicTools = append(anthropicTools, anthropic.ToolUnionParam{OfTool: &tp})
	}

	// 3. 构建请求并发送
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(p.model),
		MaxTokens: 4096,
		Messages:  anthropicMsgs,
	}

	if p.effort != "" {
		params.OutputConfig = anthropic.OutputConfigParam{
			Effort: anthropic.OutputConfigEffort(p.effort),
		}
	}

	if systemPrompt != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: systemPrompt},
		}
	}

	if len(anthropicTools) > 0 {
		params.Tools = anthropicTools
	}

	resp, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("Claude API 请求失败: %w", err)
	}

	// 4. 反向解析
	resultMsg := &schema.Message{
		Role: schema.RoleAssistant,
	}

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			fmt.Printf("[claude provider]===============: return text result \n")
			resultMsg.Content += block.Text
		case "tool_use":
			fmt.Printf("[claude provider]===============: return tool use \n")
			argsBytes, _ := json.Marshal(block.Input)
			resultMsg.ToolCalls = append(resultMsg.ToolCalls, schema.ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: argsBytes,
			})
		}
	}

	return resultMsg, nil
}

func loadDeepSeekClaudeConfig() deepSeekClaudeConfig {
	configPath := strings.TrimSpace(os.Getenv("CLAW_CONFIG"))
	if configPath == "" {
		var ok bool
		configPath, ok = findConfigFile(defaultConfigFile)
		if !ok {
			return deepSeekClaudeConfig{}
		}
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return deepSeekClaudeConfig{}
		}
		panic(fmt.Sprintf("读取配置文件 %s 失败: %v", configPath, err))
	}

	var cfg clawConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		panic(fmt.Sprintf("解析配置文件 %s 失败: %v", configPath, err))
	}

	return cfg.DeepSeekClaude
}

func findConfigFile(name string) (string, bool) {
	dir, err := os.Getwd()
	if err != nil {
		return "", false
	}

	for {
		candidate := filepath.Join(dir, name)
		if _, err := os.Stat(candidate); err == nil {
			return candidate, true
		} else if !errors.Is(err, os.ErrNotExist) {
			return candidate, true
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	return "", false
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			return value
		}
	}
	return ""
}
