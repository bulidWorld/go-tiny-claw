package engine

import (
	"context"
	"fmt"
	"log"
	"sync"

	ctxpkg "github.com/yourname/go-tiny-claw/internal/context" // 引入我们新建的 context 包

	"github.com/yourname/go-tiny-claw/internal/provider"
	"github.com/yourname/go-tiny-claw/internal/schema"
	"github.com/yourname/go-tiny-claw/internal/tools"
)

type AgentEngine struct {
	provider       provider.LLMProvider
	registry       tools.Registry
	WorkDir        string
	EnableThinking bool                   // 【新增】慢思考开关
	composer       *ctxpkg.PromptComposer // 【新增】引擎持有 Composer 实例

}

func NewAgentEngine(p provider.LLMProvider, r tools.Registry, workDir string, enableThinking bool) *AgentEngine {
	return &AgentEngine{
		provider:       p,
		registry:       r,
		WorkDir:        workDir,
		EnableThinking: enableThinking,
		composer:       ctxpkg.NewPromptComposer(workDir), // 初始化组装器
	}
}

// internal/engine/loop.go (续)

func (e *AgentEngine) Run(ctx context.Context, session *Session, reporter Reporter) error {
	log.Printf("[Engine] 唤醒会话 [%s]，锁定工作区: %s\n", session.ID, session.WorkDir)

	// 【核心修改】动态组装 System Prompt，彻底替换掉以前硬编码的面条提示词！
	composer := ctxpkg.NewPromptComposer(session.WorkDir)
	systemMsg := composer.Build()

	turnCount := 0

	for {
		turnCount++
		log.Printf("\n========== [Turn %d] 开始 ==========\n", turnCount)

		// 获取当前挂载的所有工具定义
		availableTools := e.registry.GetAvailableTools()

		// 1. 【上下文组装】: System Prompt + 截取最近的 6 条消息作为 Working Memory
		// 在实际业务中，由于工具返回结果可能很长，短期工作记忆往往设为 6-10 条足以维系连贯对话
		workingMemory := session.GetWorkingMemory(6)
		var contextHistory []schema.Message
		contextHistory = append(contextHistory, systemMsg)
		contextHistory = append(contextHistory, workingMemory...)

		// ====================================================================
		// Phase 1: 慢思考阶段 (Thinking) - 剥夺工具，强制规划
		// ====================================================================
		if e.EnableThinking {
			log.Println("[Engine][Phase 1] 剥夺工具访问权，强制进入慢思考与规划阶段...")

			if reporter != nil {
				// 【触发 Reporter】: 开始慢思考
				reporter.OnThinking(ctx)
			}

			// 核心机制：传入的 availableTools 为 nil！
			// 大模型看不到任何 JSON Schema，被迫只能输出纯文本的思考过程。
			thinkResp, err := e.provider.Generate(ctx, contextHistory, nil)
			if err != nil {
				return fmt.Errorf("Thinking 阶段生成失败: %w", err)
			}

			// 如果模型输出了思考过程，我们将其作为 Assistant 消息追加到上下文中
			if thinkResp.Content != "" {
				fmt.Printf("🧠 [内部思考 Trace]: %s\n", thinkResp.Content)
				contextHistory = append(contextHistory, *thinkResp)
			}
		}

		// ====================================================================
		// Phase 2: 行动阶段 (Action) - 恢复工具，顺着规划执行
		// ====================================================================
		log.Println("[Engine][Phase 2] 恢复工具挂载，等待模型采取行动...")

		// 此时的 contextHistory 中已经包含了上一阶段模型自己的 Thinking Trace。
		// 模型会顺着自己的逻辑，结合恢复的 availableTools 发起精准的工具调用。
		actionResp, err := e.provider.Generate(ctx, contextHistory, availableTools)
		if err != nil {
			return fmt.Errorf("Action 阶段生成失败: %w", err)
		}

		contextHistory = append(contextHistory, *actionResp)

		if actionResp.Content != "" && reporter != nil {
			// 【触发 Reporter】: 输出阶段性总结或最终回复
			reporter.OnMessage(ctx, actionResp.Content)
		}

		// ====================================================================
		// 退出与执行逻辑 (与上一讲保持一致)
		// ====================================================================
		if len(actionResp.ToolCalls) == 0 {
			log.Println("[Engine] 模型未请求调用工具，任务宣告完成。")
			break
		}

		log.Printf("[Engine] 模型请求调用 %d 个工具...\n", len(actionResp.ToolCalls))

		// 1. 预分配切片，并发执行所有工具调用
		toolResults := make([]schema.ToolResult, len(actionResp.ToolCalls))

		// 2. 声明 WaitGroup 用于阻塞等待所有协程完成
		var wg sync.WaitGroup
		// 3. 遍历模型请求的所有工具，为每一个工具单独 Fork 出一个 Goroutine
		for i, toolCall := range actionResp.ToolCalls {
			wg.Add(1) // 增加计数器

			// 开启协程。注意：一定要将索引 i 和 toolCall 作为参数传入匿名函数，防止闭包变量捕获陷阱！
			go func(idx int, call schema.ToolCall) {
				defer wg.Done() // 协程结束时计数器减一

				if reporter != nil {
					// 【触发 Reporter】: 报告即将在底层执行的工具
					reporter.OnToolCall(ctx, call.Name, string(call.Arguments))
				}

				log.Printf("  -> [Go-%d] 🛠️ 触发并行执行: %s\n", idx, call.Name)

				// 调用底层 Registry 执行工具（物理操作）
				result := e.registry.Execute(ctx, call)

				if reporter != nil {
					// 为了防止大文件读取导致飞书消息过长被截断，我们仅汇报工具执行状态 // 注意：传递给大模型的 observationMsgs 依然是完整数据，只是人类看到的 Reporter 是缩略版
					displayOutput := result.Output
					if len(displayOutput) > 200 {
						displayOutput = displayOutput[:200] + "... (已截断)"
					}
					// 【触发 Reporter】: 汇报工具物理执行的结果
					reporter.OnToolResult(ctx, call.Name, displayOutput, result.IsError)
				}

				if result.IsError {
					log.Printf("  -> [Go-%d] ❌ 工具执行报错: %s\n", idx, result.Output)
				} else {
					log.Printf("  -> [Go-%d] ✅ 工具执行成功 (返回 %d 字节)\n", idx, len(result.Output))
				}

				// 【线程安全】: 由于每个 Goroutine 操作的是预分配切片的不同索引，
				// 这里不需要加锁 (Mutex)，性能极高！
				toolResults[idx] = schema.ToolResult{
					ToolCallID: call.ID,
					Output:     result.Output,
					IsError:    result.IsError,
				}

			}(i, toolCall) // 闭包传参
		}
		// 4. Join 阻塞等待：主循环挂起，直到所有的并发协程全部执行完毕
		wg.Wait()
		log.Println("[Engine] 所有并发工具执行完毕，开始聚合观察结果 (Observation)...")

		// 5. 关键：将所有 tool_result 合并到同一条 user 消息中！
		// Anthropic/DeepSeek API 要求：同一 assistant 消息中的所有 tool_use，
		// 其对应的 tool_result 必须放在紧接着的下一条 user 消息里，不能拆成多条。
		contextHistory = append(contextHistory, schema.Message{
			Role:        schema.RoleUser,
			ToolResults: toolResults,
		})
		// 循环回到开头，模型将带着这一批新的 Observation 继续它的下一轮思考...
	}

	return nil
}
