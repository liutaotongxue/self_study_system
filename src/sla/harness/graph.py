"""LangGraph harness:把 Phase 4 的 while 循环用 StateGraph 重写。

逻辑跟 Phase 4 完全一致,只是表达方式换了:
  Phase 4 while 循环               LangGraph
  -----------------------------    ------------------------------
  while True:                      START → agent (节点)
      response = bound.invoke(...)   agent_node 调 bound.invoke
      messages.append(response)      add_messages reducer 自动追加
      if not response.tool_calls:    conditional_edges → END
          break
      for tc in response.tool_calls: tools (policy_aware_tool_node)
          policy check + dispatch    自动做 policy 校验 + 执行 + 计数

Phase 1B-1 起,tools 节点不再用 prebuilt ToolNode,改用
policy_aware_tool_node:工具执行前先按 policy 校验。
"""
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from sla.config import settings
from sla.harness.policy import Policy, policy_aware_tool_node
from sla.harness.prompts import LEARNING_AGENT_SYSTEM


class AgentState(TypedDict):
    """LangGraph state schema。

    - messages: 消息历史。Annotated[list, add_messages] 让节点返回的 messages
      被 append(而不是 overwrite)。
    - policy: 本次 task 的策略对象,policy_aware_tool_node 读它做校验。
    - artifacts_created: 已产出工件计数(note / question_batch ...),
      policy_aware_tool_node 在工具成功执行后更新。无 reducer:dict 默认 overwrite,
      节点每次返回完整新 dict 即可。
    - run_id: 当前 Run 行的 id。
        * runner.py 传入真实 id 时,policy_aware_tool_node 会把 save_note /
          save_questions 的产出写到 Artifact 表;
        * smoke 脚本传 None,跳过 Artifact 写入。
    - document_id: 当前 task 关联的 Document.id。
        * policy_aware_tool_node 在调工具前注入到 tc["args"]["document_id"],
          让 tools 按 document 过滤查询(跨文档同 chapter_id 不混淆)。
        * None 时 tools 不过滤(向后兼容 fixture / 单文档场景)。
    """

    messages: Annotated[list, add_messages]
    policy: Policy
    artifacts_created: dict[str, int]
    run_id: int | None
    document_id: int | None


def build_graph(
    model_name: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    tools=None,
    terminal_tool: str | None = None,
):
    """构造并 compile 一个 agent graph。

    调用方:
      graph = build_graph()
      result = graph.invoke(
          initial_state("ch1.3", Policy(read_scope=["ch1.3"])),
          config={"recursion_limit": policy.max_steps * 2},
      )

    tools=None  → 原 ALL_TOOLS(枚举的 4 个零参调用者字节同行为,零回归)。
                  显式传入 → bind 恰好这些,save_note 等对结构 subagent 从
                  模型 schema【真缺席】(非 runtime 才被 policy 拦)。
    terminal_tool → 非空时,某轮成功调用该工具后结构性 END(单交付即终止,
                  不靠模型自觉停)。None=维持原无条件 tools→agent 回环。
                  注:这是 happy-path 终止;硬兜底仍是调用方传的
                  recursion_limit=max_steps*2(见 runner;policy 不自计步)。
    """
    # 注意:这里不再 import / 用 ToolNode,改由 policy.py 里的
    # policy_aware_tool_node 接管 tools 节点的职责
    if tools is None:
        from sla.harness.tools import ALL_TOOLS
        tools = ALL_TOOLS

    model = ChatAnthropic(
        model=model_name,
        max_tokens=max_tokens,
        api_key=settings.anthropic_api_key,
    )
    bound = model.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """LLM 决策节点:对当前 messages 调一次 bound.invoke。

        返回 {"messages": [response]},LangGraph 用 add_messages reducer
        把 response 追加到 state["messages"]。
        """
        response = bound.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """conditional_edges 判断函数:
          - 最后一条 message 有 tool_calls → 走 tools 节点继续
          - 否则 → END(等价于 Phase 4 的 if not tool_calls: break)
        """
        last = state["messages"][-1]
        return "tools" if last.tool_calls else END

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", policy_aware_tool_node)   # ← Phase 1B-1 替换点

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )

    if terminal_tool is None:
        builder.add_edge("tools", "agent")              # 默认:原无条件回环(零回归)
    else:
        def after_tools(state: AgentState) -> str:
            """本轮 tools 含【成功】terminal_tool(非 denied/error)→ END;否则
            回 agent。经最近 AIMessage 的 tool_calls 关联 name↔id(ToolMessage
            不带 name),不改共享 node,确定性、最小爆炸面。"""
            msgs = state["messages"]
            ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
            if ai is None:
                return "agent"
            target_ids = {
                tc["id"] for tc in (ai.tool_calls or [])
                if tc["name"] == terminal_tool
            }
            if not target_ids:
                return "agent"
            for m in reversed(msgs):
                if isinstance(m, ToolMessage) and m.tool_call_id in target_ids:
                    st = getattr(m, "status", None)
                    if st != "error" and not str(m.content).startswith(
                        "Denied by policy."
                    ):
                        return END
                    return "agent"   # terminal_tool 被 denied/error → 不终止,回 agent
            return "agent"

        builder.add_conditional_edges(
            "tools", after_tools, {END: END, "agent": "agent"}
        )

    return builder.compile()


def initial_state(
    chapter_id: str,
    policy: Policy,
    run_id: int | None = None,
    document_id: int | None = None,
) -> dict:
    """构造一次 task 的初始 state。

    Phase 1B-1 起,initial_state 必须接受 policy 参数。
    Phase 1B-2 起,可选 run_id(决定是否落 Artifact)。
    Phase 2-W2-0 起,可选 document_id(跨文档过滤;None 时不过滤)。
    """
    # SystemMessage.content 用 list-of-blocks 格式 + cache_control,
    # 启用 Anthropic prompt caching:每次 Run 内多轮 / 跨 Run 5min 内 cache 命中
    # 长 prompt(此处 ~600 字符 + tools schema 一起 > 1024 tokens 才会生效)
    return {
        "messages": [
            SystemMessage(content=[{
                "type": "text",
                "text": LEARNING_AGENT_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }]),
            HumanMessage(
                content=f"请阅读 chapter_id={chapter_id} 这一章,完成学习笔记和思考题。"
            ),
        ],
        "policy": policy,
        "artifacts_created": {},
        "run_id": run_id,
        "document_id": document_id,
    }
