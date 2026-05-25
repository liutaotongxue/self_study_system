"""LangGraph harness: rewrites Phase 4's while loop using StateGraph.

Semantics are identical to Phase 4; only the representation changes:
  Phase 4 while loop               LangGraph
  -----------------------------    ------------------------------
  while True:                      START -> agent (node)
      response = bound.invoke(...)   agent_node calls bound.invoke
      messages.append(response)      add_messages reducer appends automatically
      if not response.tool_calls:    conditional_edges -> END
          break
      for tc in response.tool_calls: tools (policy_aware_tool_node)
          policy check + dispatch    runs policy check + execution + counting

Starting in Phase 1B-1, the tools node no longer uses prebuilt ToolNode; it uses
policy_aware_tool_node, which policy-checks before tool execution.
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
    """LangGraph state schema.

    - messages: message history. Annotated[list, add_messages] makes the messages returned by
      a node be appended (rather than overwritten).
    - policy: this task's policy object; policy_aware_tool_node reads it to perform checks.
    - artifacts_created: count of produced artifacts (note / question_batch ...),
      updated by policy_aware_tool_node after a successful tool execution. No reducer: dicts default
      to overwrite, so each node returns a complete new dict.
    - run_id: id of the current Run row.
        * When runner.py passes a real id, policy_aware_tool_node writes save_note /
          save_questions outputs into the Artifact table;
        * When smoke scripts pass None, Artifact writing is skipped.
    - document_id: Document.id associated with the current task.
        * policy_aware_tool_node injects this into tc["args"]["document_id"] before invoking a tool,
          letting tools filter queries by document (no cross-document chapter_id collisions).
        * When None, tools do not filter (backward compatible for fixture / single-document scenarios).
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
    """Build and compile an agent graph.

    Caller:
      graph = build_graph()
      result = graph.invoke(
          initial_state("ch1.3", Policy(read_scope=["ch1.3"])),
          config={"recursion_limit": policy.max_steps * 2},
      )

    tools=None  -> the original ALL_TOOLS (the 4 enumerated zero-arg callers are byte-identical, zero regression).
                   Explicit pass -> bind exactly these; save_note etc. are truly ABSENT from the
                   structure subagent's model schema (not blocked at runtime by policy).
    terminal_tool -> when non-None, the graph structurally ENDs after a successful invocation of that tool
                   (terminates on single delivery, does not rely on the model self-stopping). None = keep the
                   original unconditional tools -> agent loop.
                   Note: this is happy-path termination; the hard backstop is still the caller-passed
                   recursion_limit=max_steps*2 (see runner; policy does not count steps itself).
    """
    # Note: no longer imports / uses ToolNode; policy.py's policy_aware_tool_node now
    # takes over the responsibilities of the tools node
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
        """LLM decision node: invokes bound.invoke once on the current messages.

        Returns {"messages": [response]}; LangGraph uses the add_messages reducer
        to append response to state["messages"].
        """
        response = bound.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """conditional_edges decision function:
          - Last message has tool_calls -> go to the tools node
          - Otherwise -> END (equivalent to Phase 4's `if not tool_calls: break`)
        """
        last = state["messages"][-1]
        return "tools" if last.tool_calls else END

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", policy_aware_tool_node)   # <- Phase 1B-1 replacement point

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )

    if terminal_tool is None:
        builder.add_edge("tools", "agent")              # Default: original unconditional loop (zero regression)
    else:
        def after_tools(state: AgentState) -> str:
            """If this round's tools include a [successful] terminal_tool call (not denied/error) -> END;
            otherwise return to agent. Correlates name<->id via the most recent AIMessage's tool_calls
            (ToolMessage carries no name); does not modify the shared node; deterministic, minimal blast radius."""
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
                    return "agent"   # terminal_tool was denied/error -> do not terminate, return to agent
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
    """Build the initial state for a single task.

    Starting in Phase 1B-1, initial_state must accept a policy parameter.
    Starting in Phase 1B-2, optionally accepts run_id (decides whether to persist Artifacts).
    Starting in Phase 2-W2-0, optionally accepts document_id (cross-document filtering; None means no filter).
    """
    # SystemMessage.content uses list-of-blocks format + cache_control to enable Anthropic prompt caching:
    # cache hits across turns within a Run and across Runs within a 5-min window for long prompts
    # (this prompt of ~600 chars + tools schema must together exceed 1024 tokens before caching activates)
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
