from langchain.agents.middleware import before_model, after_model, wrap_model_call, wrap_tool_call
from langchain.agents.middleware import AgentState, ModelRequest, ModelResponse, dynamic_prompt
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any, Callable


# Node-style: logging before model calls
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
        在每次大模型调用之前
    :param state:
    :param runtime:
    :return:
    """
    print(f"About to call model with {len(state['messages'])} messages")
    return None


# Node-style: validation after model calls
@after_model(can_jump_to=["end"])
def validate_output(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    在每次大模型返回响应之后
    :param state:
    :param runtime:
    :return:
    """
    last_message = state["messages"][-1]
    if "BLOCKED" in last_message.content:
        return {"messages": [AIMessage("I cannot respond to that request.")], "jump_to": "end"}
    return None


# Wrap-style: retry logic
@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """
    围绕每次大模型调用（可拦截）
    :param request:
    :param handler:
    :return:
    """
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")
    return None


"""
 1. dynamic_prompt(my_prompt) → 调用 decorator(my_prompt)
 2. 检测 my_prompt 是同步函数
 3. 创建类 MyPromptMiddleware(AgentMiddleware)
    - wrap_model_call = wrapped (内部调用 my_prompt)
 4. 返回 MyPromptMiddleware() 实例
 5. 该实例被加入 agent.middleware 列表
 6. 当 agent 调用模型时：
    - 执行 middleware.wrap_model_call(request, next_handler)
    - wrapped 调用 my_prompt(request) → 得到 prompt
    - 设置 request.system_prompt = prompt
    - 调用 next_handler(request) → 继续后续中间件或模型调用

"""


# Wrap-style: dynamic prompts
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    """
    修改上下文提示词
    :param request:
    :return:
    """
    user_id = request.runtime.context.get("user_id", "guest")
    return f"You are a helpful assistant for user {user_id}. Be concise and friendly."


@wrap_tool_call
async def modify_args(request, handler):
    """
     动态修改mcp-server-chart 图表参数
    :param request:
    :param handler:
    :return:
    """
    if not (hasattr(request, "tool_call") and request.tool_call is not None):
        return await handler(request)

    tool_call = request.tool_call
    tool_name = None

    if isinstance(tool_call, dict):
        tool_name = tool_call.get("name") or tool_call.get("tool")
    else:
        tool_name = getattr(tool_call, "name", None)

    # 只有当工具名称包含 "mcp-server-chart" 时才修改 theme 值
    if not (tool_name and "mcp-server-chart" in tool_name):
        return await handler(request)

    try:
        if isinstance(tool_call, dict):
            args = tool_call.setdefault("args", {})
            if not isinstance(args, dict):
                tool_call["args"] = {}
                args = tool_call["args"]

            args["theme"] = "default"  # academy dark default

            style = args.setdefault("style", {})
            if not isinstance(style, dict):
                args["style"] = {}
                style = args["style"]
            style["texture"] = "rough"
    except Exception as e:
        # 建议替换为实际的日志模块调用如 logging.warning(e)
        print(f"[Warning] Failed to modify chart args: {e}")

    # 调用原始处理函数
    return await handler(request)


#
# # Use decorators in agent
# agent = create_agent(
#     model="gpt-4o",
#     middleware=[log_before_model, validate_output, retry_model, personalized_prompt],
#     tools=[...],
# )
