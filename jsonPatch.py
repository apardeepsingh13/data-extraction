from graph import *
import logging
from IPython.display import Image, display
from pydantic import BaseModel, Field, field_validator
from graph import *
from graph import _bind_validator_with_retries


logger = logging.getLogger("extraction")


def bind_validator_with_jsonpatch_retries(
    llm: BaseChatModel,
    *,
    tools: list,
    tool_choice: Optional[str] = None,
    max_attempts: int = 3,
) -> Runnable[Union[List[AnyMessage], PromptValue], AIMessage]:
    """Binds validators + retry logic ensure validity of generated tool calls.

    This method is similar to `bind_validator_with_retries`, but uses JSONPatch to correct
    validation errors caused by passing in incorrect or incomplete parameters in a previous
    tool call. This method requires the 'jsonpatch' library to be installed.

    Using patch-based function healing can be more efficient than repopulating the entire
    tool call from scratch, and it can be an easier task for the LLM to perform, since it typically
    only requires a few small changes to the existing tool call.

    Args:
        llm (Runnable): The llm that will generate the initial messages (and optionally fallba)
        tools (list): The tools to bind to the LLM.
        tool_choice (Optional[str]): The tool choice to use.
        max_attempts (int): The number of attempts to make.

    Returns:
        Runnable: A runnable that can be invoked with a list of messages and returns a single AI message.
    """

    try:
        import jsonpatch  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The 'jsonpatch' library is required for JSONPatch-based retries."
        )

    class JsonPatch(BaseModel):
        """A JSON Patch document represents an operation to be performed on a JSON document.

        Note that the op and path are ALWAYS required. Value is required for ALL operations except 'remove'.
        Examples:

        \`\`\`json
        {"op": "add", "path": "/a/b/c", "patch_value": 1}
        {"op": "replace", "path": "/a/b/c", "patch_value": 2}
        {"op": "remove", "path": "/a/b/c"}
        \`\`\`
        """

        op: Literal["add", "remove", "replace"] = Field(
            ...,
            description="The operation to be performed. Must be one of 'add', 'remove', 'replace'.",
        )
        path: str = Field(
            ...,
            description="A JSON Pointer path that references a location within the target document where the operation is performed.",
        )
        value: Any = Field(
            ...,
            description="The value to be used within the operation. REQUIRED for 'add', 'replace', and 'test' operations.",
        )

    class PatchFunctionParameters(BaseModel):
        """Respond with all JSONPatch operation to correct validation errors caused by passing in incorrect or incomplete parameters in a previous tool call."""

        tool_call_id: str = Field(
            ...,
            description="The ID of the original tool call that generated the error. Must NOT be an ID of a PatchFunctionParameters tool call.",
        )
        reasoning: str = Field(
            ...,
            description="Think step-by-step, listing each validation error and the"
            " JSONPatch operation needed to correct it. "
            "Cite the fields in the JSONSchema you referenced in developing this plan.",
        )
        patches: list[JsonPatch] = Field(
            ...,
            description="A list of JSONPatch operations to be applied to the previous tool call's response.",
        )

    bound_llm = llm.bind_tools(tools, tool_choice=tool_choice)
    fallback_llm = llm.bind_tools([PatchFunctionParameters])

    def aggregate_messages(messages: Sequence[AnyMessage]) -> AIMessage:
        # Get all the AI messages and apply json patches
        resolved_tool_calls: Dict[Union[str, None], ToolCall] = {}
        content: Union[str, List[Union[str, dict]]] = ""
        for m in messages:
            if m.type != "ai":
                continue
            if not content:
                content = m.content
            for tc in m.tool_calls:
                if tc["name"] == PatchFunctionParameters.__name__:
                    tcid = tc["args"]["tool_call_id"]
                    if tcid not in resolved_tool_calls:
                        logger.debug(
                            f"JsonPatch tool call ID {tc['args']['tool_call_id']} not found."
                            f"Valid tool call IDs: {list(resolved_tool_calls.keys())}"
                        )
                        tcid = next(iter(resolved_tool_calls.keys()), None)
                    orig_tool_call = resolved_tool_calls[tcid]
                    current_args = orig_tool_call["args"]
                    patches = tc["args"].get("patches") or []
                    orig_tool_call["args"] = jsonpatch.apply_patch(
                        current_args,
                        patches,
                    )
                    orig_tool_call["id"] = tc["id"]
                else:
                    resolved_tool_calls[tc["id"]] = tc.copy()
        return AIMessage(
            content=content,
            tool_calls=list(resolved_tool_calls.values()),
        )

    def format_exception(error: BaseException, call: ToolCall, schema: Type[BaseModel]):
        return (
            f"Error:\n\n\`\`\`\n{repr(error)}\n\`\`\`\n"
            "Expected Parameter Schema:\n\n" + f"\`\`\`json\n{schema.schema_json()}\n\`\`\`\n"
            f"Please respond with a JSONPatch to correct the error for tool_call_id=[{call['id']}]."
        )

    validator = ValidationNode(
        tools + [PatchFunctionParameters],
        format_error=format_exception,
    )
    retry_strategy = RetryStrategy(
        max_attempts=max_attempts,
        fallback=fallback_llm,
        aggregate_messages=aggregate_messages,
    )
    return _bind_validator_with_retries(
        bound_llm,
        validator=validator,
        retry_strategy=retry_strategy,
        tool_choice=tool_choice,
    ).with_config(metadata={"retry_strategy": "jsonpatch"})

bound_llm = bind_validator_with_jsonpatch_retries(llm, tools=tools)


try:
    display(Image(bound_llm.get_graph().draw_mermaid_png()))
except Exception:
    pass


chain = prompt | bound_llm
results = chain.invoke(
    {
        "messages": [
            (
                "user",
                f"Extract the summary from the following conversation:\n\n<convo>\n{formatted}\n</convo>",
            ),
        ]
    },
)
results.pretty_print()