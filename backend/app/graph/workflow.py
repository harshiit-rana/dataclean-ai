"""LangGraph StateGraph wiring. PRD §13.1.

    goal_intake -> profiling_agent -> explanation_agent -> human_approval [interrupt]
        -> cleaning_agent -> conversational_loop -> audit_agent -> END

human_approval uses LangGraph's dynamic `interrupt()` call (inside the node itself, see
nodes/human_approval.py) rather than the static `interrupt_before=[...]` compile-time
option, so no special compile-time argument is needed to pause there -- the graph pauses
wherever `interrupt()` is actually called and resumes via `Command(resume=...)`.

The checkpointer is built separately (see graph/checkpointer.py, an async context manager
around Redis) and passed in here at compile time, since its lifecycle is tied to the
FastAPI app's, not the graph module's.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph.nodes.audit_agent import audit_agent
from app.graph.nodes.cleaning_agent import cleaning_agent
from app.graph.nodes.conversational_loop import conversational_loop
from app.graph.nodes.explanation_agent import explanation_agent
from app.graph.nodes.goal_intake import goal_intake
from app.graph.nodes.human_approval import human_approval
from app.graph.nodes.profiling_agent import profiling_agent
from app.models.graph_state import GraphState


def build_graph(checkpointer=None):
    """Builds and compiles the DataClean AI workflow graph.

    `checkpointer` is any LangGraph BaseCheckpointSaver -- pass the AsyncRedisSaver from
    `build_checkpointer()` in production, or `MemorySaver()` in tests (PRD §15.2).
    """
    builder = StateGraph(GraphState)

    builder.add_node("goal_intake", goal_intake)
    builder.add_node("profiling_agent", profiling_agent)
    builder.add_node("explanation_agent", explanation_agent)
    builder.add_node("human_approval", human_approval)
    builder.add_node("cleaning_agent", cleaning_agent)
    builder.add_node("conversational_loop", conversational_loop)
    builder.add_node("audit_agent", audit_agent)

    builder.set_entry_point("goal_intake")
    builder.add_edge("goal_intake", "profiling_agent")
    builder.add_edge("profiling_agent", "explanation_agent")
    builder.add_edge("explanation_agent", "human_approval")
    builder.add_edge("human_approval", "cleaning_agent")
    builder.add_edge("cleaning_agent", "conversational_loop")
    builder.add_edge("conversational_loop", "audit_agent")
    builder.add_edge("audit_agent", END)

    return builder.compile(checkpointer=checkpointer)
