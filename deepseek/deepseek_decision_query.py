#!/usr/bin/env python3
"""
deepseek_decision_query - Query Routing Schema for DeepSeek RAG

Defines the structured output schema used to route user queries to either
the local DeepSeek model or an external API, based on query complexity and
privacy requirements.

Usage:
    from deepseek_decision_query import RouteQuery

    # Use with a LangChain LLM that supports structured output:
    structured_llm = llm.with_structured_output(RouteQuery)
    route = structured_llm.invoke("What is 2+2?")
    print(route.destination)  # "local_model"

Dependencies:
    - langchain-core>=1.1.0

Version: 1.0.0
"""

from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field

# 1. Define the decision structure
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    destination: Literal["local_model", "external_api"] = Field(
        ...,
        description="Given a user question, choose to route it to 'local_model' for simple/private questions or 'external_api' for complex/hard reasoning."
    )
    reasoning: str = Field(..., description="Brief explanation of why you chose this route.")
