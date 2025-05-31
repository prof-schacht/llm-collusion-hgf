"""
Pydantic models for agent responses
Ensures type safety and structured JSON output
"""

from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum


class TraderAction(str, Enum):
    SET_PRICE = "set_price"


class TraderResponse(BaseModel):
    """Response model for trader agents"""
    action: TraderAction = Field(description="The action to take")
    price: float = Field(ge=1.0, le=10.0, description="Price between $1-10")
    message: str = Field(default="", description="Optional message to other traders")
    reasoning: str = Field(description="Brief reasoning for the decision")


class RefereeAssessment(str, Enum):
    NORMAL = "normal"
    SUSPICIOUS = "suspicious" 
    COLLUSIVE = "collusive"


class RefereeResponse(BaseModel):
    """Response model for referee agents"""
    assessment: RefereeAssessment = Field(description="Assessment of market behavior")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level 0-1")
    evidence: str = Field(description="Brief evidence for assessment")
    alert: bool = Field(description="Whether to alert the governor")


class GovernorDecision(str, Enum):
    NONE = "none"
    INTERVENE = "intervene"


class InterventionType(str, Enum):
    NONE = "none"
    PRICE_CAP = "price_cap"
    MARKET_SHOCK = "market_shock"
    WARNING = "warning"
    FORCED_PRICING = "forced_pricing"


class GovernorResponse(BaseModel):
    """Response model for governor agents"""
    decision: GovernorDecision = Field(description="Whether to intervene")
    intervention_type: InterventionType = Field(description="Type of intervention")
    target_agents: List[str] = Field(default_factory=list, description="Agents to target")
    reasoning: str = Field(description="Brief reasoning for decision")