from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ValidationResult(BaseModel):
    """
    验证结果
    """

    is_sql_valid: bool = Field(description="SQL 查询是否有效且可以安全执行")
    issues: List[str] = Field(default_factory=list, description="SQL 查询已识别问题列表")
    suggested_fix: str = Field(default="", description="如果发现问题，建议修复 SQL 查询")


class EvaluationResult(BaseModel):
    """
    评估结果
    """

    is_result_relevant: bool = Field(description="生成的SQL查询是否与用户查询相关")
    explanation: str = Field(default="", description="解释为什么 SQL 查询相关或不相关")
    improvement_suggestion: str = Field(default="", description="关于如何改进 SQL 查询以提高相关性的讨论")


class SQLCorrectionResult(BaseModel):
    """
    sql修正
    """

    analysis: str = Field(
        default="未提供分析",
        description="分析当前 SQL 查询未产生相关结果的原因",
    )
    identified_issues: List[str] = Field(default="未发现问题", description="当前 SQL 查询中识别的特定问题列表")
    corrected_sql_query: str = Field(default="", description="已更正的SQL，用于解决已识别的问题")


class ExecutionResult(BaseModel):
    """
    sql执行结果
    """

    success: bool
    data: Optional[List[Dict[str, Any]]] = None  # 执行结果
    error: Optional[str] = None


class AgentState(TypedDict):
    """
    Graph/Sate定义
    """

    user_query: str  # 用户问题
    sql_reasoning: Optional[str]  # SQL 推理过程
    db_info: Optional[Dict]  # 数据库信息
    table_relationship: Optional[List[Dict[str, Any]]]  # 表关系
    generated_sql: Optional[str]  # 生成的 SQL
    execution_result: Optional[ExecutionResult]  # SQL 执行结果
    validation_result: Optional[ValidationResult]  # SQL 验证结果
    evaluation_result: Optional[EvaluationResult]  # SQL 评估结果
    correction_result: Optional[SQLCorrectionResult]  # SQL 修正结果
    report_summary: Optional[str]  # 报告摘要
    attempts: int = 0  # 尝试次数
    correct_attempts: int = 0  # 正确尝试次数
    chart_url: Optional[str]  # AntV MCP图表地址
    chart_type: Optional[str]  # 图表类型
    apache_chart_data: Optional[Dict[str, Any]]  # Apache图表数据
