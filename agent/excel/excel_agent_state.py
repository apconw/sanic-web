from typing import TypedDict, Optional, Dict, Any, List

from pydantic import BaseModel


class ExecutionResult(BaseModel):
    """
    sql执行结果
    """

    success: bool
    columns: List[str]  # 表格列名
    data: Optional[List[Dict[str, Any]]] = None  # 执行结果
    error: Optional[str] = None


class FileInfo(BaseModel):
    """
    文件信息模型
    """
    file_name: str  # 文件名
    file_path: str  # 文件路径
    catalog_name: str  # DuckDB catalog名称
    sheet_count: int  # Sheet数量
    upload_time: str  # 上传时间


class SheetInfo(BaseModel):
    """
    Sheet信息模型
    """
    sheet_name: str  # Sheet名称
    table_name: str  # 注册的表名
    catalog_name: str  # 所属catalog
    row_count: int  # 行数
    column_count: int  # 列数
    columns_info: Dict[str, Any]  # 列信息
    sample_data: Optional[List[Dict[str, Any]]] = None  # 样本数据


class ExcelAgentState(TypedDict):
    """
    表格问答 - 支持多文件多Sheet统一分析
    """

    user_query: str  # 用户问题
    file_list: list  # 文件列表
    chat_id: Optional[str]  # 聊天ID，用于隔离不同用户的资源
    file_metadata: Dict[str, FileInfo]  # 文件元数据信息 {file_path: FileInfo}
    sheet_metadata: Dict[str, SheetInfo]  # Sheet元数据信息 {table_name: SheetInfo}
    db_info: list[dict]  # 把表格映射成数据库表结构（扩展支持多catalog）
    catalog_info: Dict[str, str]  # Catalog信息 {catalog_name: file_path}
    generated_sql: Optional[str]  # 生成的 SQL
    chart_url: Optional[str]  # AntV MCP图表地址
    chart_type: Optional[str]  # 图表类型
    apache_chart_data: Optional[Dict[str, Any]]  # Apache图表数据
    execution_result: Optional[ExecutionResult]  # SQL 执行结果
    report_summary: Optional[str]  # 报告摘要
