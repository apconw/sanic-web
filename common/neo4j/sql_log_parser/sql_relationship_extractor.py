#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SQL关系提取器
从SQL语句中提取表之间的关系
"""

import re
from typing import List, Dict, Set, Tuple, Optional


class SQLRelationshipExtractor:
    """SQL关系提取器"""

    def __init__(self):
        self.tables = set()
        self.relationships = []

    def extract_from_sql(self, sql: str, source: str = "unknown", sql_id: str = "") -> List[Dict]:
        """
        从SQL语句中提取表关系
        :param sql: SQL语句
        :param source: SQL来源（如文件名、数据库名等）
        :param sql_id: SQL标识符
        :return: 关系列表
        """
        relationships = []
        
        # 清理SQL
        sql_clean = self._clean_sql(sql)
        
        # 提取所有表名
        tables = self._extract_tables_from_sql(sql_clean)
        self.tables.update(tables)
        
        if len(tables) < 2:
            # 单个表的SQL，无法提取关系
            return relationships
        
        # 提取JOIN关系
        join_rels = self._extract_join_relationships(sql_clean, tables)
        relationships.extend(join_rels)
        
        # 提取WHERE子句中的关系
        where_rels = self._extract_where_relationships(sql_clean, tables)
        relationships.extend(where_rels)
        
        # 提取子查询中的关系
        subquery_rels = self._extract_subquery_relationships(sql_clean, tables)
        relationships.extend(subquery_rels)
        
        # 提取UNION关系
        union_rels = self._extract_union_relationships(sql_clean, tables)
        relationships.extend(union_rels)
        
        # 为每个关系添加元数据
        for rel in relationships:
            rel['source'] = source
            rel['sql_id'] = sql_id
            rel['sql_preview'] = sql_clean[:200] if len(sql_clean) > 200 else sql_clean
            
        return relationships

    def _clean_sql(self, sql: str) -> str:
        """
        清理SQL语句
        :param sql: 原始SQL
        :return: 清理后的SQL
        """
        # 移除单行注释
        sql = re.sub(r'--.*?$', '', sql, flags=re.MULTILINE)
        # 移除多行注释
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        # 移除多余空格和换行
        sql = re.sub(r'\s+', ' ', sql)
        # 移除字符串字面量（避免字符串中的关键词干扰）
        sql = re.sub(r"'[^']*'", "'?'", sql)
        sql = re.sub(r'"[^"]*"', '"?"', sql)
        return sql.strip()

    def _extract_tables_from_sql(self, sql: str) -> Set[str]:
        """
        从SQL中提取所有表名
        :param sql: SQL语句
        :return: 表名集合
        """
        tables = set()
        sql_upper = sql.upper()
        
        # FROM子句中的表
        from_pattern = r'FROM\s+(?:`?([a-zA-Z_][a-zA-Z0-9_]*)`?)(?:\s+AS\s+\w+)?'
        from_matches = re.findall(from_pattern, sql_upper, re.IGNORECASE)
        tables.update([t.lower() for t in from_matches if t])
        
        # JOIN子句中的表
        join_pattern = r'(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\s+(?:`?([a-zA-Z_][a-zA-Z0-9_]*)`?)(?:\s+AS\s+\w+)?'
        join_matches = re.findall(join_pattern, sql_upper, re.IGNORECASE)
        tables.update([t.lower() for t in join_matches if t])
        
        # INSERT INTO中的表
        insert_pattern = r'INSERT\s+(?:INTO\s+)?(?:`?([a-zA-Z_][a-zA-Z0-9_]*)`?)'
        insert_matches = re.findall(insert_pattern, sql_upper, re.IGNORECASE)
        tables.update([t.lower() for t in insert_matches if t])
        
        # UPDATE中的表
        update_pattern = r'UPDATE\s+(?:`?([a-zA-Z_][a-zA-Z0-9_]*)`?)'
        update_matches = re.findall(update_pattern, sql_upper, re.IGNORECASE)
        tables.update([t.lower() for t in update_matches if t])
        
        # DELETE FROM中的表
        delete_pattern = r'DELETE\s+FROM\s+(?:`?([a-zA-Z_][a-zA-Z0-9_]*)`?)'
        delete_matches = re.findall(delete_pattern, sql_upper, re.IGNORECASE)
        tables.update([t.lower() for t in delete_matches if t])
        
        # 移除空字符串
        tables.discard('')
        
        return tables

    def _extract_join_relationships(self, sql: str, tables: Set[str]) -> List[Dict]:
        """
        从JOIN语句中提取表关系
        :param sql: SQL语句
        :param tables: 表名集合
        :return: 关系列表
        """
        relationships = []
        
        # 匹配各种JOIN语句
        # 格式: [INNER|LEFT|RIGHT|FULL] JOIN table2 [AS alias] ON table1.field1 = table2.field2
        join_pattern = r'(?i)((?:INNER|LEFT|RIGHT|FULL|CROSS)\s+)?JOIN\s+(?:`?([a-zA-Z_][a-zA-Z0-9_]*)`?)(?:\s+AS\s+([a-zA-Z_][a-zA-Z0-9_]*))?\s+ON\s+([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)'
        
        matches = re.findall(join_pattern, sql)
        
        for match in matches:
            join_type, table2, alias, left_field, right_field = match
            
            # 解析字段
            left_parts = left_field.split('.')
            right_parts = right_field.split('.')
            
            if len(left_parts) == 2 and len(right_parts) == 2:
                table1_ref = left_parts[0].lower()
                field1 = left_parts[1].lower()
                table2_ref = right_parts[0].lower()
                field2 = right_parts[1].lower()
                
                # 确定实际表名
                table1_name = self._resolve_table_name(table1_ref, tables)
                table2_name = self._resolve_table_name(table2_ref, tables) or table2.lower()
                
                if table1_name and table2_name and table1_name != table2_name:
                    join_type_clean = (join_type.strip().upper() if join_type else 'INNER') + ' JOIN'
                    
                    relationship = {
                        'from_table': table1_name,
                        'to_table': table2_name,
                        'description': f"{table1_name} {join_type_clean} {table2_name}",
                        'field_relation': f"{field1} = {field2}",
                        'join_type': join_type_clean,
                        'relation_type': 'JOIN'
                    }
                    relationships.append(relationship)
        
        return relationships

    def _extract_where_relationships(self, sql: str, tables: Set[str]) -> List[Dict]:
        """
        从WHERE子句中提取表关系
        :param sql: SQL语句
        :param tables: 表名集合
        :return: 关系列表
        """
        relationships = []
        
        # 匹配WHERE子句中的表关联
        # 格式: WHERE table1.field1 = table2.field2
        where_pattern = r'(?i)WHERE\s+.*?([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)'
        
        matches = re.findall(where_pattern, sql)
        
        for match in matches:
            left_field, right_field = match
            
            left_parts = left_field.split('.')
            right_parts = right_field.split('.')
            
            if len(left_parts) == 2 and len(right_parts) == 2:
                table1_ref = left_parts[0].lower()
                field1 = left_parts[1].lower()
                table2_ref = right_parts[0].lower()
                field2 = right_parts[1].lower()
                
                table1_name = self._resolve_table_name(table1_ref, tables)
                table2_name = self._resolve_table_name(table2_ref, tables)
                
                if table1_name and table2_name and table1_name != table2_name:
                    relationship = {
                        'from_table': table1_name,
                        'to_table': table2_name,
                        'description': f"{table1_name} references {table2_name}",
                        'field_relation': f"{field1} = {field2}",
                        'join_type': 'WHERE',
                        'relation_type': 'WHERE'
                    }
                    relationships.append(relationship)
        
        return relationships

    def _extract_subquery_relationships(self, sql: str, tables: Set[str]) -> List[Dict]:
        """
        从子查询中提取表关系
        :param sql: SQL语句
        :param tables: 表名集合
        :return: 关系列表
        """
        relationships = []
        
        # 匹配子查询中的IN/EXISTS关系
        # 格式: WHERE field IN (SELECT field FROM table2 WHERE ...)
        subquery_pattern = r'(?i)([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)\s+(?:IN|EXISTS)\s*\(\s*SELECT\s+.*?\s+FROM\s+(?:`?([a-zA-Z_][a-zA-Z0-9_]*)`?)'
        
        matches = re.findall(subquery_pattern, sql)
        
        for match in matches:
            left_field, table2 = match
            
            left_parts = left_field.split('.')
            if len(left_parts) == 2:
                table1_ref = left_parts[0].lower()
                field1 = left_parts[1].lower()
                table2_name = table2.lower()
                
                table1_name = self._resolve_table_name(table1_ref, tables)
                
                if table1_name and table2_name and table1_name != table2_name:
                    relationship = {
                        'from_table': table1_name,
                        'to_table': table2_name,
                        'description': f"{table1_name} references {table2_name} (subquery)",
                        'field_relation': f"{field1} IN/EXISTS subquery",
                        'join_type': 'SUBQUERY',
                        'relation_type': 'SUBQUERY'
                    }
                    relationships.append(relationship)
        
        return relationships

    def _extract_union_relationships(self, sql: str, tables: Set[str]) -> List[Dict]:
        """
        从UNION语句中提取表关系
        :param sql: SQL语句
        :param tables: 表名集合
        :return: 关系列表
        """
        relationships = []
        
        # UNION语句通常表示表结构相似，可以建立关系
        if 'UNION' in sql.upper():
            table_list = list(tables)
            for i in range(len(table_list) - 1):
                relationship = {
                    'from_table': table_list[i],
                    'to_table': table_list[i + 1],
                    'description': f"{table_list[i]} UNION {table_list[i + 1]}",
                    'field_relation': 'UNION compatible',
                    'join_type': 'UNION',
                    'relation_type': 'UNION'
                }
                relationships.append(relationship)
        
        return relationships

    def _resolve_table_name(self, table_ref: str, tables: Set[str]) -> Optional[str]:
        """
        解析表名（处理别名）
        :param table_ref: 表引用（可能是别名）
        :param tables: 已知表名集合
        :return: 实际表名
        """
        table_ref_lower = table_ref.lower()
        
        # 如果直接匹配，返回
        if table_ref_lower in tables:
            return table_ref_lower
        
        # 尝试匹配表名的首字母缩写
        for table in tables:
            if table.startswith(table_ref_lower):
                return table
            
            # 检查首字母
            initials = ''.join([word[0] for word in table.split('_') if word])
            if initials == table_ref_lower:
                return table
        
        # 如果找不到，返回原始引用（可能是别名）
        return table_ref_lower

    def get_tables(self) -> Set[str]:
        """获取所有提取的表名"""
        return self.tables

    def get_relationships(self) -> List[Dict]:
        """获取所有提取的关系"""
        return self.relationships

    def deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """
        去除重复的关系
        :param relationships: 关系列表
        :return: 去重后的关系列表
        """
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            # 创建关系的唯一标识
            key = (rel['from_table'], rel['to_table'], rel.get('field_relation', ''))
            
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships

