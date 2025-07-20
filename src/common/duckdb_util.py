import duckdb
import pandas as pd

if __name__ == "__main__":
    """
    查询excel数据表
    """
    df = pd.read_excel("data.xlsx", engine="openpyxl")
    con = duckdb.connect(database=":memory")
    con.register("excel_table", df)

    sql = """
        SELECT * FROM excel_table where 产地='中国'
    """

    result = con.execute(sql).fetchall()
    print(result)
