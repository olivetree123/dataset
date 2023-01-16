# 测试clickhouse 和 csv文件的读取速度，谁更快
# clickhouse文档：https://clickhouse.com/docs/en/integrations/language-clients/python/intro/

import clickhouse_connect

client = clickhouse_connect.get_client(host="192.168.10.31",
                                       port=8123,
                                       username="wefe",
                                       password="wefe2020")
