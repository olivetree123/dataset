class Filter(object):
    """
    过滤器
    示例：
    name__in=["Tom", "Alice"]
    age__gte=20
    """

    def __init__(self, column_name, operator, dest_val):
        self.column_name = column_name
        self.operator = operator
        self.dest_val = dest_val

    def is_match(self, src_val) -> bool:
        if self.operator == "gt":
            if src_val > self.dest_val:
                return True
        elif self.operator == "gte":
            if src_val >= self.dest_val:
                return True
        elif self.operator == "lt":
            if src_val < self.dest_val:
                return True
        elif self.operator == "lte":
            if src_val <= self.dest_val:
                return True
        elif self.operator == "eq":
            if src_val == self.dest_val:
                return True
        elif self.operator == "in":
            if src_val in self.dest_val:
                return True
        return False

    @classmethod
    def create(cls, key, value):
        column_name, operator = key.split("__")
        return cls(column_name, operator, value)
