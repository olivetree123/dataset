from typing import List

from entity.filter import Filter


class Instance(object):

    def __init__(self, pk, with_label, label, features):
        self.pk = pk
        self.label = label
        self.with_label = with_label
        self.features = features
        self.columns = []

    def json(self):
        return self.__dict__

    def to_csv(self):
        """转成逗号分隔的字符串"""
        rs = self.to_list()
        return rs

    def to_list(self):
        rs = [self.pk]
        if self.with_label:
            rs.append(self.label)
        rs += self.features
        return rs

    def match_filter(self, filter_list: List[Filter]) -> bool:
        """TODO: 可以过滤特征，也可以过滤id 和 label"""
        for _filter in filter_list:
            src_index = self.columns.index(_filter.column_name) - 2
            src_val = self.features[src_index]
            status = _filter.is_match(src_val)
            if not status:
                return False
        return True

    @classmethod
    def from_line(cls, line, with_label):
        values = line.strip().split(",")
        label = values[1] if with_label else None
        features = values[2:] if with_label else values[1:]
        return cls(pk=values[0],
                   with_label=with_label,
                   label=label,
                   features=features)
