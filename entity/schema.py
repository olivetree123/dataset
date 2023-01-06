from errors import DatasetError


class Schema(object):

    def __init__(self, id_name, with_label, label_name, features: list):
        self.id_name = id_name
        self.with_label = with_label
        self.label_name = label_name
        self.features = features

    @property
    def id_index(self):
        return 0

    @property
    def label_index(self):
        index = 1 if self.with_label else -1
        return index

    def get_columns(self):
        columns = [self.id_name]
        if self.with_label:
            columns.append(self.label_name)
        columns += self.features
        return columns

    @classmethod
    def default(cls, id_name="id", label_name=None):
        return cls(id_name=id_name,
                   with_label=False,
                   label_name=label_name,
                   features=[])

    @classmethod
    def from_columns(cls, id_name, label_name, columns: list):
        if not columns:
            return cls.default(id_name=id_name, label_name=label_name)
        if id_name not in columns:
            raise DatasetError(f"primary key column '{id_name}' not found")
        if columns.index(id_name) != 0:
            raise DatasetError(
                f"primary key column should be on the first column")

        with_label = True if label_name in columns else False
        if with_label and columns.index(label_name) != 1:
            raise DatasetError(f"label column should be on the second column")

        features = columns[2:] if with_label else columns[2:]
        return cls(id_name=id_name,
                   with_label=with_label,
                   label_name=label_name,
                   features=features)

    def json(self):
        return self.__dict__
