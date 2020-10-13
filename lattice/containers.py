__all__ = ['ElementList']


class ElementList(list):
    """
    A special list that can print descriptive info about contained Elements
    Note: inheriting from collections abstract implementations messes with instance checks,
    but should be added at some point
    """

    @property
    def names(self):
        return [el.id for el in super().__iter__()]

    @property
    def elements(self):
        return [(el.id, el.__class__.__name__, el.l) for el in super().__iter__()]
