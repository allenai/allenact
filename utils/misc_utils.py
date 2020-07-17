class HashableDict(dict):
    """A dictionary which is hashable so long as all of its values are
    hashable.

    A HashableDict object will allow setting / deleting of items until
    the first time that `__hash__()` is called on it after which
    attempts to set or delete items will throw `RuntimeError`
    exceptions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._hash_has_been_called = False

    def __key(self):
        return tuple((k, self[k]) for k in sorted(self))

    def __hash__(self):
        self._hash_has_been_called = True
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __setitem__(self, *args, **kwargs):
        if not self._hash_has_been_called:
            return super(HashableDict, self).__setitem__(*args, **kwargs)
        raise RuntimeError("Cannot set item in HashableDict after having called hash.")

    def __delitem__(self, *args, **kwargs):
        if not self._hash_has_been_called:
            return super(HashableDict, self).__delitem__(*args, **kwargs)
        raise RuntimeError(
            "Cannot delete item in HashableDict after having called hash."
        )
