def valid_tuple_sizes(entry_size, max_value=64):
    valid_values = []
    for tuple_size in range(3, max_value):
        if entry_size % tuple_size == 0:
            valid_values.append(tuple_size)
    return valid_values
