def format_i(string: str):
    string = str(string)
    if len(string) == 1:
        new = "00" + string
    elif len(string) == 2:
        new = "0" + string
    else:
        new = string
    return new
