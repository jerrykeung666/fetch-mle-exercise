def int2date(year, date_index):
    if date_index < 31:
        return f"{year}-01-{date_index+1}"
    elif date_index < 59:
        return f"{year}-02-{date_index-30}"
    elif date_index < 90:
        return f"{year}-03-{date_index-58}"
    elif date_index < 120:
        return f"{year}-04-{date_index-89}"
    elif date_index < 151:
        return f"{year}-05-{date_index-119}"
    elif date_index < 181:
        return f"{year}-06-{date_index-150}"
    elif date_index < 212:
        return f"{year}-07-{date_index-180}"
    elif date_index < 243:
        return f"{year}-08-{date_index-211}"
    elif date_index < 273:
        return f"{year}-09-{date_index-242}"
    elif date_index < 304:
        return f"{year}-10-{date_index-272}"
    elif date_index < 334:
        return f"{year}-11-{date_index-303}"
    else:
        return f"{year}-12-{date_index-333}"