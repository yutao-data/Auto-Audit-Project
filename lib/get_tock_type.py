def get_stock_type(stock_code):
    """判断股票ID对应的证券市场
    匹配规则
    ['50', '51', '60', '90', '110'] 为 SH
    ['00', '13', '18', '15', '16', '18', '20', '30', '39', '115'] 为 SZ
    ['5', '6', '9'] 开头的为 SH， 其余为 SZ"""

    if stock_code.startswith(
        ("50", "51", "60", "90", "110", "113", "132", "204")
    ):
        return "SH"
    if stock_code.startswith(
        ("00", "13", "18", "15", "16", "18", "20", "30", "39", "115", "1318")
    ):
        return "SZ"
    if stock_code.startswith(("5", "6", "9", "7")):
        return "SH"
    return "SZ"
