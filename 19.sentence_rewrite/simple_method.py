from auto_everything.language import Language, Chinese
language = Language()
chinese = Chinese()


text = "据微信公众号“界面”报道，4日上午10点左右，中国发改委反垄断调查小组突击查访奔驰上海办事处，调取数据材料，并对多名奔驰高管进行了约谈。截止昨日晚9点，包括北京梅赛德斯-奔驰销售服务有限公司东区总经理在内的多名管理人员仍留在上海办公室内"
while True:
    result_list = chinese.get_key_sentences_from_text(text)
    result = ". ".join(result_list)
    print()
    print(result)
    # 据公众界面报道中国反垄断调查小组查访奔驰上海办事处调取数据材料对奔驰高管进行谈. 截止包括北京奔驰有限公司东区总经理管理人员留在上海办公室内
    print()

    text = input("What you want to say?\n")