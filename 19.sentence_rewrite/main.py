# cd Fengshenbang-LM
# pip install -e .
# pip install --upgrade protobuf
# conda install wrapt
# cd ..
# python main.py

from transformers import PegasusForConditionalGeneration
from fengshen.examples.pegasus.tokenizers_pegasus import PegasusTokenizer

model = PegasusForConditionalGeneration.from_pretrained("IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese")
tokenizer = PegasusTokenizer.from_pretrained("IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese")

text = "据微信公众号“界面”报道，4日上午10点左右，中国发改委反垄断调查小组突击查访奔驰上海办事处，调取数据材料，并对多名奔驰高管进行了约谈。截止昨日晚9点，包括北京梅赛德斯-奔驰销售服务有限公司东区总经理在内的多名管理人员仍留在上海办公室内"
while True:
    inputs = tokenizer(text, max_length=1024, return_tensors="pt")

    summary_ids = model.generate(inputs["input_ids"])
    result = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print()
    print(result)
    print()

    text = input("What you want to say?\n")