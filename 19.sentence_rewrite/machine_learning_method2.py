#pip install paddlepaddle-gpu
#pip install --upgrade paddlenlp
#conda install -c conda-forge cudnn
#export LD_LIBRARY_PATH="$HOME/anaconda3/lib":$LD_LIBRARY_PATH

# title generation

from paddlenlp import Taskflow
summarizer = Taskflow("text_summarization")

text = "据微信公众号“界面”报道，4日上午10点左右，中国发改委反垄断调查小组突击查访奔驰上海办事处，调取数据材料，并对多名奔驰高管进行了约谈。截止昨日晚9点，包括北京梅赛德斯-奔驰销售服务有限公司东区总经理在内的多名管理人员仍留在上海办公室内"
while True:
    if text.strip() != "":
        result = summarizer(text)
        print()
        print(result)
        # 发改委突查奔驰上海办事处
        print()

    text = input("What you want to say?\n")