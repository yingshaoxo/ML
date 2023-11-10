# Rules for Human General Relationship

## All in all

How you treat others is how others treat you.

## The Algorithm:

1. We create a panda table, each row has 4 properties: name, positive energy, negative energy, positive and negative energy balanced counting.
2. Every time when you give somebody benefits, add 1 to positive energy. (You lose, others gain)
3. Every time when you take something from others, add 1 to negative energy. (You gain, others lose)
4. Each time, if the positive energy equals to negative energy, add 1 to positive and negative energy balance counting. (We made a deal)



* `positive energy == negative energy == 0` , you are strangers to each other.
* `positive energy > negative energy`, try to get help from that guy, he will help you. If he refused to help you, he is an Asshole.
* `negative energy > positive energy`, if others ask you to help, you have to do it.



* `positive energy > negative energy`, `(positive energy - negative energy) > a value that you can endure`: You take revenge. Take what you deserve from others. You can even add that guy to a blacklist.
* `negative energy > positive energy`, `(negative energy - positive energy) > a value that others can endure`: Others take revenge on you. They'll get what they deserve from you. You'll be added to their friendship blacklist.
* You'll be more likely to make a deal with someone who has a higher `balanced counting`. For those people, we would like to call them friends.



```python
import pandas as pd
import os


class Friendship():
    def __init__(self, csv_path=None):
        if csv_path == None:
            self.__df_path = os.path.join(os.getcwd(), 'friendships.csv')
        else:
            self.__df_path = os.path.abspath(csv_path)

        if not os.path.exists(self.__df_path):
            self.df = pd.DataFrame(columns=[
                'name',
                'description',
                'positive_power',
                'negative_power',
                'balanced_times'
            ])
        else:
            self.df = pd.read_csv(self.__df_path)

    def __str__(self):
        return '-'*30 + '\n' + str(self.df.head())

    def add_person(self, name):
        temp_df = self.df.loc[self.df['name'] == name]
        if len(temp_df) >= 1:
            print('You already have it')
        else:
            self.df = self.df.append(
                {
                    'name': name,
                    'description': '',
                    'positive_power': 0,
                    'negative_power': 0,
                    'balanced_times': 0,
                },
                ignore_index=True
            )

    def delete_person(self, name):
        self.df = self.df[self.df['name'] != name]

    def giving(self, name, power):
        temp_df = self.df.loc[self.df['name'] == name]
        if len(temp_df) >= 1:
            new_value = temp_df.iloc[0]['positive_power'] + power
            temp_df.loc[:, 'positive_power'] = new_value
            temp_df = self.__calculate_balanced_times(temp_df)
            self.df.update(temp_df)
        else:
            print(f"{name} does not exist in your friendship list!")

    def taking(self, name, power):
        temp_df = self.df.loc[self.df['name'] == name]
        if len(temp_df) >= 1:
            new_value = temp_df.iloc[0]['negative_power'] + power
            temp_df.loc[:, 'negative_power'] = new_value
            temp_df = self.__calculate_balanced_times(temp_df)
            self.df.update(temp_df)
        else:
            print(f"{name} does not exist in your friendship list!")

    def __calculate_balanced_times(self, single_row_df):
        positive = single_row_df.iloc[0]['positive_power']
        negative = single_row_df.iloc[0]['negative_power']
        balanced_times = single_row_df.iloc[0]['balanced_times']
        if positive == negative:
            single_row_df.loc[:, 'balanced_times'] = balanced_times + 1
        return single_row_df

    def seek_for_help(self, description=''):
        temp_df = self.df.loc[self.df['positive_power'] > self.df['negative_power']]
        temp_df['released_power'] = temp_df['positive_power'] - temp_df['negative_power']
        sorted_df = temp_df.sort_values(['released_power', 'balanced_times'])
        print(sorted_df.head())

    def can_i_help(self, name=None):
        temp_df = self.df.loc[self.df['negative_power'] > self.df['positive_power']]
        temp_df['borrowed_power'] = temp_df['negative_power'] - temp_df['positive_power']
        sorted_df = temp_df.sort_values(['borrowed_power', 'balanced_times'])
        print(sorted_df.head())

    def commit(self):
        self.df.to_csv(self.__df_path, index=False)


if __name__ == "__main__":
    print('\n'*50)
    friendship = Friendship('/home/yingshaoxo/Documents/friendship.csv')
    friendship.add_person('A')
    friendship.add_person('B')
    friendship.giving('A', 5)
    friendship.taking('B', 3)
    friendship.seek_for_help()
    friendship.can_i_help()
    #friendship.commit()
```



> Now you know how to represent the friendship with a math model, it is good. You can then use it to create an AI that has something people called 'EQ'.

> Here the `positive energy` and `negative energy`, they are just variables. In the real-world, energy only has a size, no features like positive or negative.

## 人际关系的利益学算法：

1. 建立一张pandas表，每行4个参数，姓名、正能量、负能量、正负能量达到平衡的交易次数(1:1)
2. 每次你给某个人好处，正能量增加。你付出，别人获利。
3. 每次你损害他人的利益(从别人那儿获取好处)，负能量增加。别人付出，你获利。
4. 每一次操作后，若正负能量达到平衡，则“交易次数”加一



* 正能量 = 负能量 = 0，陌生人
* 正能量 > 负能量，尝试从这个人获取好处，有困难找对方帮忙
* 负能量 > 正能量，对方(以及他的直系亲属、朋友、党羽)找你帮忙，你得帮。甚至还得主动揣测对方需求，主动帮忙



* 正能量 > 负能量，正能量-负能量 > 个人最大承受值，你报复他人(指短时间内，一次性从别人那里获取你认为应得的利益)，你将对方加入黑名单
* 负能量 > 正能量，负能量-正能量 > 他人最大承受值，他人对你采取报复行为(指短时间内，一次性从你那里获取他们认为应得的利益)，被对方加入黑名单
* 你会更倾向于与“交易次数”高的人完成下一次交易，这些人，通常被称为好友



> 注： 这套理论不光可以解释一堆人情世故相关的问题，还可以作为“数学模型”，让机器学会为人处世、拥有“情商”。
>
> 注：正能量或负能量在这里只是两个变量，在真实世界里，能量只有大小之分，没有正负的分别。



* 微笑是向别人示好
* 送礼是向别人示好
* 示好是为了建立联系
* 打招呼、叫名字、拜访送礼是为了表示协议还存在 （我之前讲过朋友关系就是一种协议，上面写着“互帮互助”）

1. 分享知识(包括身边的新闻)，想别人可能会想的，为别人做一点预测，再返回给他； 这可以算作正能量
2. 问问题 或 让别人知道自己的事情，(主动或不主动)请求别人的帮忙; 这个可以算作负能量
3. 嘘寒问暖、让对方感觉他还存在，对大多数人来讲是正能量 (除非对方正在做一些艰难的工作，不喜欢被人打扰。)

## **在这个理论中，重要的部分是：**

1. 把具体的利益事件 分级，转化为能量值 (不同时代、背景，做同一事情的难度不一样
2. 选择合适的连接对象 (就像贷款人筛选，要求你选择能还得起“贷款(人情)”的人(`有潜力`)、愿意还人情的人(注重`友情协议`，苟富贵、不相忘))

这里的第二点，具体体现(延伸)是：

我们(大多数人)喜欢和那些`面部洁净`、`衣着整洁`的人(若是女生，还得有`体香`(其实是香水))的人做朋友(结交)

因为她们更有可能还得起 人情(能量)

因为细皮嫩肉是钱养出来的，香水要花钱，穷人没时间打扮，富人受教育的可能性高

> 我只是按高概率作推理，这些事情不完全按这个逻辑走。但你可以按这个道理进行独自推理。

## 额外的引申1

如果你想与某人建立联系，你总是可以的: if you want, you could

只是要注意展开的原则: 从`疏`到`近`，循序渐进，渐渐地从一个`陌生人`到`熟人`。

最后提醒你一下，不是什么人都可以结交，有的人`人品差`，欠了一堆债，和他结交就是倒霉的开始。同时，不要和你不喜欢的人`做联系`，因为你要对你的`联系(协议)`负责，不能中途而废(`毁坏协议，单方面断交`)，只有这样你才能成为一个世俗意义上的`会做人的人`。

## 额外的引申2

不要轻易与和你思维相差太大的人建立联系。

比如你是理性、科学的思维。而对方是个顽固、不遵从`实验出真知原则`的人，那你们两个人总是会争吵。

> 有些人缺乏科学思维，对事实、实验不屑一顾，不知道认错，这些人没办法取得更大的进步(你都完美了，还提个什么升？)，同时也更可能通过争吵让人们的生活不愉悦。所以生活中要注意远离傻逼。
